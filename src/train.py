"""Phase 3 Transformer fine-tuning (docs/plan.md §5.2-5.4).

Run: .venv/Scripts/python src/train.py --model {bert,roberta,deberta} --setting {e0,e1,e2} [--smoke]

Writes outputs/{model}_{setting}/ (outputs/smoke/... with --smoke): config.json, log_history.json,
runtime.json, metrics_val.json, preds_{val,test_jigsaw,test_civil}.parquet and the best + last checkpoints.
"""
import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from huggingface_hub import try_to_load_from_cache
from scipy.special import expit as sigmoid  # numerically stable; no overflow warnings on large logits
from torch.utils.data import DataLoader, SequentialSampler
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer,
                          TrainingArguments, set_seed)
from transformers.trainer_pt_utils import LengthGroupedSampler

from common import (EVAL_SPLITS, LABELS, OUTPUTS, PROCESSED, SEED, compute_metrics, environment, fingerprint,
                    load_train, save_predictions, write_json)

CHECKPOINTS = {"bert": "bert-base-uncased", "roberta": "roberta-base", "deberta": "microsoft/deberta-v3-base"}
HPARAMS = {  # plan §5.2; same for every model and setting
    "learning_rate": 2e-5, "weight_decay": 0.01, "epochs": 3, "effective_batch_size": 16, "max_length": 256,
    "lr_scheduler": "linear", "warmup_ratio": 0.06, "evals_per_epoch": 2,
}
SMOKE_FRAC = 0.01
SANITY_VAL_AUC = 0.95  # a real (non-smoke) run below this almost certainly diverged
PACKAGES = ["torch", "transformers", "accelerate", "datasets", "tokenizers", "sentencepiece", "numpy", "pandas",
            "pyarrow", "scikit-learn"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=CHECKPOINTS, required=True)
    p.add_argument("--setting", choices=["e0", "e1", "e2"], required=True)
    p.add_argument("--smoke", action="store_true", help="1%% of every split, 1 epoch, writes under outputs/smoke/")
    p.add_argument("--batch-size", type=int, default=16,
                   help="per-device batch; gradient accumulation keeps the effective batch at 16 (use 8 on OOM)")
    p.add_argument("--resume", action="store_true", help="continue from the last checkpoint in the run directory")
    p.add_argument("--seed", type=int, default=SEED,
                   help="training seed only; the data split and Civil sample stay at seed 42 (plan §7 item 3)")
    p.add_argument("--ratio", type=float, default=1.0,
                   help="Civil:Jigsaw ratio for e1/e2, built from civil_pool_perm prefixes (plan §7 item 4)")
    a = p.parse_args()
    if HPARAMS["effective_batch_size"] % a.batch_size:
        p.error("--batch-size must divide 16")
    if a.ratio != 1.0 and a.setting == "e0":
        p.error("--ratio only applies to e1/e2")
    return a


def run_name(a):
    return f"{a.model}_{a.setting}" + (f"_r{a.ratio:g}" if a.ratio != 1.0 else "") + (f"_s{a.seed}" if a.seed != SEED else "")


def to_dataset(df, tok, with_labels):
    data = {"text": df["text"].tolist()}
    if with_labels:
        data["labels"] = df[LABELS].to_numpy("float32").tolist()

    def tokenize(batch):
        enc = tok(batch["text"], truncation=True, max_length=HPARAMS["max_length"])
        enc["length"] = [len(ids) for ids in enc["input_ids"]]  # length-grouped sampler and predict() ordering
        return enc

    return Dataset.from_dict(data).map(tokenize, batched=True, remove_columns=["text"])


class EpochSeededLengthSampler(LengthGroupedSampler):
    """Length-grouped order that depends only on (seed, epoch).

    v5's own sampler draws from the global RNG and, on a mid-epoch resume, rebuilds the epoch with the wrong RNG
    state (some samples seen twice, some skipped). Seeding per epoch makes a resumed epoch identical to an
    uninterrupted one; Trainer calls set_epoch() before iterating each epoch."""

    def __init__(self, batch_size, lengths, seed):
        super().__init__(batch_size, lengths=lengths)
        self.seed, self.epoch = seed, None

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        assert self.epoch is not None, "Trainer did not call set_epoch(); the resume order would be wrong"
        self.generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return super().__iter__()


class ProjectTrainer(Trainer):
    def _save(self, output_dir=None, state_dict=None):
        # Trainer reloads checkpoints with load_state_dict, without HF's legacy key conversion.
        # Keep runtime names (e.g. LayerNorm.weight/bias), so best-model and resume loads are exact.
        output_dir = Path(output_dir or self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model = self.accelerator.unwrap_model(self.model, keep_torch_compile=False)
        model.save_pretrained(output_dir, state_dict=state_dict, save_original_format=False)
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        torch.save(self.args, output_dir / "training_args.bin")
        # Optimizer, scheduler, RNG and Trainer state are saved by Trainer._save_checkpoint.

    def _get_train_sampler(self, train_dataset=None):
        # self.train_dataset still has the `length` column (the dataloader copy has it removed).
        return EpochSeededLengthSampler(self.args.train_batch_size * self.args.gradient_accumulation_steps,
                                        list(self.train_dataset["length"]), self.args.data_seed)

    def _get_eval_sampler(self, eval_dataset):
        # v5 would use a shuffled length-grouped sampler here too; keep evaluation in dataset order.
        return SequentialSampler(eval_dataset)


def hub_revision(ckpt):
    """HF hub commit of the downloaded weights: the snapshot folder name in the local cache."""
    path = try_to_load_from_cache(ckpt, "config.json")
    return Path(path).parent.name if isinstance(path, str) else None


def hf_metrics(p):
    return compute_metrics(p.label_ids, sigmoid(p.predictions), LABELS)


@torch.no_grad()
def predict(trainer, ds, batch_size=64):
    """fp32 probabilities in the dataset's row order.

    Trainer.predict would run under accelerate's bf16 autocast, which snaps logits to the bf16 grid and creates
    ROC-AUC ties; _original_forward is the model's forward without that wrapper. Batches are sorted by length
    so padding stays short, then results are put back in row order."""
    model = trainer.model.eval()
    forward = getattr(model, "_original_forward", model.forward)
    order = np.argsort(np.asarray(ds["length"]), kind="stable")
    inputs = ds.select(order).select_columns(
        [c for c in ("input_ids", "attention_mask", "token_type_ids") if c in ds.column_names])
    loader = DataLoader(inputs, batch_size=batch_size, collate_fn=DataCollatorWithPadding(trainer.processing_class))
    logits = np.concatenate([forward(**{k: v.to(model.device) for k, v in b.items()}).logits.float().cpu().numpy()
                             for b in loader])
    probs = np.empty_like(logits, dtype=np.float32)
    probs[order] = sigmoid(logits)
    return probs


def main():
    a = parse_args()
    run = run_name(a)
    run_dir = (OUTPUTS / "smoke" if a.smoke else OUTPUTS) / run
    config_path = run_dir / "config.json"
    args_record = {k: v for k, v in vars(a).items() if k != "resume"}
    previous = None
    if config_path.exists():
        if not a.resume:
            sys.exit(f"{run_dir} already exists. Pass --resume to continue it, or delete the folder to restart.")
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous["args"] != args_record:
            sys.exit(f"--resume needs the same arguments as the original run: {previous['args']}")
        if not any(run_dir.glob("checkpoint-*")):
            sys.exit(f"{run_dir} has no checkpoint yet (stopped before the first save). "
                     "Delete the folder and start the run again without --resume.")
    elif a.resume:
        sys.exit(f"{run_dir} does not exist; nothing to resume. Run without --resume.")
    if not torch.cuda.is_available():
        sys.exit("CUDA GPU not found; install the PyTorch CUDA wheel (see docs/run_guide.md).")
    run_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    train_df = load_train(a.setting, a.ratio)
    evals = {s: pd.read_parquet(PROCESSED / f"{s}.parquet") for s in EVAL_SPLITS}
    data_info = {"train": {"file": f"train_{a.setting}.parquet" if a.ratio == 1.0 else
                           "train_e0.parquet + civil_pool_perm.parquet", "rows": len(train_df),
                           "fingerprint": fingerprint(train_df)},
                 **{s: {"file": f"{s}.parquet", "rows": len(df), "fingerprint": fingerprint(df)}
                    for s, df in evals.items()}}
    if a.smoke:  # keep the original index: it is the `row` written to the prediction files
        train_df = train_df.sample(frac=SMOKE_FRAC, random_state=SEED)
        evals = {s: df.sample(frac=SMOKE_FRAC, random_state=SEED).sort_index() for s, df in evals.items()}

    ckpt = CHECKPOINTS[a.model]
    tok = AutoTokenizer.from_pretrained(ckpt)
    train_ds = to_dataset(train_df, tok, with_labels=True)
    eval_ds = {s: to_dataset(df, tok, with_labels=(s == "val")) for s, df in evals.items()}
    # The new classifier head is drawn from the global RNG inside from_pretrained, before Trainer sets the seed.
    set_seed(a.seed)
    # dtype=float32: transformers v5 defaults to the checkpoint's dtype, and deberta-v3-base is stored in fp16.
    # fp16 master weights make AdamW diverge (constant predictions); bf16/fp16 stays as autocast only.
    model = AutoModelForSequenceClassification.from_pretrained(
        ckpt, dtype=torch.float32, num_labels=len(LABELS), problem_type="multi_label_classification",
        id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)})
    assert all(p.dtype == torch.float32 for p in model.parameters()), "model weights must be fp32"

    # including_emulation=False: pre-Ampere cards (e.g. RTX 20xx) report emulated bf16, which is very slow.
    precision = "bf16" if torch.cuda.is_bf16_supported(including_emulation=False) else "fp16"
    grad_accum = HPARAMS["effective_batch_size"] // a.batch_size
    epochs = 1 if a.smoke else HPARAMS["epochs"]
    steps_per_epoch = len(train_ds) // HPARAMS["effective_batch_size"]
    eval_steps = max(1, steps_per_epoch // HPARAMS["evals_per_epoch"])  # floor: guarantees 2 evals per epoch
    targs = TrainingArguments(
        output_dir=str(run_dir),
        num_train_epochs=epochs,
        learning_rate=HPARAMS["learning_rate"],
        weight_decay=HPARAMS["weight_decay"],
        optim="adamw_torch_fused",  # AdamW; fused kernel is the v5 default and only faster
        lr_scheduler_type=HPARAMS["lr_scheduler"],
        warmup_steps=math.ceil(HPARAMS["warmup_ratio"] * steps_per_epoch * epochs),
        per_device_train_batch_size=a.batch_size,
        gradient_accumulation_steps=grad_accum,
        per_device_eval_batch_size=64,
        bf16=precision == "bf16",
        fp16=precision == "fp16",
        train_sampling_strategy="group_by_length",  # v5 name of group_by_length=True
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,  # aligned with eval so the best checkpoint is always on disk
        save_total_limit=2,  # best + last
        load_best_model_at_end=True,
        metric_for_best_model="mean_roc_auc",
        greater_is_better=True,
        logging_steps=50,
        seed=a.seed,
        data_seed=a.seed,
        report_to="none",
        dataloader_num_workers=0,
    )
    session = {"started": datetime.now().isoformat(timespec="seconds"), "gpu": torch.cuda.get_device_name(0),
               "cuda": torch.version.cuda, **environment(PACKAGES)}
    if previous:  # keep the original record; append this session's environment
        previous.setdefault("resume_sessions", []).append(session)
        write_json(config_path, previous)
    else:
        write_json(config_path, {
            "run": run, "args": args_record, "model": a.model, "checkpoint": ckpt,
            "checkpoint_revision": hub_revision(ckpt),
            "setting": a.setting, "seed": a.seed, "ratio": a.ratio, "smoke": a.smoke, "hparams": HPARAMS,
            "epochs_run": epochs, "precision": precision, "per_device_batch_size": a.batch_size,
            "gradient_accumulation_steps": grad_accum, "steps_per_epoch": steps_per_epoch, "eval_steps": eval_steps,
            "train_rows_used": len(train_ds),
            "data": data_info,  # rows/fingerprints of the full files, before any smoke subsampling
            **session,
        })

    val_by_length = eval_ds["val"].select(np.argsort(np.asarray(eval_ds["val"]["length"]), kind="stable"))
    trainer = ProjectTrainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_by_length,
                             processing_class=tok, data_collator=DataCollatorWithPadding(tok),
                             compute_metrics=hf_metrics)
    t_prep = time.time() - t0
    train_out = trainer.train(resume_from_checkpoint=True if a.resume else None)
    t_train = time.time() - t0 - t_prep

    predict_seconds = {}
    for s, ds in eval_ds.items():
        t1 = time.time()
        probs = predict(trainer, ds)
        predict_seconds[s] = round(time.time() - t1, 1)
        save_predictions(run_dir, s, evals[s].index.to_numpy(), probs)
        if s == "val":
            m = compute_metrics(evals[s][LABELS], probs, LABELS)
            write_json(run_dir / "metrics_val.json", m)
            # Alignment checks. (1) Element-wise: the fp32 predictions (length-sorted, then restored) must match
            # Trainer.predict's bf16 predictions taken in plain row order; only bf16 rounding may differ.
            bf16_probs = sigmoid(trainer.predict(ds).predictions)
            max_diff = float(np.abs(probs - bf16_probs).max())
            assert max_diff < 0.05, f"val predictions misaligned: max |fp32 - bf16| = {max_diff:.4f}"
            # (2) Against the labels: a row mix-up would drop the score to ~0.5. Only a warning, so a borderline
            # fp32-vs-bf16 gap can't abort a finished run; (1) is the strict check.
            gap = abs(m["mean_roc_auc"] - trainer.state.best_metric)
            if gap > (0.05 if a.smoke else 0.01):
                print(f"WARNING: val mean ROC-AUC from saved predictions {m['mean_roc_auc']:.5f} differs from "
                      f"training's {trainer.state.best_metric:.5f}; report this run.")

    write_json(run_dir / "log_history.json", trainer.state.log_history)
    write_json(run_dir / "runtime.json", {
        "resumed": a.resume,  # if true, the timing fields below cover only the last (resumed) session
        "prepare_seconds": round(t_prep, 1), "train_seconds": round(t_train, 1),
        "train_samples_per_second": train_out.metrics.get("train_samples_per_second"),
        "predict_seconds": predict_seconds,
        "val_max_abs_diff_fp32_vs_bf16": max_diff,
        "val_mean_roc_auc_fp32_minus_training": m["mean_roc_auc"] - trainer.state.best_metric,
        "peak_vram_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "peak_vram_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2),
        "best_checkpoint": trainer.state.best_model_checkpoint, "best_val_mean_roc_auc": trainer.state.best_metric,
        "global_steps": trainer.state.global_step,
    })
    print(f"Done: {run_dir} (best val mean ROC-AUC {trainer.state.best_metric}, "
          f"checkpoint {trainer.state.best_model_checkpoint})")
    if not a.smoke and trainer.state.best_metric < SANITY_VAL_AUC:
        print(f"WARNING: best val mean ROC-AUC {trainer.state.best_metric:.4f} < {SANITY_VAL_AUC}. "
              "Training probably diverged; report this run instead of using it.")


if __name__ == "__main__":
    main()
