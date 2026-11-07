"""Phase 4 evaluation (docs/plan.md §6). Reads prediction files only; never re-runs a model.

Run: .venv/Scripts/python src/evaluate.py [--outputs outputs/smoke]

Writes {outputs}/results.md, metrics_long.csv, val_curves.png and error_samples_{model}.csv.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (EVAL_SPLITS, LABELS, OUTPUTS, PROCESSED, SEED, SPLIT_LABELS, compute_metrics, fingerprint,
                    load_train, md_table)

MODEL_ORDER = ["tfidf_lr", "bert", "roberta", "deberta"]
DELTAS = [("e1", "e0", "E1 − E0（hard augmentation）"), ("e2", "e0", "E2 − E0（soft augmentation）"),
          ("e2", "e1", "E2 − E1（soft vs. hard）")]
FOCUS_LABELS = ["threat", "identity_hate"]  # plan §6.4: read 20 E0 false negatives each
ERROR_MODELS = ["deberta"]
N_SAMPLES = 20


def load_runs(outdir: Path, truth: dict) -> pd.DataFrame:
    local_fp = {s: fingerprint(df) for s, df in truth.items()}
    train_fp = {}  # (setting, ratio) -> fingerprint of the local training set, computed once
    rows = []
    for cfg_path in sorted(outdir.glob("*/config.json")):
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        run_dir = cfg_path.parent
        if not all((run_dir / f"preds_{s}.parquet").exists() for s in EVAL_SPLITS):
            print(f"skip {run_dir.name}: predictions incomplete (still training?)")
            continue
        # The training set must match too, otherwise E1/E2 may hold a different external sample on another machine.
        key = (cfg["setting"], cfg.get("ratio", 1.0))
        if key not in train_fp:
            train_fp[key] = fingerprint(load_train(*key))
        assert cfg["data"]["train"]["fingerprint"] == train_fp[key], (
            f"{run_dir.name}: trained on different data than data/processed (train fingerprint mismatch)")
        for s in EVAL_SPLITS:
            assert cfg["data"][s]["fingerprint"] == local_fp[s], (
                f"{run_dir.name}: {s} was built from different data than data/processed (fingerprint mismatch)")
            preds = pd.read_parquet(run_dir / f"preds_{s}.parquet")
            if not cfg["smoke"]:
                assert np.array_equal(np.sort(preds["row"]), np.arange(len(truth[s]))), f"{run_dir.name}: {s} rows"
            labels = SPLIT_LABELS[s]
            m = compute_metrics(truth[s][labels].iloc[preds["row"]], preds[labels], labels)
            if not cfg["smoke"]:  # a NaN would silently turn the mean into a mean over fewer labels
                assert not any(np.isnan(v) for v in m.values()), f"{run_dir.name}: NaN metric on {s}"
            rows.append({"run": cfg["run"], "model": cfg["model"], "setting": cfg["setting"],
                         "ratio": cfg.get("ratio", 1.0), "seed": cfg["seed"], "split": s, **m})
    df = pd.DataFrame(rows)
    df["model_rank"] = df["model"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    return df.sort_values(["model_rank", "setting", "ratio", "seed", "split"]).drop(columns="model_rank")


def fmt(x, signed=False):
    return "—" if pd.isna(x) else (f"{x:+.4f}" if signed else f"{x:.4f}")


def main_runs(metrics):
    return metrics[(metrics["ratio"] == 1.0) & (metrics["seed"] == SEED)]


def tables(metrics) -> list[str]:
    md = []
    test, val = metrics[metrics["split"] == "test_jigsaw"], metrics[metrics["split"] == "val"].set_index("run")
    md += ["## 1. 主表：Jigsaw official labeled test", ""]
    md += md_table(["Run", "Model", "Setting", "Val mean ROC-AUC", "Mean ROC-AUC", "Mean PR-AUC", "Macro F1"],
                   [[r.run, r.model, r.setting, fmt(val.loc[r.run, "mean_roc_auc"]), fmt(r.mean_roc_auc),
                     fmt(r.mean_pr_auc), fmt(r.macro_f1)] for r in test.itertuples()])

    md += ["## 2. Per-label：Jigsaw official labeled test", ""]
    for key, name in [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC")]:
        md += [f"**{name}**", ""]
        md += md_table(["Run", *LABELS], [[r["run"], *(fmt(r[f"{key}/{l}"]) for l in LABELS)]
                                          for _, r in test.iterrows()])

    delta_md = []
    main = main_runs(test).set_index(["model", "setting"])
    for model in [m for m in MODEL_ORDER if m in main.index.get_level_values(0)]:
        pairs = [(a, b, title) for a, b, title in DELTAS if (model, a) in main.index and (model, b) in main.index]
        if not pairs:
            continue
        delta_md += [f"**{model}**", ""]
        for key, name in [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC")]:
            mean_key = f"mean_{key}"
            delta_md += md_table([f"Δ {name}", "Mean", *LABELS],
                                 [[title, fmt(main.loc[(model, a), mean_key] - main.loc[(model, b), mean_key], True),
                                   *(fmt(main.loc[(model, a), f"{key}/{l}"] - main.loc[(model, b), f"{key}/{l}"],
                                         True) for l in LABELS)] for a, b, title in pairs])
    if delta_md:
        md += ["## 3. Δ 表（Jigsaw test；主實驗 runs：ratio 1、seed 42）", "", *delta_md]

    civ_labels = SPLIT_LABELS["test_civil"]
    civ = metrics[metrics["split"] == "test_civil"]
    md += ["## 4. Cross-dataset：Civil test（五類）", ""]
    md += md_table(["Run", "Mean ROC-AUC", "Mean PR-AUC", "Macro F1"],
                   [[r["run"], fmt(r["mean_roc_auc"]), fmt(r["mean_pr_auc"]), fmt(r["macro_f1"])]
                    for _, r in civ.iterrows()])
    for key, name in [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC")]:
        md += [f"**{name}**", ""]
        md += md_table(["Run", *civ_labels], [[r["run"], *(fmt(r[f"{key}/{l}"]) for l in civ_labels)]
                                              for _, r in civ.iterrows()])
    return md


def val_curves(outdir: Path, path: Path) -> bool:
    fig, ax = plt.subplots(figsize=(9, 5))
    drawn = False
    for log_path in sorted(outdir.glob("*/log_history.json")):
        log = [e for e in json.loads(log_path.read_text(encoding="utf-8")) if "eval_mean_roc_auc" in e]
        if log:
            ax.plot([e["epoch"] for e in log], [e["eval_mean_roc_auc"] for e in log], marker="o",
                    label=log_path.parent.name)
            drawn = True
    if drawn:
        ax.set(xlabel="Epoch", ylabel="Val mean ROC-AUC", title="Validation mean ROC-AUC during training")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
    plt.close(fig)
    return drawn


def error_analysis(outdir: Path, metrics, test: pd.DataFrame) -> list[str]:
    """Plan §6.4 at threshold 0.5 on Jigsaw test: FP/FN per setting, E0 -> E2 flips, E0 false-negative samples."""
    md = []
    main = main_runs(metrics[metrics["split"] == "test_jigsaw"])
    for model in ERROR_MODELS:
        runs = main[main["model"] == model].set_index("setting")["run"]
        if not {"e0", "e2"} <= set(runs.index):
            continue
        preds = {s: pd.read_parquet(outdir / run / "preds_test_jigsaw.parquet").set_index("row").sort_index()
                 for s, run in runs.items()}
        rows = preds["e0"].index.to_numpy()  # all rows for real runs, the 1% subset for smoke runs
        assert all(np.array_equal(p.index, rows) for p in preds.values()), f"{model}: settings cover different rows"
        sub = test.iloc[rows]
        y = sub[LABELS].to_numpy() >= 0.5
        pred = {s: p[LABELS].to_numpy() for s, p in preds.items()}
        hit = {k: v >= 0.5 for k, v in pred.items()}
        md += [f"## 5. 錯誤分析：{model}（Jigsaw test，threshold 0.5）", ""]
        md += md_table(["Setting", *(f"{l} FP / FN" for l in LABELS)],
                       [[s, *(f"{(h[:, i] & ~y[:, i]).sum():,} / {(~h[:, i] & y[:, i]).sum():,}"
                              for i in range(len(LABELS)))] for s, h in sorted(hit.items())])
        e0, e2 = hit["e0"], hit["e2"]
        md += ["**E0 → E2 預測翻轉**", ""]
        md += md_table(["Flip", *LABELS], [
            [name, *(f"{(mask[:, i]).sum():,}" for i in range(len(LABELS)))] for name, mask in [
                ("FN → TP", ~e0 & e2 & y), ("TP → FN", e0 & ~e2 & y),
                ("TN → FP", ~e0 & e2 & ~y), ("FP → TN", e0 & ~e2 & ~y)]])
        samples = []
        for l in FOCUS_LABELS:
            i = LABELS.index(l)
            fn = np.flatnonzero(~e0[:, i] & y[:, i])
            pick = np.random.default_rng(SEED).choice(fn, size=min(N_SAMPLES, len(fn)), replace=False)
            samples.append(pd.DataFrame({"label": l, "row": rows[pick], "id": sub["id"].to_numpy()[pick],
                                         "text": sub["text"].to_numpy()[pick], "prob_e0": pred["e0"][pick, i],
                                         "prob_e2": pred["e2"][pick, i]}))
        out = outdir / f"error_samples_{model}.csv"
        samples = pd.concat(samples)
        samples.to_csv(out, index=False, encoding="utf-8-sig")
        counts = "、".join(f"{l} {n} 筆" for l, n in samples.groupby("label", sort=False).size().items())
        md += [f"E0 false negatives（每類最多 {N_SAMPLES} 筆，seed {SEED}；實際 {counts}）已存到 `{out.name}`，"
               "供人工閱讀。", ""]
    return md


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outputs", type=Path, default=OUTPUTS, help="folder holding one sub-folder per run")
    outdir = p.parse_args().outputs
    truth = {s: pd.read_parquet(PROCESSED / f"{s}.parquet") for s in EVAL_SPLITS}
    metrics = load_runs(outdir, truth)
    assert len(metrics), f"no finished runs under {outdir}"
    metrics.to_csv(outdir / "metrics_long.csv", index=False)

    md = ["# Results", "", f"Generated by `src/evaluate.py` from `{outdir.name}/`; do not edit by hand.", "",
          "Metrics follow plan §6.1; F1 uses threshold 0.5.", ""]
    md += tables(metrics)
    md += error_analysis(outdir, metrics, truth["test_jigsaw"])
    if val_curves(outdir, outdir / "val_curves.png"):
        md += ["## 6. 訓練曲線", "", "![val curves](val_curves.png)", ""]
    (outdir / "results.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Evaluated {metrics['run'].nunique()} runs. Wrote {outdir / 'results.md'}")


if __name__ == "__main__":
    main()
