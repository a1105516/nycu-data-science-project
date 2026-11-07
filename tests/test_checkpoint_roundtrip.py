"""CPU regression checks for Trainer checkpoint saving, best loading and resume.

Run with: .venv/Scripts/python -m unittest discover -s tests -v
The tiny legacy BERT fixture reproduces the gamma/beta naming of bert-base-uncased.
No experiment data, downloaded weights or production hyperparameters are changed.
"""
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from datasets import Dataset
from safetensors.torch import load_file, save_file
from transformers import (BertConfig, BertForSequenceClassification, TrainerCallback,
                          TrainingArguments, default_data_collator, set_seed)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train import ProjectTrainer


class StopAfterTwoSteps(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 2:
            control.should_save = True
            control.should_training_stop = True
        return control


class CheckpointRoundtripTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        # Keep scratch files in the writable workspace; Windows sandbox TEMP may
        # allow creating a directory but deny access to its children.
        self.temp_root = (Path(__file__).resolve().parents[1] / "outputs").resolve()
        self.temp_root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="nycu-checkpoint-test-", dir=self.temp_root)
        self.root = Path(self.temp.name).resolve()
        self.assertEqual(self.root.parent, self.temp_root)
        self.addCleanup(self.cleanup_temp)
        self.legacy = self.root / "legacy-bert"
        self.legacy.mkdir()
        set_seed(42)
        config = BertConfig(
            vocab_size=47, hidden_size=16, num_hidden_layers=1,
            num_attention_heads=2, intermediate_size=32, max_position_embeddings=32,
            num_labels=6, problem_type="multi_label_classification",
            hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
        )
        model = BertForSequenceClassification(config)
        config.save_pretrained(self.legacy)
        legacy_state = {
            key.replace("LayerNorm.weight", "LayerNorm.gamma").replace("LayerNorm.bias", "LayerNorm.beta"):
            value.detach().clone().contiguous()
            for key, value in model.state_dict().items()
        }
        self.assertTrue(any(key.endswith("LayerNorm.gamma") for key in legacy_state))
        save_file(legacy_state, self.legacy / "model.safetensors", metadata={"format": "pt"})

    def cleanup_temp(self):
        # Check the resolved target before TemporaryDirectory recursively removes it.
        target = Path(self.temp.name).resolve()
        if target.parent != self.temp_root or not target.name.startswith("nycu-checkpoint-test-"):
            raise RuntimeError(f"Refusing to clean unexpected temporary path: {target}")
        self.temp.cleanup()

    def new_model(self):
        model = BertForSequenceClassification.from_pretrained(
            self.legacy, local_files_only=True, dtype=torch.float32,
        )
        self.assertTrue(getattr(model, "_weight_conversions", None))
        return model

    def arguments(self, directory, **overrides):
        options = dict(
            output_dir=str(directory), use_cpu=True, bf16=False, fp16=False,
            report_to="none", disable_tqdm=True, dataloader_num_workers=0,
            dataloader_pin_memory=False, seed=42, data_seed=42,
            save_strategy="no", eval_strategy="no", logging_strategy="no",
        )
        options.update(overrides)
        return TrainingArguments(**options)

    def assert_state_equal(self, expected, actual):
        self.assertEqual(set(expected), set(actual))
        for key in expected:
            self.assertTrue(torch.equal(expected[key], actual[key]), f"Tensor differs: {key}")

    def snapshot(self, model):
        return {key: value.detach().clone() for key, value in model.state_dict().items()}

    def test_saved_keys_and_best_checkpoint_restore_all_weights(self):
        model = self.new_model()
        trainer = ProjectTrainer(model=model, args=self.arguments(self.root / "best-run"))
        expected = self.snapshot(model)
        best = self.root / "best-run" / "checkpoint-1"
        trainer.save_model(str(best))
        saved = load_file(best / "model.safetensors")
        self.assert_state_equal(expected, saved)
        self.assertFalse(any(key.endswith(("LayerNorm.gamma", "LayerNorm.beta")) for key in saved))
        self.assertTrue((best / "training_args.bin").is_file())

        # Simulate later training: every parameter, including LayerNorm, must change.
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(0.125)
        trainer.state.global_step = 2
        trainer.state.best_model_checkpoint = str(best)
        trainer.state.best_metric = 1.0
        trainer._load_best_model()
        self.assert_state_equal(expected, model.state_dict())

    def test_resume_loader_restores_all_weights(self):
        model = self.new_model()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(0.125)
        expected = self.snapshot(model)
        checkpoint = self.root / "resume-load" / "checkpoint-2"
        trainer = ProjectTrainer(model=model, args=self.arguments(self.root / "resume-load"))
        trainer.save_model(str(checkpoint))

        fresh_model = self.new_model()
        self.assertFalse(torch.equal(
            expected["bert.embeddings.LayerNorm.weight"],
            fresh_model.state_dict()["bert.embeddings.LayerNorm.weight"],
        ))
        resumed = ProjectTrainer(model=fresh_model, args=self.arguments(self.root / "resume-load"))
        resumed._load_from_checkpoint(str(checkpoint))
        self.assert_state_equal(expected, fresh_model.state_dict())

    def assert_nested_equal(self, expected, actual, path="state"):
        if isinstance(expected, torch.Tensor):
            self.assertTrue(torch.equal(expected, actual), f"Tensor differs: {path}")
        elif isinstance(expected, dict):
            self.assertEqual(set(expected), set(actual), path)
            for key in expected:
                self.assert_nested_equal(expected[key], actual[key], f"{path}.{key}")
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(expected), len(actual), path)
            for index, (left, right) in enumerate(zip(expected, actual)):
                self.assert_nested_equal(left, right, f"{path}[{index}]")
        else:
            self.assertEqual(expected, actual, path)

    def test_interrupted_resume_matches_uninterrupted_training(self):
        records = []
        for index in range(16):
            length = 3 + index % 8
            records.append({
                "input_ids": [2 + (index + offset) % 43 for offset in range(length)] + [0] * (12 - length),
                "attention_mask": [1] * length + [0] * (12 - length),
                "labels": [float((index + label) % 3 == 0) for label in range(6)],
                "length": length,
            })
        dataset = Dataset.from_list(records)

        def make_trainer(directory, callbacks=None):
            return ProjectTrainer(
                model=self.new_model(),
                args=self.arguments(
                    directory, max_steps=6, per_device_train_batch_size=2,
                    gradient_accumulation_steps=2, learning_rate=1e-3,
                    weight_decay=0.01, warmup_steps=1, optim="adamw_torch",
                    train_sampling_strategy="group_by_length", save_strategy="steps",
                    save_steps=2, save_total_limit=3,
                ),
                train_dataset=dataset, data_collator=default_data_collator,
                callbacks=callbacks,
            )

        full = make_trainer(self.root / "full")
        full.train()
        interrupted = make_trainer(self.root / "interrupted", [StopAfterTwoSteps()])
        interrupted.train()
        self.assertEqual(interrupted.state.global_step, 2)
        checkpoint = self.root / "interrupted" / "checkpoint-2"
        for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json"):
            self.assertTrue((checkpoint / name).is_file(), name)

        resumed = make_trainer(self.root / "interrupted")
        resumed.train(resume_from_checkpoint=str(checkpoint))
        self.assertEqual(full.state.global_step, 6)
        self.assertEqual(resumed.state.global_step, 6)
        self.assertEqual(full.state.epoch, resumed.state.epoch)
        self.assert_state_equal(full.model.state_dict(), resumed.model.state_dict())
        self.assert_nested_equal(full.optimizer.state_dict(), resumed.optimizer.state_dict(), "optimizer")
        self.assert_nested_equal(full.lr_scheduler.state_dict(), resumed.lr_scheduler.state_dict(), "scheduler")


if __name__ == "__main__":
    unittest.main()
