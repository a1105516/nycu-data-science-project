"""Paths, label definitions and helpers shared by every script (docs/plan.md §2.3)."""
import hashlib
import json
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "datasets"
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"

SEED = 42
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
CIVIL_EVAL_LABELS = [l for l in LABELS if l != "severe_toxic"]  # Civil test has no severe_toxicity positives
# Civil Comments column -> Jigsaw target (plan §4 Step 5); sexual_explicit has no target and is dropped.
CIVIL_TO_JIGSAW = {
    "toxicity": "toxic",
    "severe_toxicity": "severe_toxic",
    "obscene": "obscene",
    "threat": "threat",
    "insult": "insult",
    "identity_attack": "identity_hate",
}
HARD_THRESHOLD = 0.5  # score >= 0.5 -> 1, as in the Jigsaw Unintended Bias competition
F1_THRESHOLD = 0.5  # plan §6.1
EVAL_SPLITS = ["val", "test_jigsaw", "test_civil"]
# Labels each eval split is scored on.
SPLIT_LABELS = {"val": LABELS, "test_jigsaw": LABELS, "test_civil": CIVIL_EVAL_LABELS}


def clean(text: pd.Series) -> pd.Series:
    """The only text cleaning step (plan §4 Step 2): strip leading/trailing whitespace."""
    return text.str.strip()


def build_augmented(train_e0: pd.DataFrame, ext: pd.DataFrame, seed: int = SEED):
    """E1/E2 from the Jigsaw train split plus external Civil rows (plan §4 Step 5).

    E2 keeps the raw 0-1 scores; E1 is E2 thresholded at >= 0.5, so both share texts and row order.
    Used by prepare_data.py (1:1) and by train.py --ratio (ablation prefixes of civil_pool_perm).
    """
    combined = pd.concat([train_e0.assign(source="jigsaw"), ext.assign(source="civil")], ignore_index=True)
    e2 = combined.sample(frac=1, random_state=seed).reset_index(drop=True)
    e1 = e2.copy()
    e1[LABELS] = (e2[LABELS] >= HARD_THRESHOLD).astype("float32")  # Jigsaw rows are 0/1, unchanged
    return e1, e2


def load_train(setting: str, ratio: float = 1.0) -> pd.DataFrame:
    """Training set for a setting; ratio != 1 rebuilds E1/E2 from a civil_pool_perm prefix (plan §7 item 4)."""
    if ratio == 1.0:
        return pd.read_parquet(PROCESSED / f"train_{setting}.parquet")
    e0 = pd.read_parquet(PROCESSED / "train_e0.parquet").drop(columns="source")
    pool = pd.read_parquet(PROCESSED / "civil_pool_perm.parquet")
    n_ext = round(ratio * len(e0))
    assert n_ext <= len(pool), f"pool has {len(pool):,} rows, ratio {ratio} needs {n_ext:,}"
    e1, e2 = build_augmented(e0, pool.iloc[:n_ext])
    return e1 if setting == "e1" else e2


def fingerprint(df: pd.DataFrame) -> str:
    """Content hash of a processed file, to check two machines built identical data."""
    return hashlib.md5(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()[:12]


def compute_metrics(y_true, y_prob, labels) -> dict:
    """Plan §6.1 metrics. A label with a single class in y_true gets NaN (only expected on tiny smoke subsets;
    evaluate.py asserts there is none for real runs)."""
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    out = {}
    for i, l in enumerate(labels):
        t, p = y_true[:, i] >= 0.5, y_prob[:, i]
        two_classes = 0 < t.sum() < len(t)
        out[f"roc_auc/{l}"] = roc_auc_score(t, p) if two_classes else np.nan
        out[f"pr_auc/{l}"] = average_precision_score(t, p) if two_classes else np.nan
        out[f"f1/{l}"] = f1_score(t, p >= F1_THRESHOLD, zero_division=0)
    for name, key in [("mean_roc_auc", "roc_auc"), ("mean_pr_auc", "pr_auc"), ("macro_f1", "f1")]:
        out[name] = float(np.nanmean([out[f"{key}/{l}"] for l in labels]))
    return out


def git_commit() -> str:
    """HEAD hash, suffixed -dirty when tracked files have uncommitted changes."""
    def git(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    head = git("rev-parse", "HEAD") or "unknown"
    return head + ("-dirty" if git("status", "--porcelain", "--untracked-files=no") else "")


def environment(packages) -> dict:
    def ver(p):
        try:
            return version(p)
        except PackageNotFoundError:
            return None
    return {"python": sys.version.split()[0], "git_commit": git_commit(), "packages": {p: ver(p) for p in packages}}


def save_predictions(run_dir: Path, split: str, rows: np.ndarray, probs: np.ndarray):
    """preds_{split}.parquet: `row` indexes data/processed/{split}.parquet, then one probability column per label."""
    df = pd.DataFrame(probs.astype("float32"), columns=LABELS)
    df.insert(0, "row", rows.astype("int64"))
    df.to_parquet(run_dir / f"preds_{split}.parquet", index=False)


def md_table(header, rows) -> list[str]:
    return (["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
            + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows] + [""])


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
