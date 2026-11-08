"""Paths, label definitions and helpers shared by every script (docs/plan.md §2.3)."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "datasets"
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"

SEED = 42
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
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


def clean(text: pd.Series) -> pd.Series:
    """The only text cleaning step (plan §4 Step 2): strip leading/trailing whitespace."""
    return text.str.strip()
