"""Phase 3 baseline: TF-IDF + one LogisticRegression per label, E0 only (docs/plan.md §5.1).

Picks C on val mean ROC-AUC, then writes the same prediction files as train.py to outputs/tfidf_lr_e0/.
CPU only.

Run: .venv/Scripts/python src/baseline.py
"""
import time

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from common import (EVAL_SPLITS, LABELS, OUTPUTS, PROCESSED, SEED, compute_metrics, environment, fingerprint,
                    save_predictions, write_json)

RUN = "tfidf_lr_e0"
C_GRID = [0.5, 1, 2, 4]
VECTORIZERS = {
    "word": dict(analyzer="word", ngram_range=(1, 2), max_features=100_000, sublinear_tf=True),
    "char_wb": dict(analyzer="char_wb", ngram_range=(2, 5), max_features=100_000, sublinear_tf=True),
}
PACKAGES = ["numpy", "pandas", "pyarrow", "scikit-learn", "scipy"]

run_dir = OUTPUTS / RUN
run_dir.mkdir(parents=True, exist_ok=True)
t0 = time.time()

train = pd.read_parquet(PROCESSED / "train_e0.parquet")
evals = {s: pd.read_parquet(PROCESSED / f"{s}.parquet") for s in EVAL_SPLITS}

vecs = {k: TfidfVectorizer(**kw) for k, kw in VECTORIZERS.items()}
X_train = hstack([v.fit_transform(train["text"]) for v in vecs.values()]).tocsr()
X = {s: hstack([v.transform(df["text"]) for v in vecs.values()]).tocsr() for s, df in evals.items()}
t_features = time.time() - t0

# One classifier per label and C; keep the fitted models so the chosen C needs no refit.
models, val_scores = {}, {}
for c in C_GRID:
    models[c] = [LogisticRegression(C=c, solver="liblinear", random_state=SEED).fit(X_train, train[l] >= 0.5)
                 for l in LABELS]
    probs = np.column_stack([m.predict_proba(X["val"])[:, 1] for m in models[c]])
    val_scores[c] = compute_metrics(evals["val"][LABELS], probs, LABELS)["mean_roc_auc"]
    print(f"C={c}: val mean ROC-AUC {val_scores[c]:.5f}")
best_c = max(val_scores, key=val_scores.get)
t_fit = time.time() - t0 - t_features

t1 = time.time()
for s, df in evals.items():
    probs = np.column_stack([m.predict_proba(X[s])[:, 1] for m in models[best_c]])
    save_predictions(run_dir, s, np.arange(len(df)), probs)
    if s == "val":
        write_json(run_dir / "metrics_val.json", compute_metrics(df[LABELS], probs, LABELS))

write_json(run_dir / "config.json", {
    "run": RUN, "model": "tfidf_lr", "setting": "e0", "seed": SEED, "smoke": False,
    "vectorizers": VECTORIZERS, "lowercase": True,  # TfidfVectorizer default; part of the features, not of clean()
    "classifier": "LogisticRegression(solver='liblinear'), one per label",
    "c_grid": C_GRID, "val_mean_roc_auc_by_c": val_scores, "best_c": best_c,
    "n_features": X_train.shape[1],
    "data": {"train": {"file": "train_e0.parquet", "rows": len(train), "fingerprint": fingerprint(train)},
             **{s: {"file": f"{s}.parquet", "rows": len(df), "fingerprint": fingerprint(df)} for s, df in evals.items()}},
    **environment(PACKAGES),
})
write_json(run_dir / "runtime.json", {"features_seconds": round(t_features, 1), "fit_seconds": round(t_fit, 1),
                                      "predict_seconds": round(time.time() - t1, 1),
                                      "total_seconds": round(time.time() - t0, 1)})
print(f"Best C={best_c} (val mean ROC-AUC {val_scores[best_c]:.5f}). Wrote {run_dir}")
