"""Stage 4 - TRAIN & EVALUATE one seed.

Run:  PYTHONHASHSEED=0 python -m src.train --seed 42

Protocol (no leakage):
  * train = Fall 2024 cohort, test = Spring 2025 cohort (no student appears in both);
  * every learned step (truncation is stateless; TF-IDF vocabulary and IDF weights; the
    classifier) lives inside a scikit-learn Pipeline, so in cross-validation it is fit on
    the training folds only;
  * hyper-parameters are chosen by 5-fold StratifiedGroupKFold on the training cohort,
    grouped by student, so one student's turns never sit on both sides of a fold;
  * the test cohort is used once, after selection.
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src import config
from src.utils import set_seed


def truncate(texts):
    """Keep the first MAX_CHARS characters of each prompt (stateless, so leakage-free)."""
    return [t[: config.MAX_CHARS] for t in texts]


def tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_df=0.95,
                           sublinear_tf=True, max_features=100_000,
                           token_pattern=r"(?u)\b\w+\b")


def candidates(seed: int) -> dict:
    trunc = ("truncate", FunctionTransformer(truncate))
    return {
        "majority_baseline": (DummyClassifier(strategy="most_frequent"), {}),
        "tfidf_complementnb": (
            Pipeline([trunc, ("tfidf", tfidf()), ("clf", ComplementNB())]),
            {"clf__alpha": [0.1, 0.5, 1.0]},
        ),
        "tfidf_logreg": (
            Pipeline([trunc, ("tfidf", tfidf()),
                      ("clf", LogisticRegression(class_weight="balanced", solver="liblinear",
                                                 max_iter=2000, random_state=seed))]),
            {"clf__C": [0.1, 1.0, 10.0, 100.0]},
        ),
    }


def scores(y_true, y_pred, y_score) -> dict:
    out = {
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_delegation": precision_score(y_true, y_pred, zero_division=0),
        "recall_delegation": recall_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
    }
    return {k: round(float(v), 4) for k, v in out.items()}


def load() -> pd.DataFrame:
    return pd.read_csv(config.PROCESSED_FILE)


def main(seed: int = 42, df: pd.DataFrame | None = None) -> list[dict]:
    set_seed(seed)
    df = load() if df is None else df
    tr, te = df[df.split == "train"], df[df.split == "test"]
    Xtr, ytr, gtr = tr.prompt.tolist(), tr.y_delegation.values, tr.user_id.values
    Xte, yte = te.prompt.tolist(), te.y_delegation.values
    unseen = ~te.seen_in_train.values  # robustness: drop test prompts copied verbatim from train
    cv = StratifiedGroupKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=seed)

    results = []
    for name, (estimator, grid) in candidates(seed).items():
        gs = GridSearchCV(estimator, grid, scoring="f1_macro", cv=cv, n_jobs=1, refit=True)
        gs.fit(Xtr, ytr, groups=gtr)
        y_pred = gs.predict(Xte)
        y_score = gs.predict_proba(Xte)[:, 1]
        test = scores(yte, y_pred, y_score)
        test_unseen = scores(yte[unseen], y_pred[unseen], y_score[unseen])
        results.append({
            "seed": seed,
            "model": name,
            "best_params": json.dumps(gs.best_params_, sort_keys=True),
            "cv_f1_macro": round(float(gs.best_score_), 4),
            **{f"test_{k}": v for k, v in test.items()},
            "test_unseen_f1_macro": test_unseen["f1_macro"],
        })
    return results


def select(results: list[dict]) -> dict:
    """Pick the model with the best CROSS-VALIDATION score (never the test score)."""
    learned = [r for r in results if r["model"] != "majority_baseline"]
    return max(learned, key=lambda r: r["cv_f1_macro"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    res = main(ap.parse_args().seed)
    print(pd.DataFrame(res).drop(columns="best_params").to_string(index=False))
    best = select(res)
    print(json.dumps({"selected_model": best["model"], "best_params": json.loads(best["best_params"]),
                      "cv_f1_macro": best["cv_f1_macro"], "test_f1_macro": best["test_f1_macro"]}))
