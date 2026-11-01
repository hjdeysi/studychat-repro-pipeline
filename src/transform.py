"""Stage 3 - TRANSFORM: build the model-ready table (interim -> processed).

Run:  python -m src.transform

Adds the target variable `y_delegation` and a leakage flag for the test cohort:
`seen_in_train` marks test prompts whose exact text also appears in the training cohort
(e.g. assignment instructions pasted by many students). It is computed from the training
side only, so no test information reaches the model.
"""
import pandas as pd

from src import config
from src.utils import sha256, update_manifest


def normalise(s: pd.Series) -> pd.Series:
    return s.str.strip().str.lower()


def main() -> pd.DataFrame:
    df = pd.read_csv(config.INTERIM_FILE)
    df["label_top"] = df.label_fine.str.split(">").str[0]
    df["y_delegation"] = df.label_top.isin(config.DELEGATION_TOP_LABELS).astype(int)
    df["split"] = df.semester.map({config.TRAIN_SEMESTER: "train", config.TEST_SEMESTER: "test"})

    train_texts = set(normalise(df.loc[df.split == "train", "prompt"]))
    df["seen_in_train"] = (df.split == "test") & normalise(df.prompt).isin(train_texts)

    cols = ["chat_id", "user_id", "semester", "split", "topic", "interaction_count",
            "prompt", "label_fine", "label_top", "y_delegation", "seen_in_train"]
    df = df[cols]
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.PROCESSED_FILE, index=False, compression={"method": "gzip", "mtime": 0})

    rate = df.groupby("split").y_delegation.mean().round(4).to_dict()
    update_manifest("processed", {
        "file": str(config.PROCESSED_FILE.relative_to(config.ROOT)),
        "derived_from": "interim",
        "rows": len(df),
        "rows_by_split": df.split.value_counts().sort_index().to_dict(),
        "delegation_rate_by_split": rate,
        "test_rows_seen_in_train": int(df.seen_in_train.sum()),
        "target_definition": "y_delegation = 1 if top-level llm_label in "
                             f"{sorted(config.DELEGATION_TOP_LABELS)}, else 0",
        "sha256": sha256(config.PROCESSED_FILE),
        "committed_to_git": True,
    })
    print(f"[transform] {len(df)} rows; delegation rate {rate}; "
          f"test rows seen in train: {int(df.seen_in_train.sum())}")
    return df


if __name__ == "__main__":
    main()
