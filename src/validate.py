"""Stage 2 - VALIDATE: check every raw row against the expected schema (raw -> interim).

Run:  python -m src.validate

Rows that break the schema are dropped and counted, never silently fixed.
Writes data/interim/studychat_turns.csv.gz and reports/data_quality.json.
"""
import json
from collections import Counter

import pandas as pd

from src import config
from src.utils import sha256, update_manifest, write_json

REQUIRED = ["prompt", "topic", "chatId", "userId", "interactionCount", "timestamp", "semester", "llm_label"]
SEMESTERS = {config.TRAIN_SEMESTER, config.TEST_SEMESTER}


def main() -> pd.DataFrame:
    issues = Counter()
    rows, seen = [], set()
    n_raw = 0
    with open(config.RAW_FILE, encoding="utf-8") as f:
        for line in f:
            n_raw += 1
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                issues["invalid_json"] += 1
                continue
            missing = [k for k in REQUIRED if k not in r]
            if missing:
                issues["missing_field"] += 1
                continue
            label = r["llm_label"].get("label") if isinstance(r["llm_label"], dict) else None
            prompt = r["prompt"] if isinstance(r["prompt"], str) else ""
            key = (r["chatId"], r["interactionCount"])
            if not prompt.strip():
                issues["empty_prompt"] += 1
                continue
            if not label or ">" not in label:
                issues["missing_or_malformed_label"] += 1
                continue
            if r["semester"] not in SEMESTERS:
                issues["unknown_semester"] += 1
                continue
            if key in seen:
                issues["duplicate_turn"] += 1
                continue
            seen.add(key)
            rows.append({
                "chat_id": r["chatId"], "user_id": r["userId"], "semester": r["semester"],
                "topic": r["topic"], "interaction_count": int(r["interactionCount"]),
                "timestamp": int(r["timestamp"]), "prompt": prompt, "label_fine": label,
            })
    df = pd.DataFrame(rows).sort_values(["semester", "chat_id", "interaction_count"]).reset_index(drop=True)
    config.INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.INTERIM_FILE, index=False, compression={"method": "gzip", "mtime": 0})

    report = {
        "raw_rows": n_raw,
        "valid_rows": len(df),
        "dropped": dict(issues),
        "users": int(df.user_id.nunique()),
        "chats": int(df.chat_id.nunique()),
        "rows_by_semester": df.semester.value_counts().sort_index().to_dict(),
        "users_by_semester": df.groupby("semester").user_id.nunique().to_dict(),
        "users_in_both_semesters": int(len(set(df.user_id[df.semester == config.TRAIN_SEMESTER])
                                           & set(df.user_id[df.semester == config.TEST_SEMESTER]))),
        "prompt_chars_median": int(df.prompt.str.len().median()),
        "prompt_chars_max": int(df.prompt.str.len().max()),
        "prompts_longer_than_max_chars": int((df.prompt.str.len() > config.MAX_CHARS).sum()),
    }
    write_json(config.REPORTS_DIR / "data_quality.json", report)
    update_manifest("interim", {
        "file": str(config.INTERIM_FILE.relative_to(config.ROOT)),
        "derived_from": "raw",
        "rows": len(df),
        "sha256": sha256(config.INTERIM_FILE),
        "committed_to_git": False,
    })
    print(f"[validate] {len(df)}/{n_raw} rows valid; dropped {dict(issues)}")
    return df


if __name__ == "__main__":
    main()
