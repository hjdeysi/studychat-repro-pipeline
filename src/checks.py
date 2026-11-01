"""Self-check: verify the data lake and the protocol before trusting any number.

Run:  python -m src.checks
Exits with an error if any check fails.
"""
import sys

import pandas as pd

from src import config
from src.utils import read_manifest, sha256


def main() -> bool:
    m = read_manifest()
    checks = []

    if config.RAW_FILE.exists():
        checks.append(("raw file matches pinned SHA-256", sha256(config.RAW_FILE) == config.RAW_SHA256))
    else:
        checks.append(("raw file present (skipped: not downloaded)", True))

    proc_ok = config.PROCESSED_FILE.exists() and sha256(config.PROCESSED_FILE) == m.get("processed", {}).get("sha256")
    checks.append(("processed file matches DATA_MANIFEST checksum", proc_ok))

    if config.PROCESSED_FILE.exists():
        df = pd.read_csv(config.PROCESSED_FILE)
        tr = set(df.user_id[df.split == "train"])
        te = set(df.user_id[df.split == "test"])
        checks.append(("no student in both train and test", not (tr & te)))
        checks.append(("target is binary and both classes present", set(df.y_delegation) == {0, 1}))
        checks.append(("seen_in_train only flags test rows", not df.seen_in_train[df.split == "train"].any()))

    summary = config.REPORTS_DIR / "results_summary.csv"
    checks.append(("results_summary.csv exists", summary.exists()))

    for name, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    all_ok = all(ok for _, ok in checks)
    print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    return all_ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
