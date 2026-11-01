"""Run the whole pipeline end to end: ingest -> validate -> transform -> 5 seeds -> checks.

Run:  PYTHONHASHSEED=0 python -m src.pipeline                 # downloads the raw file
      PYTHONHASHSEED=0 python -m src.pipeline --local PATH    # uses your local copy
      PYTHONHASHSEED=0 python -m src.pipeline --from-processed  # skip raw, use committed table
"""
import argparse
import sys

from src import checks, ingest, run_seeds, transform, validate


def main(local: str | None = None, from_processed: bool = False) -> None:
    if not from_processed:
        ingest.main(local)
        validate.main()
        transform.main()
    run_seeds.main()
    if not checks.main():
        sys.exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", help="path to a data.jsonl already downloaded")
    ap.add_argument("--from-processed", action="store_true",
                    help="start from data/processed (committed) instead of the raw zone")
    a = ap.parse_args()
    main(a.local, a.from_processed)
