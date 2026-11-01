"""Stage 1 - INGEST: bring the source file into the raw zone and prove it is the right one.

Run:  python -m src.ingest                    # download from Hugging Face (pinned revision)
      python -m src.ingest --local PATH       # use a copy you already downloaded

The raw zone is immutable: the file is accepted only if its size and SHA-256 match the
values pinned in src/config.py. Any mismatch stops the pipeline.
"""
import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path

from src import config
from src.utils import sha256, update_manifest


def verify(path: Path) -> str:
    size = path.stat().st_size
    if size != config.RAW_SIZE:
        raise SystemExit(f"Raw file size {size} != expected {config.RAW_SIZE}: wrong file or version.")
    digest = sha256(path)
    if digest != config.RAW_SHA256:
        raise SystemExit(f"Raw SHA-256 {digest} != expected {config.RAW_SHA256}: wrong file or version.")
    return digest


def download() -> Path:
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo_id=config.HF_REPO_ID, repo_type=config.HF_REPO_TYPE,
                        filename=config.HF_FILENAME, revision=config.HF_REVISION,
                        local_dir=config.RAW_DIR)
    return Path(p)


def main(local: str | None = None) -> Path:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = config.RAW_FILE
    if target.exists():
        origin = "already in raw zone"
    elif local:
        shutil.copyfile(local, target)
        origin = "local copy"
    else:
        downloaded = download()
        if downloaded.resolve() != target.resolve():
            shutil.move(str(downloaded), target)
        origin = "huggingface_hub download"
    digest = verify(target)
    update_manifest("raw", {
        "file": str(target.relative_to(config.ROOT)),
        "source": f"https://huggingface.co/datasets/{config.HF_REPO_ID}",
        "revision": config.HF_REVISION,
        "license": config.SOURCE_LICENSE,
        "citation": config.SOURCE_CITATION,
        "size_bytes": config.RAW_SIZE,
        "sha256": digest,
        "committed_to_git": False,
        "verified_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    })
    print(f"[ingest] OK ({origin}) sha256={digest[:12]}...")
    return target


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", help="path to a data.jsonl already downloaded")
    main(ap.parse_args().local)
