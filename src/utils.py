"""Small helpers shared by the pipeline stages."""
import hashlib
import json
import os
import random
import subprocess
from pathlib import Path

import numpy as np

from src import config


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def git_commit() -> str:
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                       text=True, cwd=config.ROOT)
    return r.stdout.strip() or "unknown"


def read_manifest() -> dict:
    if config.MANIFEST.exists():
        return json.loads(config.MANIFEST.read_text(encoding="utf-8"))
    return {}


def update_manifest(section: str, payload: dict) -> None:
    """Record one zone of the data lake (checksum, rows, lineage) in DATA_MANIFEST.json."""
    m = read_manifest()
    m[section] = payload
    config.MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    config.MANIFEST.write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def require_hash_seed() -> None:
    if os.environ.get("PYTHONHASHSEED") != "0":
        print("WARNING: run with PYTHONHASHSEED=0 for bit-identical results.")
