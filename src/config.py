"""Single source of truth for paths, data provenance and experiment settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- Data lake zones -------------------------------------------------------
# raw       : immutable copy of the source file, verified by SHA-256 (never edited, never committed)
# interim   : rows that passed schema validation, flattened (regenerable, not committed)
# processed : model-ready table with the target variable (small, committed)
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT / "reports"
MANIFEST = DATA_DIR / "DATA_MANIFEST.json"

RAW_FILE = RAW_DIR / "data.jsonl"
INTERIM_FILE = INTERIM_DIR / "studychat_turns.csv.gz"
PROCESSED_FILE = PROCESSED_DIR / "studychat_prompts.csv.gz"

# --- Source (pinned) -------------------------------------------------------
HF_REPO_ID = "wmcnicho/StudyChat"
HF_REPO_TYPE = "dataset"
HF_REVISION = "24d7987d9fbb30d9da12acc53455a10f1cdd2d7f"  # commit of 2026-05-26 (v1.1)
HF_FILENAME = "data.jsonl"
RAW_SHA256 = "67927f73327639904c417c2f6e6200e11427920a8435b39cb7c87b5c6601e130"
RAW_SIZE = 626_248_888
SOURCE_LICENSE = "CC BY 4.0"
SOURCE_CITATION = (
    "McNichols, H., Ikram, F., & Lan, A. (2026). The StudyChat Dataset: Analyzing Student "
    "Dialogues With ChatGPT in an Artificial Intelligence Course. In Proceedings of LAK26, 53-63."
)

# --- Target variable -------------------------------------------------------
# A student turn counts as a DELEGATION request when its dialogue-act label asks the
# assistant to produce or edit the work (code or text) instead of helping to understand it.
DELEGATION_TOP_LABELS = {"writing_request", "editing_request"}

# --- Split -----------------------------------------------------------------
# Train on the earlier cohort, test once on the later cohort (different students).
TRAIN_SEMESTER = "f24"
TEST_SEMESTER = "s25"

# --- Experiment ------------------------------------------------------------
SEEDS = [0, 1, 2, 3, 42]
N_FOLDS = 5
MAX_CHARS = 5000  # long pasted code is truncated before vectorizing (see README)
MLFLOW_EXPERIMENT = "d05-studychat-delegation"
