# Reproducible Data + ML Pipeline on StudyChat · Deliverable 05 (v2)

Research Methods and Scientific Integrity in AI and Advanced Technologies · UNMSM · 2026-II
Student: Deisy Lizbeth Acosta Ticse · Instructor: Dr. Loveleen Gaur

**EN:** Detect when a university student asks a generative-AI assistant to *do the work* (write or edit code/text) rather than to help them understand it, using **real student–ChatGPT dialogues** from the public StudyChat dataset. The repository implements a small **data lake** (raw → interim → processed) and a leakage-free, seeded ML pipeline that a stranger can rerun to the same numbers.
**ES:** Detectar cuándo un estudiante universitario le pide a un asistente de IA generativa que *haga el trabajo* (escribir o editar código/texto) en lugar de ayudarle a entender, usando **diálogos reales estudiante–ChatGPT** del dataset público StudyChat. El repositorio implementa un pequeño **data lake** (raw → interim → processed) y un pipeline de ML sin fuga de datos y con semillas fijas, que un extraño puede volver a ejecutar y obtener los mismos números.

**Link to the thesis · Relación con la tesis:** the thesis designs a conversational agent that scaffolds reflection when students delegate academic work to generative AI. Recognising a delegation request is the trigger for that scaffolding, so this pipeline prototypes the detector on real data before the thesis collects its own Spanish-language corpus.

## Data · Datos

| | |
|---|---|
| Source | [`wmcnicho/StudyChat`](https://huggingface.co/datasets/wmcnicho/StudyChat), revision `24d7987d9fbb30d9da12acc53455a10f1cdd2d7f` |
| Paper | McNichols, Ikram & Lan (2026), LAK26, pp. 53–63 · [arXiv:2503.07928](https://arxiv.org/abs/2503.07928) |
| License | CC BY 4.0 (attribution above) |
| Content | 16,851 student turns, 2,214 chats, 203 consenting students, COMPSCI 383 (UMass Amherst), Fall 2024 and Spring 2025 |
| Raw file | `data.jsonl`, 626,248,888 bytes, SHA-256 `67927f73…c6601e130` |

### Data lake zones · Zonas del data lake

```
Hugging Face (pinned revision)
        │  src/ingest.py     size + SHA-256 must match, otherwise STOP
        ▼
data/raw/        immutable source file              not in Git (626 MB)
        │  src/validate.py   schema checks; bad rows dropped and counted
        ▼
data/interim/    validated, flattened turns         not in Git (regenerable)
        │  src/transform.py  target variable + leakage flag
        ▼
data/processed/  model-ready table (3.5 MB)         committed
        │
data/DATA_MANIFEST.json  checksum, row count and lineage of every zone
```

Each zone is regenerated from the previous one and fingerprinted in `DATA_MANIFEST.json`, so any change in the data is visible as a checksum change.

**Target · Variable objetivo.** `y_delegation = 1` when the turn's top-level dialogue-act label is `writing_request` or `editing_request`; `0` for conceptual or contextual questions, verification, providing context and off-topic turns. Delegation rate: 26.5% (train), 29.4% (test).

## How to reproduce · Cómo reproducir

Tested with **Python 3.13** on Linux (pins also have wheels for Python 3.11–3.12).

```bash
git clone https://github.com/hjdeysi/studychat-repro-pipeline.git && cd studychat-repro-pipeline
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# A) full pipeline from the source (downloads 626 MB from Hugging Face)
PYTHONHASHSEED=0 python -m src.pipeline
# B) quick path from the committed processed table
PYTHONHASHSEED=0 python -m src.pipeline --from-processed
```

`PYTHONHASHSEED=0` must precede `python`: it only takes effect when the interpreter starts. Single seed: `PYTHONHASHSEED=0 python -m src.train --seed 42`. Browse runs: `mlflow ui --backend-store-uri sqlite:///mlflow.db`.
Colab: open `notebooks/D05_StudyChat_Reproducible_Pipeline.ipynb`.

## Protocol · Protocolo

- **Split by cohort:** train = Fall 2024 (6,863 turns, 84 students), test = Spring 2025 (9,987 turns, 119 students). No student appears in both (checked by `src/checks.py`).
- **Model selection on training data only:** 5-fold `StratifiedGroupKFold` grouped by student, scored by macro-F1. The test cohort is used once.
- **No leakage:** prompt truncation (stateless), TF-IDF vocabulary/IDF and the classifier live inside one scikit-learn `Pipeline`, so in every fold they are fit on the training folds only ([scikit-learn: Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)).
- **Copied-text check:** 311 test prompts are verbatim copies of training prompts (mostly pasted assignment instructions). Test macro-F1 is also reported without them (`test_unseen_f1_macro`).
- **Candidates:** majority-class baseline, TF-IDF + Complement Naive Bayes (α ∈ {0.1, 0.5, 1}), TF-IDF + logistic regression (C ∈ {0.1, 1, 10, 100}, balanced class weights).

## Results · Resultados (5 seeds: 0, 1, 2, 3, 42 · mean ± sample SD)

| Model | CV macro-F1 (train) | Test macro-F1 | Balanced acc. | Precision (deleg.) | Recall (deleg.) | ROC-AUC | Test macro-F1, copies removed |
|---|---|---|---|---|---|---|---|
| Majority baseline | 0.4223 ± 0.0007 | 0.4138 ± 0.0000 | 0.5000 | 0.0000 | 0.0000 | 0.5000 | 0.4150 |
| TF-IDF + ComplementNB | 0.7599 ± 0.0016 | 0.7059 ± 0.0000 | 0.6940 | 0.6465 | 0.5026 | 0.7636 | 0.7011 |
| **TF-IDF + LogReg (selected)** | **0.8062 ± 0.0034** | **0.7607 ± 0.0022** | 0.7519 ± 0.0039 | 0.6991 ± 0.0052 | 0.6138 ± 0.0127 | 0.8572 ± 0.0061 | 0.7577 ± 0.0027 |

**Reported result: test macro-F1 0.7607 ± 0.0022** for the model selected by cross-validation in every seed (logistic regression; C = 10 in seeds 0, 3, 42 and C = 100 in seeds 1, 2). Files: `reports/results_summary.csv`, `reports/results_by_seed.csv`, `reports/selected_model.json`.

Seeds change the cross-validation folds and therefore the selected hyper-parameter. Naive Bayes always selects the same α, so its test score has SD 0: the SD measures selection instability, not uncertainty about new students.

## What makes it reproducible · Qué lo hace reproducible

| Requirement | How it is met |
|---|---|
| ≥ 4 commits | History in small, labelled steps (`git log`) |
| Same data for everyone | Source pinned to a Hugging Face commit; raw file accepted only if size and SHA-256 match; processed table committed with its checksum in `DATA_MANIFEST.json` |
| Data pipeline, not only a model | Separate ingest → validate → transform stages with a data-quality report (`reports/data_quality.json`) |
| No data leakage | Cohort split with disjoint students, grouped CV, all learned steps inside `Pipeline` |
| Fixed randomness | Python/NumPy seeds; seeds passed to the CV splitter and the solver; `n_jobs=1`; `PYTHONHASHSEED=0`; gzip written with a fixed timestamp |
| 5 seeds in MLflow, mean ± SD | `src/run_seeds.py` logs seed, model, hyper-parameters, git commit, data SHA-256 and HF revision for every run |
| Exact environment | `requirements.txt` pins direct dependencies; `requirements-lock.txt` freezes the whole environment |
| Self-check | `python -m src.checks` verifies checksums, cohort separation and outputs |

## Stranger test · Prueba del extraño

Before submission the repository was cloned into an empty folder, a new virtual environment was created from `requirements.txt` only, and the pipeline was run from the committed processed table. The numbers matched the table above exactly.

## Limitations · Limitaciones

1. **Labels are machine-generated.** StudyChat's dialogue acts were assigned by GPT-4.1, not by human annotators. The classifier learns to reproduce those labels; its scores measure agreement with GPT-4.1, not with an expert ground truth.
2. **"Delegation" is an operational proxy.** Some delegation is implicit (e.g. pasting an error message, labelled `provide_context`), and some writing requests are legitimate. The label says nothing about virtue or intent.
3. **Cohort shift.** Macro-F1 drops from 0.81 (cross-validation) to 0.76 (new cohort), which is the realistic estimate for deployment on new students.
4. **Context.** One U.S. course, English prompts. The thesis targets Spanish-speaking introductory-programming students in Peru, so these numbers do not transfer; the pipeline does.
5. **Truncation.** 428 prompts longer than 5,000 characters (mostly pasted code) are truncated before vectorising.

## AI-use disclosure · Declaración de uso de IA

Claude (Anthropic) helped design the pipeline, write the code and this README, and run the stranger test. The author defined the task and its link to the thesis, reviewed the code and verified the results.

## Citation · Cita

McNichols, H., Ikram, F., & Lan, A. (2026). The StudyChat Dataset: Analyzing Student Dialogues With ChatGPT in an Artificial Intelligence Course. *Proceedings of the 16th International Learning Analytics and Knowledge Conference (LAK26)*, 53–63.
