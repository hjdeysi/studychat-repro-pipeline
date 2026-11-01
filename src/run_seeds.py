"""Stage 5 - Run 5 seeds, log every run in MLflow, report mean +/- SD across seeds.

Run:  PYTHONHASHSEED=0 python -m src.run_seeds
Writes reports/results_by_seed.csv, reports/results_summary.csv and
reports/selected_model.json (committed) and mlflow.db (local, git-ignored).
"""
import json

import mlflow
import pandas as pd

from src import config, train
from src.utils import git_commit, read_manifest, require_hash_seed, write_json

METRICS = ["cv_f1_macro", "test_f1_macro", "test_balanced_accuracy", "test_precision_delegation",
           "test_recall_delegation", "test_roc_auc", "test_unseen_f1_macro"]


def main() -> pd.DataFrame:
    require_hash_seed()
    mlflow.set_tracking_uri(f"sqlite:///{config.ROOT / 'mlflow.db'}")
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
    commit = git_commit()
    data_sha = read_manifest().get("processed", {}).get("sha256", "unknown")
    df = train.load()

    rows, selected = [], []
    for seed in config.SEEDS:
        res = train.main(seed, df)
        best = train.select(res)
        selected.append({"seed": seed, "model": best["model"], "best_params": best["best_params"],
                         "test_f1_macro": best["test_f1_macro"]})
        for r in res:
            with mlflow.start_run(run_name=f"seed-{seed}-{r['model']}"):
                mlflow.log_params({"seed": seed, "model": r["model"], "best_params": r["best_params"],
                                   "train": config.TRAIN_SEMESTER, "test": config.TEST_SEMESTER,
                                   "cv": f"StratifiedGroupKFold({config.N_FOLDS}, groups=user_id)",
                                   "max_chars": config.MAX_CHARS, "git_commit": commit,
                                   "data_sha256": data_sha, "hf_revision": config.HF_REVISION})
                mlflow.log_metrics({m: r[m] for m in METRICS})
                mlflow.set_tag("selected", str(r["model"] == best["model"]))
            rows.append(r)
        print(f"[seed {seed}] selected {best['model']} {best['best_params']} "
              f"test_f1_macro={best['test_f1_macro']}")

    by_seed = pd.DataFrame(rows)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    by_seed.to_csv(config.REPORTS_DIR / "results_by_seed.csv", index=False)

    summary = by_seed.groupby("model", sort=False)[METRICS].agg(["mean", "std"]).round(4)
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]  # std = sample SD (ddof=1)
    summary = summary.reset_index()
    summary.to_csv(config.REPORTS_DIR / "results_summary.csv", index=False)

    sel = pd.DataFrame(selected)
    write_json(config.REPORTS_DIR / "selected_model.json", {
        "rule": "best cross-validation macro-F1 on the training cohort; test used once",
        "per_seed": selected,
        "test_f1_macro_mean": round(float(sel.test_f1_macro.mean()), 4),
        "test_f1_macro_sd": round(float(sel.test_f1_macro.std(ddof=1)), 4),
        "git_commit": commit,
    })
    print("\n" + summary.to_string(index=False))
    print(f"\nSelected model, test macro-F1 across {len(config.SEEDS)} seeds: "
          f"{sel.test_f1_macro.mean():.4f} +/- {sel.test_f1_macro.std(ddof=1):.4f} (mean +/- sample SD)")
    return summary


if __name__ == "__main__":
    main()
