import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from pipelines.build_models import build_pipelines
from pipelines.eda import run_eda
from pipelines.preprocessing import encode_labels
from pipelines.tuning import tune_with_gridsearch, find_best_threshold_cv
from utils.comparison_utils import write_comparative_analysis, save_model_comparison_artifacts
from utils.data_utils import load_data

SEED = 42
np.random.seed(SEED)

GRID_N_JOBS = int(os.environ.get("GRID_N_JOBS", "1"))
GRID_BACKEND = os.environ.get("GRID_BACKEND", "threading")

DATA_DIR = os.environ.get("DATA_DIR", "./data")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./outputs")

if __name__ == "__main__":
    X_train, y_raw, X_test = load_data(DATA_DIR)
    y = encode_labels(y_raw)

    print("\nClass distribution (train):")
    print(y.value_counts(dropna=False))

    run_eda(X_train, y, OUTPUT_DIR)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    pipelines = build_pipelines(y_train=y, missing_threshold=0.5)

    all_results = []
    best = None

    for name, (pipe, grid, scorer) in pipelines.items():
        gs, results = tune_with_gridsearch(name, pipe, grid, scorer, X_train, y, cv)
        all_results.append(results)

        if best is None or gs.best_score_ > best[1].best_score_:
            best = (name, gs)

    experiments_df = pd.concat(all_results, ignore_index=True)
    experiments_path = os.path.join(OUTPUT_DIR, "experiments.csv")
    experiments_df.to_csv(experiments_path, index=False)
    print(f"\nSaved experiments table to: {experiments_path}")
    write_comparative_analysis(experiments_df=experiments_df, output_dir=OUTPUT_DIR)

    save_model_comparison_artifacts(experiments_df=experiments_df, output_dir=OUTPUT_DIR)

    # Print a small leaderboard
    top10 = experiments_df.sort_values(["mean_test_score", "std_test_score"], ascending=[False, True]).head(10)
    print("\nTop 10 CV results:")
    print(top10[["model", "mean_test_score", "std_test_score", "params"]].to_string(index=False))

    best_name, best_gs = best
    best_estimator = best_gs.best_estimator_

    best_threshold, threshold_df = find_best_threshold_cv(best_estimator, X_train, y, cv)
    if threshold_df is not None:
        thr_path = os.path.join(OUTPUT_DIR, "threshold_tuning.csv")
        threshold_df.to_csv(thr_path, index=False)
        print(f"\nSelected threshold (CV-tuned) for {best_name}: {best_threshold:.2f}")
        print(f"Saved threshold tuning table to: {thr_path}")
    else:
        print(f"\nUsing default threshold 0.50 for {best_name} (no predict_proba).")

    best_estimator.fit(X_train, y)

    if hasattr(best_estimator, "predict_proba"):
        test_probs = best_estimator.predict_proba(X_test)[:, 1]
        test_pred = (test_probs >= best_threshold).astype(int)
    else:
        test_pred = best_estimator.predict(X_test).astype(int)

    submission_df = pd.DataFrame(index=X_test.index)
    submission_df["Prediction"] = pd.Series(test_pred, index=X_test.index).map({0: "neg", 1: "pos"})

    sub_path = os.path.join(OUTPUT_DIR, "predictions.csv")
    submission_df.to_csv(sub_path, index_label="Id")
    print(f"\nWrote Kaggle submission to: {sub_path}")

    model_path = os.path.join(OUTPUT_DIR, "models/best_model.joblib")
    joblib.dump(best_estimator, model_path)
    print(f"Saved best model pipeline to: {model_path}")

    meta = {
        "seed": SEED,
        "data_dir": os.path.abspath(DATA_DIR),
        "output_dir": os.path.abspath(OUTPUT_DIR),
        "best_model": best_name,
        "best_params": best_gs.best_params_,
        "best_cv_balanced_accuracy": float(best_gs.best_score_),
        "threshold": float(best_threshold),
        "note": "All preprocessing is inside sklearn Pipelines to avoid leakage in CV.",
    }
    meta_path = os.path.join(OUTPUT_DIR, "run_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved run metadata to: {meta_path}")
