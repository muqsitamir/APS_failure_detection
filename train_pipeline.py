import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None



SEED = 42
np.random.seed(SEED)

# GridSearchCV parallelism controls.
# Default is 1 to avoid joblib/loky hangs on some macOS setups and to avoid nested parallelism
# (since RandomForest/XGBoost already parallelize internally).
GRID_N_JOBS = int(os.environ.get("GRID_N_JOBS", "1"))
# If GRID_N_JOBS > 1, use threading by default to avoid process pickling/IPC issues with pandas DataFrames.
GRID_BACKEND = os.environ.get("GRID_BACKEND", "threading")

# Per assignment: define input directory at the beginning
DATA_DIR = os.environ.get("DATA_DIR", ".")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", ".")


def run_eda(X_train: pd.DataFrame, y: pd.Series, output_dir: str) -> None:
    """Run lightweight EDA and write tables/figures to disk.

    Produces (under OUTPUT_DIR/eda):
      - overview.json
      - missingness.csv
      - feature_summary.csv
      - target_correlation_top.csv
      - modeling_drop_thresholds.csv
      - *.png plots
    """
    eda_dir = os.path.join(output_dir, "eda")
    os.makedirs(eda_dir, exist_ok=True)

    n_rows, n_cols = X_train.shape
    class_counts = y.value_counts(dropna=False).to_dict()

    # Missingness
    missing_ratio = X_train.isnull().mean(axis=0).sort_values(ascending=False)
    missing_count = X_train.isnull().sum(axis=0).loc[missing_ratio.index]
    missing_df = pd.DataFrame({
        "missing_ratio": missing_ratio,
        "missing_count": missing_count,
        "dtype": X_train.dtypes.astype(str).loc[missing_ratio.index],
    })
    missing_df.to_csv(os.path.join(eda_dir, "missingness.csv"), index=True)

    # Numeric feature summary
    desc = X_train.describe(include=[np.number]).T
    desc["missing_ratio"] = X_train.isnull().mean(axis=0)
    desc["missing_count"] = X_train.isnull().sum(axis=0)
    desc["n_unique"] = X_train.nunique(dropna=True)
    desc.sort_values("missing_ratio", ascending=False).to_csv(
        os.path.join(eda_dir, "feature_summary.csv"), index=True
    )

    # Plot: missingness distribution
    plt.figure()
    plt.hist(missing_ratio.values, bins=30)
    plt.title("Distribution of missing-value ratios across features")
    plt.xlabel("Missing-value ratio")
    plt.ylabel("Number of features")
    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, "missingness_hist.png"), dpi=160)
    plt.close()

    # Plot: top missing features
    topk = min(25, len(missing_ratio))
    plt.figure()
    plt.bar(range(topk), missing_ratio.values[:topk])
    plt.title(f"Top {topk} features by missing-value ratio")
    plt.xlabel("Feature rank")
    plt.ylabel("Missing-value ratio")
    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, "missingness_top25.png"), dpi=160)
    plt.close()

    # Plot: class balance
    vc = y.value_counts(dropna=False).sort_index()
    plt.figure()
    plt.bar([str(i) for i in vc.index], vc.values)
    plt.title("Class distribution (train)")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, "class_balance.png"), dpi=160)
    plt.close()

    # Drop-threshold sensitivity
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    drop_rows = []
    for t in thresholds:
        n_drop = int((missing_ratio > t).sum())
        drop_rows.append({"threshold": t, "n_dropped": n_drop, "n_remaining": int(n_cols - n_drop)})
    drop_df = pd.DataFrame(drop_rows)
    drop_df.to_csv(os.path.join(eda_dir, "modeling_drop_thresholds.csv"), index=False)

    plt.figure()
    plt.plot(drop_df["threshold"], drop_df["n_dropped"], marker="o")
    plt.title("How many features would be dropped vs. missingness threshold")
    plt.xlabel("Drop threshold (missing ratio)")
    plt.ylabel("# dropped features")
    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, "dropped_features_vs_threshold.png"), dpi=160)
    plt.close()

    # Correlation with target (median-imputed numeric)
    X_num = X_train.select_dtypes(include=[np.number]).copy()
    medians = X_num.median(axis=0, skipna=True)
    X_imp = X_num.fillna(medians)

    y_arr = y.to_numpy(dtype=float)
    corr_rows = []
    for col in X_imp.columns:
        x = X_imp[col].to_numpy(dtype=float)
        if np.nanstd(x) == 0:
            r = 0.0
        else:
            r = float(np.corrcoef(x, y_arr)[0, 1])
            if np.isnan(r):
                r = 0.0
        corr_rows.append((col, r, abs(r)))

    corr_df = pd.DataFrame(corr_rows, columns=["feature", "corr_with_target", "abs_corr"]).sort_values(
        "abs_corr", ascending=False
    )
    corr_df.to_csv(os.path.join(eda_dir, "target_correlation_top.csv"), index=False)

    topc = corr_df.head(20).copy()
    plt.figure()
    plt.bar(range(len(topc)), topc["corr_with_target"].values)
    plt.title("Top 20 features by absolute correlation with target (median-imputed)")
    plt.xlabel("Feature rank")
    plt.ylabel("Correlation with target")
    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, "target_corr_top20.png"), dpi=160)
    plt.close()

    overview = {
        "n_rows": int(n_rows),
        "n_features": int(n_cols),
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "missing_ratio_max": float(missing_ratio.iloc[0]) if len(missing_ratio) else 0.0,
        "missing_ratio_median": float(missing_ratio.median()) if len(missing_ratio) else 0.0,
        "notes": [
            "EDA outputs are saved under OUTPUT_DIR/eda.",
            "Correlations are computed on median-imputed numeric features for exploratory purposes only.",
        ],
    }
    with open(os.path.join(eda_dir, "overview.json"), "w", encoding="utf-8") as f:
        json.dump(overview, f, indent=2)

    print(f"\n[EDA] Wrote EDA tables/figures to: {eda_dir}")


def load_data(data_dir: str = DATA_DIR):
    """Load datasets. Assumes X_train.csv, y_train.csv, X_test.csv exist in data_dir."""
    print(f"Loading datasets from: {os.path.abspath(data_dir)}")

    read_kwargs = {
        "na_values": ["na", "NA", "NaN", ""],
        "keep_default_na": True,
    }

    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"), **read_kwargs)
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv"), **read_kwargs)
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"), **read_kwargs)

    # Align by Id if present
    if "Id" in X_train.columns:
        X_train = X_train.set_index("Id")
    if "Id" in X_test.columns:
        X_test = X_test.set_index("Id")
    if "Id" in y_train.columns:
        y_train = y_train.set_index("Id")

    # y_train expected to have a single column
    if y_train.shape[1] != 1:
        raise ValueError(f"Expected y_train to have 1 column, got {y_train.shape[1]}")

    return X_train, y_train.iloc[:, 0], X_test


class DropHighMissingColumns(BaseEstimator, TransformerMixin):
    """Drop columns whose missing-value ratio exceeds a threshold.

    IMPORTANT: This transformer must be fitted ONLY on training folds to avoid leakage.
    Placing it inside an sklearn Pipeline ensures that.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.cols_to_drop_ = None
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.feature_names_in_ = list(getattr(X, "columns", X_df.columns))
        missing_ratio = X_df.isnull().mean(axis=0)
        self.cols_to_drop_ = list(missing_ratio[missing_ratio > self.threshold].index)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)

        # If original had column names, restore them for consistent dropping
        if self.feature_names_in_ is not None and len(self.feature_names_in_) == X_df.shape[1]:
            X_df.columns = self.feature_names_in_

        if self.cols_to_drop_:
            X_df = X_df.drop(columns=self.cols_to_drop_, errors="ignore")

        return X_df


def encode_labels(y_raw: pd.Series) -> pd.Series:
    """Map labels to {0,1}. Supports 'neg'/'pos' or already-numeric labels."""
    if y_raw.dtype == object:
        y = y_raw.map({"neg": 0, "pos": 1})
        if y.isnull().any():
            bad = y_raw[y.isnull()].unique()
            raise ValueError(f"Unexpected label values: {bad}")
        return y.astype(int)

    # If already numeric/bool
    return y_raw.astype(int)


def build_pipelines(y_train: pd.Series, missing_threshold: float = 0.5):
    """Create model pipelines and their parameter grids."""

    dropper = DropHighMissingColumns(threshold=missing_threshold)
    scorer = make_scorer(balanced_accuracy_score)

    pipelines = {}

    # Logistic Regression (whitebox; scaled)
    lr_pipe = Pipeline(
        steps=[
            ("drop_missing", dropper),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    random_state=SEED,
                    class_weight="balanced",
                    max_iter=4000,
                    tol=1e-3,
                    solver="saga",
                    n_jobs=(os.cpu_count() or 1),
                ),
            ),
        ]
    )

    lr_grid = {
        "clf__C": [0.01, 0.1, 1.0],
        "clf__penalty": ["l1", "l2"],
    }
    #  very large C with L1 is extremely slow on this dataset; keeping the C value modest for runtime.

    pipelines["LogisticRegression"] = (lr_pipe, lr_grid, scorer)

    # Random Forest (no scaling necessary)
    rf_pipe = Pipeline(
        steps=[
            ("drop_missing", dropper),
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                RandomForestClassifier(
                    random_state=SEED,
                    class_weight="balanced",
                    n_jobs=(os.cpu_count() or 1),
                ),
            ),
        ]
    )

    rf_grid = {
        "clf__n_estimators": [300, 600],
        "clf__max_depth": [None, 10, 20],
        "clf__min_samples_leaf": [1, 5, 10],
        "clf__max_features": ["sqrt", 0.3, 0.5],
    }

    pipelines["RandomForest"] = (rf_pipe, rf_grid, scorer)

    # XGBoost (optional; more black-box, use sparingly + justify in report)
    if XGBClassifier is not None:
        neg = int(np.sum(y_train == 0))
        pos = int(np.sum(y_train == 1))
        scale_pos_weight = (neg / pos) if pos > 0 else 1.0

        xgb_pipe = Pipeline(
            steps=[
                ("drop_missing", dropper),
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    XGBClassifier(
                        random_state=SEED,
                        n_jobs=(os.cpu_count() or 1),
                        eval_metric="logloss",
                        scale_pos_weight=scale_pos_weight,
                        tree_method="hist",
                    ),
                ),
            ]
        )

        xgb_grid = {
            "clf__n_estimators": [300, 600],
            "clf__max_depth": [3, 5],
            "clf__learning_rate": [0.05, 0.1],
            "clf__subsample": [0.8, 1.0],
            "clf__colsample_bytree": [0.8, 1.0],
        }

        pipelines["XGBoost"] = (xgb_pipe, xgb_grid, scorer)

    return pipelines


def tune_with_gridsearch(name: str, pipe: Pipeline, grid: dict, scorer, X, y, cv):
    """Grid search with leakage-safe CV (preprocessing is inside Pipeline)."""
    print(f"\nTuning {name}...")
    gs = GridSearchCV(
        estimator=pipe,
        param_grid=grid,
        scoring=scorer,
        cv=cv,
        n_jobs=GRID_N_JOBS,
        refit=True,
        return_train_score=True,
        verbose=2,
        error_score="raise",
    )
    if GRID_N_JOBS > 1:
        # Use threads to avoid loky process overhead / hangs with pandas objects.
        with joblib.parallel_backend(GRID_BACKEND):
            gs.fit(X, y)
    else:
        gs.fit(X, y)

    best_score = gs.best_score_
    print(f"Best CV balanced accuracy for {name}: {best_score:.5f}")
    print(f"Best params for {name}: {gs.best_params_}")

    results = pd.DataFrame(gs.cv_results_)
    results.insert(0, "model", name)

    # Keep a compact set of columns for the report
    keep_cols = [
        "model",
        "mean_test_score",
        "std_test_score",
        "mean_train_score",
        "std_train_score",
        "rank_test_score",
        "params",
    ]
    results = results[keep_cols].sort_values(["rank_test_score", "mean_test_score"], ascending=[True, False])

    return gs, results


def find_best_threshold_cv(estimator: Pipeline, X, y, cv, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)

    if not hasattr(estimator, "predict_proba"):
        return 0.5, None

    all_probs = np.zeros(len(y), dtype=float)

    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr = y.iloc[train_idx]

        est = clone(estimator)
        est.fit(X_tr, y_tr)
        probs = est.predict_proba(X_val)[:, 1]
        all_probs[val_idx] = probs

    best_t = 0.5
    best_score = -1.0
    scores = []

    for t in thresholds:
        preds = (all_probs >= t).astype(int)
        score = balanced_accuracy_score(y, preds)
        scores.append((t, score))
        if score > best_score:
            best_score = score
            best_t = float(t)

    threshold_df = pd.DataFrame(scores, columns=["threshold", "balanced_accuracy"])

    return best_t, threshold_df


def save_model_comparison_artifacts(experiments_df: pd.DataFrame, output_dir: str) -> None:
    best_per_model = experiments_df.loc[experiments_df.groupby("model")["mean_test_score"].idxmax()]
    best_overall = best_per_model.loc[best_per_model["mean_test_score"].idxmax()]

    lines = []
    lines.append("# Model Comparison Report\n")
    lines.append(f"Best overall model: **{best_overall['model']}** with mean CV balanced accuracy "
                 f"{best_overall['mean_test_score']:.4f}\n")

    table_df = best_per_model[["model", "mean_test_score", "std_test_score", "mean_train_score", "std_train_score", "params"]].copy()
    lines.append("```\n" + table_df.to_string(index=False) + "\n```")

    report_path = os.path.join(output_dir, "model_comparison_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nSaved model comparison report to: {report_path}")




def write_comparative_analysis(experiments_df: pd.DataFrame, output_dir: str) -> None:
    """Comparative analysis across model families and configs.

    Writes:
      - comparisons/best_per_model.csv
      - comparisons/best_per_model_with_gap.csv
      - comparisons/best_cv_score_by_model.png
      - comparisons/train_test_gap_by_model.png
      - comparisons/comparative_analysis.md
    """
    comp_dir = os.path.join(output_dir, "comparisons")
    _ensure_dir(comp_dir)

    # Best per model family
    best_per_model = (
        experiments_df.sort_values(["mean_test_score", "std_test_score"], ascending=[False, True])
        .groupby("model", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    best_per_model.to_csv(os.path.join(comp_dir, "best_per_model.csv"), index=False)

    # Overfit indicator
    tmp = best_per_model.copy()
    tmp["train_minus_test"] = tmp["mean_train_score"] - tmp["mean_test_score"]
    tmp = tmp.sort_values("mean_test_score", ascending=False)
    tmp.to_csv(os.path.join(comp_dir, "best_per_model_with_gap.csv"), index=False)

    # Plot: best CV score by model
    import matplotlib.pyplot as plt
    plt.figure()
    plt.bar(tmp["model"].astype(str).tolist(), tmp["mean_test_score"].astype(float).tolist())
    plt.title("Best CV balanced accuracy per model family")
    plt.xlabel("Model")
    plt.ylabel("Mean CV balanced accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(comp_dir, "best_cv_score_by_model.png"), dpi=160)
    plt.close()

    # Plot: train-test gap
    plt.figure()
    plt.bar(tmp["model"].astype(str).tolist(), tmp["train_minus_test"].astype(float).tolist())
    plt.title("Train–test CV gap for best config (overfitting indicator)")
    plt.xlabel("Model")
    plt.ylabel("Mean train - mean test")
    plt.tight_layout()
    plt.savefig(os.path.join(comp_dir, "train_test_gap_by_model.png"), dpi=160)
    plt.close()

    # Markdown snippet (avoid to_markdown dependency)
    table_df = tmp[["model", "mean_test_score", "std_test_score", "mean_train_score", "std_train_score", "train_minus_test", "params"]].copy()
    md_lines = []
    md_lines.append("# Comparative analysis (auto-generated)\n")
    md_lines.append("Best configuration per model family (sorted by mean CV balanced accuracy):\n")
    md_lines.append("```\n" + table_df.to_string(index=False) + "\n```\n")
    md_lines.append("Interpretation: larger `train_minus_test` suggests a higher risk of overfitting.\n")

    with open(os.path.join(comp_dir, "comparative_analysis.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"[Comparisons] Wrote comparative analysis to: {comp_dir}")


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def main():
    X_train, y_raw, X_test = load_data(DATA_DIR)
    y = encode_labels(y_raw)

    print("\nClass distribution (train):")
    print(y.value_counts(dropna=False))

    # Run EDA
    run_eda(X_train, y, OUTPUT_DIR)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    pipelines = build_pipelines(y_train=y, missing_threshold=0.5)

    all_results = []
    best = None  # (name, gridsearch)

    for name, (pipe, grid, scorer) in pipelines.items():
        gs, results = tune_with_gridsearch(name, pipe, grid, scorer, X_train, y, cv)
        all_results.append(results)

        if best is None or gs.best_score_ > best[1].best_score_:
            best = (name, gs)

    # Save experiment table for the report
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

    # Optional: threshold tuning (helps when false negatives are costly)
    best_threshold, threshold_df = find_best_threshold_cv(best_estimator, X_train, y, cv)
    if threshold_df is not None:
        thr_path = os.path.join(OUTPUT_DIR, "threshold_tuning.csv")
        threshold_df.to_csv(thr_path, index=False)
        print(f"\nSelected threshold (CV-tuned) for {best_name}: {best_threshold:.2f}")
        print(f"Saved threshold tuning table to: {thr_path}")
    else:
        print(f"\nUsing default threshold 0.50 for {best_name} (no predict_proba).")

    # Fit best estimator on full training data
    best_estimator.fit(X_train, y)

    # Predict on test
    if hasattr(best_estimator, "predict_proba"):
        test_probs = best_estimator.predict_proba(X_test)[:, 1]
        test_pred = (test_probs >= best_threshold).astype(int)
    else:
        test_pred = best_estimator.predict(X_test).astype(int)

    submission_df = pd.DataFrame(index=X_test.index)
    submission_df["Prediction"] = pd.Series(test_pred, index=X_test.index).map({0: "neg", 1: "pos"})

    sub_path = os.path.join(OUTPUT_DIR, "submission.csv")
    submission_df.to_csv(sub_path, index_label="Id")
    print(f"\nWrote Kaggle submission to: {sub_path}")

    # Persist model for reproducibility
    model_path = os.path.join(OUTPUT_DIR, "best_model.joblib")
    joblib.dump(best_estimator, model_path)
    print(f"Saved best model pipeline to: {model_path}")

    # Save run metadata (seeds, chosen model, params, threshold)
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


if __name__ == "__main__":
    main()
