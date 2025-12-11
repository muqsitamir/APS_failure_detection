import json
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


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
