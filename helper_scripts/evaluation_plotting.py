import ast
import os
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = "../outputs"
EXPERIMENTS_PATH = os.path.join(OUTPUT_DIR, "experiments.csv")
THRESHOLD_PATH = os.path.join(OUTPUT_DIR, "threshold_tuning.csv")


experiments_df = pd.read_csv(EXPERIMENTS_PATH)

if "params" in experiments_df.columns:
    params_series = experiments_df["params"].apply(ast.literal_eval)

    params_df = pd.DataFrame(params_series.tolist())

    experiments_df = pd.concat(
        [experiments_df.drop(columns=["params"]), params_df],
        axis=1
    )

    print("Expanded params into columns:", list(params_df.columns))


best_per_model = (
    experiments_df.sort_values("mean_test_score", ascending=False)
    .groupby("model", as_index=False)
    .first()
)

plt.figure()
plt.bar(best_per_model["model"], best_per_model["mean_test_score"],
        yerr=best_per_model["std_test_score"], capsize=4)
plt.ylabel("Mean CV balanced accuracy")
plt.ylim(0.9, 1.0)  # adjust if needed
plt.tight_layout()

fig_a_path = os.path.join(OUTPUT_DIR, "fig_best_models.png")
plt.savefig(fig_a_path, dpi=300)
plt.close()
print(f"Saved Figure A to {fig_a_path}")

xgb_df = experiments_df[experiments_df["model"] == "XGBoost"].copy()

if not xgb_df.empty:
    plt.figure()
    for depth, group in xgb_df.groupby("clf__max_depth"):
        g_sorted = group.sort_values("clf__n_estimators")
        plt.plot(g_sorted["clf__n_estimators"], g_sorted["mean_test_score"],
                 marker="o", label=f"max_depth={depth}")

    plt.xlabel("n_estimators")
    plt.ylabel("Mean CV balanced accuracy")
    plt.legend()
    plt.tight_layout()

    fig_b_path = os.path.join(OUTPUT_DIR, "fig_xgb_n_estimators.png")
    plt.savefig(fig_b_path, dpi=300)
    plt.close()
    print(f"Saved Figure B to {fig_b_path}")
else:
    print("No XGBoost rows found in experiments.csv – skipping Figure B.")

plt.figure()
markers = {"LogisticRegression": "o", "RandomForest": "s", "XGBoost": "^"}

for model_name, group in experiments_df.groupby("model"):
    plt.scatter(group["mean_train_score"], group["mean_test_score"],
                label=model_name, marker=markers.get(model_name, "o"), alpha=0.7)

plt.xlabel("Mean train balanced accuracy")
plt.ylabel("Mean CV balanced accuracy")
plt.plot([0.9, 1.0], [0.9, 1.0], linestyle="--")  # diagonal reference
plt.xlim(0.9, 1.0)
plt.ylim(0.9, 1.0)
plt.legend()
plt.tight_layout()

fig_c_path = os.path.join(OUTPUT_DIR, "fig_train_vs_cv.png")
plt.savefig(fig_c_path, dpi=300)
plt.close()
print(f"Saved Figure C to {fig_c_path}")

if os.path.exists(THRESHOLD_PATH):
    thr_df = pd.read_csv(THRESHOLD_PATH)

    threshold_col = "threshold"
    score_col = "mean_score"

    if threshold_col in thr_df.columns:
        if score_col not in thr_df.columns:
            numeric_cols = thr_df.select_dtypes(include="number").columns.tolist()
            numeric_cols = [c for c in numeric_cols if c != threshold_col]
            if numeric_cols:
                score_col = numeric_cols[0]
            else:
                raise ValueError("Could not find a numeric score column in threshold_tuning.csv")

        thr_sorted = thr_df.sort_values(threshold_col)

        plt.figure()
        plt.plot(thr_sorted[threshold_col], thr_sorted[score_col], marker="o")
        plt.xlabel("Decision threshold")
        plt.ylabel("Mean CV balanced accuracy")
        plt.tight_layout()

        fig_d_path = os.path.join(OUTPUT_DIR, "fig_threshold_tuning.png")
        plt.savefig(fig_d_path, dpi=300)
        plt.close()
        print(f"Saved Figure D to {fig_d_path}")
    else:
        print("No 'threshold' column in threshold_tuning.csv – skipping Figure D.")
else:
    print("threshold_tuning.csv not found – skipping Figure D.")