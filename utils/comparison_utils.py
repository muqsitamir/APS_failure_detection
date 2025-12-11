import os

import pandas as pd


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def write_comparative_analysis(experiments_df: pd.DataFrame, output_dir: str) -> None:
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
