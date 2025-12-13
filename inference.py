import json

import joblib
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from pipelines.preprocessing import DropHighMissingColumns  # needed so joblib can unpickle the pipeline

evaluation = True
MODEL_PATH = "outputs/models/best_model.joblib"
META_PATH = "outputs/run_metadata.json"
DATA_DIR = Path("./data")

model = joblib.load(MODEL_PATH)

with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

threshold = float(meta.get("threshold", 0.5))

X_new = pd.read_csv(DATA_DIR / f"X_infer.csv")

if hasattr(model, "predict_proba"):
    probs = model.predict_proba(X_new)[:, 1]
    preds_int = (probs >= threshold).astype(int)
else:
    preds_int = model.predict(X_new).astype(int)

preds_label = pd.Series(preds_int).map({0: "neg", 1: "pos"})

pred_counts = preds_label.value_counts().reindex(["neg", "pos"]).fillna(0).astype(int)
fig, ax = plt.subplots()
ax.bar(pred_counts.index.astype(str), pred_counts.values)
ax.set_title(f"Predicted label distribution (threshold={threshold:.3f})")
ax.set_xlabel("Predicted label")
ax.set_ylabel("Count")
for i, v in enumerate(pred_counts.values):
    ax.text(i, v, str(v), ha="center", va="bottom")
fig.tight_layout()
plt.show()

if evaluation:
    Y_PATH = DATA_DIR / f"y_infer.csv"

    if Y_PATH.exists():
        y_df = pd.read_csv(Y_PATH)

        def _pick_label_series(df: pd.DataFrame) -> pd.Series:
            for name in ["label", "y", "target", "class", "failure", "status"]:
                if name in df.columns:
                    return df[name]

            for col in df.columns:
                s = df[col]
                vals = set(
                    pd.Series(s)
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .unique()
                    .tolist()
                )
                if vals and vals.issubset({"neg", "pos", "0", "1", "false", "true"}):
                    return s

            if df.shape[1] >= 2:
                first = df.iloc[:, 0]
                if first.is_unique:
                    return df.iloc[:, -1]

            return df.iloc[:, 0]

        y_series = _pick_label_series(y_df)

        if y_series.dtype.kind in ("O", "U", "S"):
            y_true = (
                y_series.astype(str)
                .str.strip()
                .str.lower()
                .map({"neg": 0, "pos": 1, "0": 0, "1": 1, "false": 0, "true": 1})
            )
        else:
            y_true = pd.to_numeric(y_series, errors="coerce")

        if y_true.isna().any():
            bad_vals = y_series[y_true.isna()].head(8).tolist()
            raise ValueError(
                f"Could not parse binary labels from {Y_PATH}. "
                f"Columns={list(y_df.columns)}. Sample unparsed values={bad_vals}"
            )

        y_true = y_true.astype(int)

        unique_labels = set(pd.Series(y_true).unique().tolist())
        if not unique_labels.issubset({0, 1}):
            raise ValueError(f"y contains non-binary labels: {sorted(unique_labels)}")

        preds_for_eval = pd.Series(preds_int)
        if len(y_true) != len(preds_for_eval):
            min_len = min(len(y_true), len(preds_for_eval))
            print(
                f"Warning: y length ({len(y_true)}) != preds length ({len(preds_for_eval)}). "
                f"Evaluating on the first {min_len} rows."
            )
            y_true = y_true.iloc[:min_len]
            preds_for_eval = preds_for_eval.iloc[:min_len]

        bal_acc = balanced_accuracy_score(y_true, preds_for_eval)

        report = classification_report(
            y_true,
            preds_for_eval,
            labels=[0, 1],
            target_names=["neg", "pos"],
            zero_division=0,
            output_dict=True,
        )

        cm = confusion_matrix(y_true, preds_for_eval, labels=[0, 1])
        fig, ax = plt.subplots()
        im = ax.imshow(cm)
        ax.set_title(f"Confusion matrix ()\nBalanced accuracy={bal_acc:.4f} | threshold={threshold:.3f}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks([0, 1], labels=["neg", "pos"])
        ax.set_yticks([0, 1], labels=["neg", "pos"])

        for (i, j), val in pd.DataFrame(cm).stack().items():
            ax.text(j, i, str(int(val)), ha="center", va="center")

        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        plt.show()

        metrics = ["precision", "recall", "f1-score"]
        classes = ["neg", "pos"]
        values = {
            m: [report[c][m] for c in classes]
            for m in metrics
        }

        x = range(len(classes))
        width = 0.25

        fig, ax = plt.subplots()
        ax.bar([i - width for i in x], values["precision"], width, label="precision")
        ax.bar(list(x), values["recall"], width, label="recall")
        ax.bar([i + width for i in x], values["f1-score"], width, label="f1-score")

        ax.set_title(f"Per-class metrics ()")
        ax.set_xticks(list(x), labels=classes)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Score")
        ax.legend()
        fig.tight_layout()
        plt.show()

        supports = [int(report[c]["support"]) for c in classes]
        fig, ax = plt.subplots()
        ax.bar(classes, supports)
        ax.set_title(f"Support by class ()")
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        for i, v in enumerate(supports):
            ax.text(i, v, str(v), ha="center", va="bottom")
        fig.tight_layout()
        plt.show()
    else:
        print(f"Skipping evaluation: labels file not found at {Y_PATH}")