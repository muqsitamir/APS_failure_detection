import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from pipelines.preprocessing import DropHighMissingColumns # needed so joblib can unpickle the pipeline

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

SEED = 42
DATA_DIR = "../data"
OUTPUT_DIR = "../outputs"

# --- load data and model ---
X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"), index_col=0)
y = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv"), index_col=0)["Prediction"]
y = y.map({"neg": 0, "pos": 1}).astype(int)

best_model_path = os.path.join(OUTPUT_DIR, "models", "best_model.joblib")
best_estimator = joblib.load(best_model_path)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

y_proba = cross_val_predict(
    best_estimator, X_train, y, cv=cv, method="predict_proba"
)[:, 1]

precision, recall, pr_thresholds = precision_recall_curve(y, y_proba)
pr_auc = auc(recall, precision)

plt.figure()
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.tight_layout()
pr_path = os.path.join(OUTPUT_DIR, "fig_pr_curve.png")
plt.savefig(pr_path, dpi=300)
plt.close()
print(f"Saved PR curve to {pr_path}")

# --- ROC curve ---
fpr, tpr, roc_thresholds = roc_curve(y, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.tight_layout()
roc_path = os.path.join(OUTPUT_DIR, "fig_roc_curve.png")
plt.savefig(roc_path, dpi=300)
plt.close()
print(f"Saved ROC curve to {roc_path}")

BEST_THRESHOLD = 0.2

y_pred = (y_proba >= BEST_THRESHOLD).astype(int)
cm = confusion_matrix(y, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, "fig_confusion_matrix.png")
plt.savefig(cm_path, dpi=300)
plt.close()
print(f"Saved confusion matrix to {cm_path}")