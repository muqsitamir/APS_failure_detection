import json
import joblib
import pandas as pd
from pipelines.preprocessing import DropHighMissingColumns  # noqa: F401  # needed so joblib can unpickle the pipeline

MODEL_PATH = "outputs/models/best_model.joblib"
META_PATH = "outputs/run_metadata.json"

model = joblib.load(MODEL_PATH)

with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

threshold = float(meta.get("threshold", 0.5))

X_new = pd.read_csv("./data/X_test.csv")

if hasattr(model, "predict_proba"):
    probs = model.predict_proba(X_new)[:, 1]
    preds_int = (probs >= threshold).astype(int)
else:
    preds_int = model.predict(X_new).astype(int)

preds_label = pd.Series(preds_int).map({0: "neg", 1: "pos"})
print(preds_label.head())
print(preds_label.describe())