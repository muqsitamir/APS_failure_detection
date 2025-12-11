# APS Failure Detection – End-to-End ML Pipeline

This repository implements an end-to-end machine learning pipeline for **APS (Air Pressure System) failure detection**.

It covers everything from **EDA and preprocessing** to **model selection**, **threshold tuning**, **artifact generation** with pre-trained whitebox ensemble models with high balanced accuracy for inference.

---

## Key Features

- **Fully reproducible training pipeline** in `main.py`
- **EDA**: automatic descriptive statistics & plots saved to disk
- **Robust preprocessing**:
  - Missing value handling
  - Label encoding (`neg` / `pos` → 0 / 1)
  - All preprocessing wrapped inside sklearn `Pipeline`s to avoid leakage
- **Model selection**:
  - Multiple candidate models defined in `pipelines/build_models.py`
  - `StratifiedKFold` cross-validation
  - `GridSearchCV` with configurable parallelism & backend
- **Threshold tuning**:
  - Cross-validated search for best decision threshold when `predict_proba` is available  
  - Tailored to scenarios where **false negatives are costly**
- **Artifact generation**:
  - `experiments.csv` with all CV results
  - Comparative analysis report & plots
  - `threshold_tuning.csv`
  - `predictions.csv`
  - `best_model.joblib` – persisted model pipeline
  - `run_metadata.json` – seeds, paths, chosen model, CV score, threshold

---

## Project Structure

- `main.py` – orchestrates the whole training + evaluation + submission flow
- `pipelines/`
  - `build_models.py` – defines sklearn pipelines, search grids, and scorers
  - `eda.py` – EDA routines and artifact generation
  - `preprocessing.py` – label encoding and other preprocessing utilities
  - `tuning.py` – GridSearch-based hyperparameter tuning and threshold search
- `utils/`
  - `data_utils.py` – data loading (`load_data(DATA_DIR)`)
  - `comparison_utils.py` – experiment aggregation & comparative analysis
- `data/` – (default) directory for input data (train/test)
- `outputs/` – (default) directory for all outputs & artifacts

> For the expected input file names and format, see `utils/data_utils.py`.

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>.git
   cd APS_failure_detection
   ```

2. **Create and activate a virtual env (recommended)**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

The script can be configured via environment variables:

- `DATA_DIR`  
  Directory containing the training/test data.  
  **Default:** `./data`

- `OUTPUT_DIR`  
  Directory where all artifacts (plots, CSVs, models) will be written.  
  **Default:** `./outputs`

- `GRID_N_JOBS`  
  Number of jobs for parallel grid search.  
  **Default:** `"1"` → `n_jobs=1`

- `GRID_BACKEND`  
  Backend for parallel grid search (e.g. `"threading"` or `"loky"`).  
  **Default:** `"threading"`

Example:
```bash
export DATA_DIR=./data
export OUTPUT_DIR=./outputs
export GRID_N_JOBS=4
export GRID_BACKEND=loky
```

---

## Preparing the Data

1. Download the APS dataset (or your own APS-like dataset).
2. Place the required files in `DATA_DIR` (default: `./data`).
3. Make sure the file names and format match what `utils/data_utils.py` expects
   (column names, separators, target column, etc.).

The main script relies on:

```python
X_train, y_raw, X_test = load_data(DATA_DIR)
```

So `load_data` must return:

- `X_train`: training features as a `pandas.DataFrame`
- `y_raw`: raw labels (e.g. `"neg"` / `"pos"`)
- `X_test`: test features as a `pandas.DataFrame`

> Tip: open `utils/data_utils.py` to verify the exact schema expected by the loader.

---

## Running Training (End-to-End)

Once data and environment variables are set, run:

```bash
python main.py
```

This will:

1. **Load data** using `load_data(DATA_DIR)`  
   - Returns:
     - `X_train` – training features (`pandas.DataFrame`)
     - `y_raw` – raw labels (e.g. `"neg"` / `"pos"`)
     - `X_test` – test features

2. **Encode labels**:

   ```python
   y = encode_labels(y_raw)  # e.g. {"neg": 0, "pos": 1}
   ```

3. **Print class distribution** of the training set.

4. **Run EDA**:

   ```python
   run_eda(X_train, y, OUTPUT_DIR)
   ```
   This generates plots and summary tables into `OUTPUT_DIR`.

5. **Build model pipelines**:

   ```python
   cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
   pipelines = build_pipelines(y_train=y, missing_threshold=0.5)
   ```

6. **Hyperparameter tuning & model selection**:

   For each pipeline:

   - Run grid search with the specified scorer and CV.
   - Collect results into a list of dataframes.
   - Track the best estimator.

   The combined CV results are saved as:

   - `experiments.csv` in `OUTPUT_DIR`
   - A comparative analysis report via `write_comparative_analysis`
   - Additional artifacts via `save_model_comparison_artifacts`

   The script also prints a small **leaderboard** (Top 10 CV results) to stdout.

7. **Threshold tuning**:

   ```python
   best_threshold, threshold_df = find_best_threshold_cv(best_estimator, X_train, y, cv)
   ```

   - If the model supports `predict_proba`, threshold tuning is performed and  
     results are saved to `threshold_tuning.csv` in `OUTPUT_DIR`.
   - Otherwise, a default threshold of `0.50` is used.

8. **Refit best model on full training data**:

   ```python
   best_estimator.fit(X_train, y)
   ```

9. **Generate predictions for test set**:

   ```python
   if hasattr(best_estimator, "predict_proba"):
       test_probs = best_estimator.predict_proba(X_test)[:, 1]
       test_pred = (test_probs >= best_threshold).astype(int)
   else:
       test_pred = best_estimator.predict(X_test).astype(int)

   predictions_df = pd.DataFrame(index=X_test.index)
   predictions_df["Prediction"] = (
       pd.Series(test_pred, index=X_test.index)
       .map({0: "neg", 1: "pos"})
   )
   ```

   The predictions are saved as:

   - `predictions.csv` in `OUTPUT_DIR`  
     (with `Id` as index and `"Prediction"` column).

10. **Persist the best model pipeline**:

    ```python
    model_path = os.path.join(OUTPUT_DIR, "best_model.joblib")
    joblib.dump(best_estimator, model_path)
    ```

11. **Save run metadata**:

    - `run_metadata.json` containing:
      - Seed
      - Absolute data/output paths
      - Best model name & params
      - Best CV balanced accuracy
      - Chosen threshold
      - Notes about preprocessing & leakage prevention

---

## Using the Trained Model for Inference

After running `main.py`, you will have at least:

- `outputs/best_model.joblib`
- `outputs/run_metadata.json`

You can load and use the model in your own scripts or notebooks:

```python
import json
import joblib
import numpy as np
import pandas as pd

# Paths to artifacts
MODEL_PATH = "outputs/best_model.joblib"
META_PATH = "outputs/run_metadata.json"

# Load model and metadata
model = joblib.load(MODEL_PATH)

with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

threshold = float(meta.get("threshold", 0.5))

# X_new must be a DataFrame with the same features & preprocessing expectations as X_train
X_new = pd.read_csv("path/to/your/new_data.csv")  # example

if hasattr(model, "predict_proba"):
    probs = model.predict_proba(X_new)[:, 1]
    preds_int = (probs >= threshold).astype(int)
else:
    preds_int = model.predict(X_new).astype(int)

preds_label = pd.Series(preds_int).map({0: "neg", 1: "pos"})
print(preds_label.head())
```

### Important Notes for Inference

- **Input schema** must match what the pipeline expects:
  - Same column names
  - Same types (numeric/categorical)
  - Missing values will be handled as during training (via the sklearn pipeline).
- The model encapsulates **all preprocessing steps**, so you should pass **raw features** in the same format as the training data (no manual preprocessing required).

---

## Reproducibility

- Random seed is fixed in `main.py` (e.g. `SEED = 42`).
- `run_metadata.json` stores:
  - The seed used
  - The selected best model and its hyperparameters
  - The cross-validated balanced accuracy
  - The decision threshold

This makes it easy to **reproduce** a given run and understand how the final model was obtained.

---

## License

Add your project’s license here, for example:

```text
MIT License – see LICENSE file for details.
```
