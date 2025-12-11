import joblib
import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from train import GRID_BACKEND, GRID_N_JOBS


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
