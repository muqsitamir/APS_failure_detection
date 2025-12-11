import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from train import SEED
from pipelines.preprocessing import DropHighMissingColumns
from sklearn.pipeline import Pipeline


def build_pipelines(y_train: pd.Series, missing_threshold: float = 0.5):
    dropper = DropHighMissingColumns(threshold=missing_threshold)
    scorer = make_scorer(balanced_accuracy_score)

    pipelines = {}

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

