import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


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
