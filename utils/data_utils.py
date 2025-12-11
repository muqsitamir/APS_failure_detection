import os

import pandas as pd

from main import DATA_DIR


def load_data_for_eda():
    print("Loading data...")
    X_train = pd.read_csv('X_train.csv')
    y_train = pd.read_csv('y_train.csv')
    X_test = pd.read_csv('X_test.csv')
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    return X_train, y_train, X_test

def load_data(data_dir: str = DATA_DIR):
    """Load datasets. Assumes X_train.csv, y_train.csv, X_test.csv exist in data_dir."""
    print(f"Loading datasets from: {os.path.abspath(data_dir)}")

    read_kwargs = {
        "na_values": ["na", "NA", "NaN", ""],
        "keep_default_na": True,
    }

    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"), **read_kwargs)
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv"), **read_kwargs)
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"), **read_kwargs)

    # Align by Id if present
    if "Id" in X_train.columns:
        X_train = X_train.set_index("Id")
    if "Id" in X_test.columns:
        X_test = X_test.set_index("Id")
    if "Id" in y_train.columns:
        y_train = y_train.set_index("Id")

    # y_train expected to have a single column
    if y_train.shape[1] != 1:
        raise ValueError(f"Expected y_train to have 1 column, got {y_train.shape[1]}")

    return X_train, y_train.iloc[:, 0], X_test

