import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def load_and_prep():
    print("Loading data for visualization...")
    X_train = pd.read_csv('X_train.csv')
    y_train = pd.read_csv('y_train.csv')

    # Map labels
    unique_labels = y_train.iloc[:, 0].unique()
    # Assuming 'neg'/'pos' structure based on previous steps
    if 'neg' in unique_labels:
        y = y_train.iloc[:, 0].map({'neg': 0, 'pos': 1})
    else:
        y = y_train.iloc[:, 0]  # Already numeric?

    # Drop high missing (same as training pipeline)
    missing = X_train.isnull().mean()
    X = X_train.drop(columns=missing[missing > 0.5].index)

    # Impute & Scale for PCA/Model
    imp = SimpleImputer(strategy='median')
    X_imp = imp.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    return X, X_scaled, y, X.columns


def plot_class_balance(y):
    print("Plotting Class Balance...")
    plt.figure(figsize=(6, 4))
    counts = y.value_counts()
    sns.barplot(x=counts.index, y=counts.values)
    plt.title('Class Distribution (neg=0, pos=1)')
    plt.ylabel('Count')
    plt.xlabel('Class')
    plt.yscale('log')  # Log scale because difference is huge
    plt.savefig('class_balance.png')
    plt.close()


def plot_feature_importance(X_scaled, y, feature_names):
    print("Plotting Feature Importance (Random Forest)...")
    # Train a quick RF to get importance
    rf = RandomForestClassifier(n_estimators=20, random_state=42, class_weight='balanced')
    rf.fit(X_scaled, y)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Top 20 features
    top_n = 20
    plt.figure(figsize=(10, 8))
    plt.title(f'Top {top_n} Feature Importances (Random Forest)')
    plt.barh(range(top_n), importances[indices[:top_n]], align='center')
    plt.yticks(range(top_n), [feature_names[i] for i in indices[:top_n]])
    plt.xlabel('Relative Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()


def plot_pca(X_scaled, y):
    print("Plotting PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    # Downsample negative class for visibility if needed, but let's plot all
    # Plot separately to handle legend
    plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], alpha=0.3, label='Negative', s=10, c='blue')
    plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], alpha=0.8, label='Positive (outliers?)', s=20, c='red')

    plt.title('PCA 2D Projection')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.savefig('pca_projection.png')
    plt.close()