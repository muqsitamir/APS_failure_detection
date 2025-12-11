import matplotlib.pyplot as plt

from utils.plotting_utils import plot_class_balance, plot_pca, plot_feature_importance, load_and_prep

if __name__ == "__main__":
    plt.style.use('ggplot')

    X_orig, X_scaled, y, feat_names = load_and_prep()

    plot_class_balance(y)
    plot_pca(X_scaled, y)
    plot_feature_importance(X_scaled, y, feat_names)
    print("Visualizations generated.")
