#evaluation/featur_importance
import pandas as pd
import matplotlib.pyplot as plt

def plot_feature_importance(model, feature_names, save_path=None):
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df["feature"], importance_df["importance"])
    plt.gca().invert_yaxis()
    plt.title("Feature Importance")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()

    return importance_df


