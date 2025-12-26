"""
Model Training Module

This module trains a Random Forest classifier
for time-aware air quality prediction.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def train_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float = 0.8,   
    random_state: int = 42
):
    """
    Train Random Forest model using time-based split.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Labels
    train_ratio : float, optional
        Train/Test split ratio, by default 0.8
    random_state : int, optional
        Random seed

    Returns
    -------
    model : RandomForestClassifier
        Trained model
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test labels
    y_pred : array
        Predicted labels
    y_prob : array
        Predicted probabilities
    """

    split_index = int(len(X) * train_ratio)

    # Time-based split
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probability of Unhealthy

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    return model, X_train, X_test, y_train, y_test, y_pred, y_prob
