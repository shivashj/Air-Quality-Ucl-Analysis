"""
Label Generation Module

This module generates future air quality labels
based on sensor thresholds at t + 1 hour.
"""

import pandas as pd


def generate_future_labels(
    df: pd.DataFrame,
    target_columns: list,
    horizon: int = 1
) -> pd.Series:
    
    future_df = df[target_columns].shift(-horizon)

    unhealthy = pd.Series(False, index=df.index)

    for col in target_columns:
        dynamic_threshold = df[col].quantile(0.75)    #حد آستانه به صورت داینامیک ایجاد می شود
        unhealthy |= future_df[col] > dynamic_threshold

    return unhealthy.astype(int)
    """
    Generate binary labels for future air quality condition.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe sorted by time
    target_columns : list
        Pollutant columns used for labeling (e.g. CO, NO2)
    thresholds : dict
        Threshold values for pollutants
    horizon : int, optional
        Prediction horizon in hours, by default 1

    Returns
    -------
    pd.Series
        Binary labels (0 = Healthy, 1 = Unhealthy)
    """

    """df = df.copy()

    # Shift target columns to future (t + horizon)
    for col in target_columns:
        df[f"{col}_future"] = df[col].shift(-horizon)

    # Initialize label as healthy
    label = pd.Series(0, index=df.index)

    # Apply rule-based labeling
    for col in target_columns:
        label |= df[f"{col}_future"] > thresholds[col]

    # Convert boolean to int (0 / 1)
    label = label.astype(int)

    return label"""
