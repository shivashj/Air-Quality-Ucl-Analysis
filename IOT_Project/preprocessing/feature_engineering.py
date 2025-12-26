"""
Feature Engineering Module

This module is responsible for:
- Extracting temporal features from timestamps
- Selecting sensor features
- Creating final feature matrix for modeling
"""

import pandas as pd


def extract_time_features(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    """
    Extract temporal features from timestamp column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing timestamp column
    timestamp_col : str
        Name of the timestamp column

    Returns
    -------
    pd.DataFrame
        DataFrame with added temporal features
    """

    df = df.copy()

    df[timestamp_col] = pd.to_datetime(
        df[timestamp_col],
        dayfirst=True,
        errors="coerce"
)


    df["hour"] = df[timestamp_col].dt.hour
    df["day_of_week"] = df[timestamp_col].dt.dayofweek
    df["month"] = df[timestamp_col].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    return df


def select_features(
    df: pd.DataFrame,
    sensor_features: list,
    use_time_features: bool = True
) -> pd.DataFrame:
    """
    Select final feature set for modeling.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    sensor_features : list
        List of sensor feature column names
    use_time_features : bool, optional
        Whether to include temporal features, by default True

    Returns
    -------
    pd.DataFrame
        Feature matrix X
    """

    features = sensor_features.copy()

    if use_time_features:
        time_features = ["hour", "day_of_week", "month", "is_weekend"]
        features.extend(time_features)

    X = df[features]

    return X
