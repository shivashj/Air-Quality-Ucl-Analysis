"""
Preprocessing Pipeline

This module combines feature engineering and label generation
to prepare final datasets for model training.
"""

import pandas as pd

from preprocessing.feature_engineering import (
    extract_time_features,
    select_features
)

from preprocessing.label_generation import generate_future_labels


def preprocess_data(
    df: pd.DataFrame,
    timestamp_col: str,
    sensor_features: list,
    target_columns: list,
    horizon: int = 1,
    use_time_features: bool = True
):
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input dataframe
    timestamp_col : str
        Name of timestamp column
    sensor_features : list
        Sensor feature column names
    target_columns : list
        Columns used for label generation
    thresholds : dict
        Thresholds for unhealthy air
    horizon : int, optional
        Prediction horizon in hours, by default 1
    use_time_features : bool, optional
        Whether to include temporal features, by default True

    Returns
    -------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Labels
    """

    # 1. Sort by time (VERY IMPORTANT)
    df = df.sort_values(by=timestamp_col).reset_index(drop=True)

    # 2. Extract time-based features
    df = extract_time_features(df, timestamp_col=timestamp_col)

    # 3. Select final feature set
    X = select_features(
        df=df,
        sensor_features=sensor_features,
        use_time_features=use_time_features
    )

    # 4. Generate future labels
    y = generate_future_labels(
        df=df,
        target_columns=target_columns,
        horizon=horizon
    )

    # 5. Remove rows with NaN (last rows due to shifting)
    valid_length = len(df) - horizon
    X = X.iloc[:valid_length]
    y = y.iloc[:valid_length]

    # 6. Final NA handling
    X = X.fillna(X.median())
    y = y.fillna(0).astype(int)

    return X, y
