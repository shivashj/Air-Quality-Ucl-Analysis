# data_acquisition/synthetic_air_quality.py

import numpy as np
import pandas as pd
from pathlib import Path


def generate_synthetic_air_quality_data(
    start_date="2005-01-01",
    periods=1000,
    freq="H",
    save_path: Path | None = None,
    sensor_failure_rate: float = 0.03
) -> pd.DataFrame:
    """
    Generate synthetic air quality dataset similar to AirQualityUCI.

    Parameters
    ----------
    start_date : str
        Start date
    periods : int
        Number of samples
    freq : str
        Time frequency
    save_path : Path | None
        Save CSV if provided
    sensor_failure_rate : float
        Probability of sensor failure (-200 values)

    Returns
    -------
    df : pd.DataFrame
    """

    # 1- Time
    timestamps = pd.date_range(start=start_date, periods=periods, freq=freq)

    date = timestamps.date.astype(str)
    time = timestamps.time.astype(str)

    t = np.linspace(0, 10, periods)

    # 2- Environmental variables
    T = 15 + 10 * np.sin(t) + np.random.normal(0, 0.5, periods)
    RH = 60 - 15 * np.sin(t) + np.random.normal(0, 2, periods)
    AH = (RH / 100) * (T / 30) + np.random.normal(0, 0.01, periods)

    # 3- Pollutants (correlated)
    CO = 2 + 0.5 * np.sin(t) + np.random.normal(0, 0.2, periods)
    NMHC = 150 + 30 * np.cos(t) + np.random.normal(0, 10, periods)
    C6H6 = 5 + 2 * np.sin(t) + np.random.normal(0, 0.3, periods)
    NOx = 200 + 40 * np.sin(t) + np.random.normal(0, 15, periods)
    NO2 = 100 + 30 * np.cos(t) + np.random.normal(0, 10, periods)

    df = pd.DataFrame({
        "Date": date,
        "Time": time,
        "CO(GT)": CO,
        "NMHC(GT)": NMHC,
        "C6H6(GT)": C6H6,
        "NOx(GT)": NOx,
        "NO2(GT)": NO2,
        "T": T,
        "RH": RH,
        "AH": AH
    })

    # 4- Sensor failures (-200)
    for col in df.columns[2:]:
        mask = np.random.rand(periods) < sensor_failure_rate
        df.loc[mask, col] = -200

    # 5- Save
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, sep=";", decimal=",", index=False)

    return df
