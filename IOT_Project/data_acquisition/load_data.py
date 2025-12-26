
import pandas as pd
from config.setting import RAW_DATA_FILE


def load_raw_air_quality_data():
    """
    Loads the raw Air Quality UCI dataset from the raw data directory.

    Returns
    -------
    df : pandas.DataFrame
        Raw dataset loaded from CSV.
    """
    df = pd.read_csv(
        RAW_DATA_FILE,
        sep=";",
        decimal=","
    )
    return df

def inspect_raw_data(df):
    """
    Performs an initial inspection of the raw dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw dataset
    """
    print("\n🔹 First 5 rows of raw data:")
    print(df.head())

    print("\n🔹 Dataset info:")
    print(df.info())

    print("\n🔹 Statistical summary:")
    print(df.describe())


