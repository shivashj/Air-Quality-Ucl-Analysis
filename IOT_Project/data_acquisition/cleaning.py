# preprocessing/cleaning.py

import pandas as pd
#from config.setting import (
#    PROCESSED_DATA_DIR,
#    RAW_DATA_FILE
#)


def handle_missing_values(df):
    df = df.replace(-200, pd.NA)

    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    return df



