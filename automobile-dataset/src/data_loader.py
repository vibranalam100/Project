import pandas as pd
from src.config import COLUMNS

def load_raw_data(path):
    return pd.read_csv(path, names=COLUMNS, na_values="?")

def load_clean_data(path):
    return pd.read_csv(path)

def save_data(df, path):
    df.to_csv(path, index=False)