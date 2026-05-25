import os
from src.config import RAW_DATA_PATH, CLEAN_DATA_PATH
from src.data_loader import load_raw_data, save_data

def clean_data():
    os.makedirs("data/processed", exist_ok=True)

    df = load_raw_data(RAW_DATA_PATH)

    df = df.drop_duplicates()

    numeric_columns = [
        "normalized_losses",
        "wheel_base",
        "length",
        "width",
        "height",
        "curb_weight",
        "engine_size",
        "bore",
        "stroke",
        "compression_ratio",
        "horsepower",
        "peak_rpm",
        "city_mpg",
        "highway_mpg",
        "price"
    ]

    for col in numeric_columns:
        df[col] = df[col].astype(float)

    df = df.dropna(subset=["price"])

    for col in numeric_columns:
        df[col] = df[col].fillna(df[col].median())

    categorical_columns = df.select_dtypes(include="object").columns

    for col in categorical_columns:
        df[col] = df[col].fillna(df[col].mode()[0])
        df[col] = df[col].str.lower().str.strip()

    save_data(df, CLEAN_DATA_PATH)

    print(f"Cleaned data saved to {CLEAN_DATA_PATH}")

if __name__ == "__main__":
    clean_data()