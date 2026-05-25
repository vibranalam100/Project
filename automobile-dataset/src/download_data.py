import os
import requests
from src.config import RAW_DATA_URL, RAW_DATA_PATH

def download_data():
    os.makedirs("data/raw", exist_ok=True)

    response = requests.get(RAW_DATA_URL, timeout=20)
    response.raise_for_status()

    with open(RAW_DATA_PATH, "wb") as file:
        file.write(response.content)

    print(f"Downloaded data to {RAW_DATA_PATH}")

if __name__ == "__main__":
    download_data()