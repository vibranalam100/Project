import os
import subprocess
from pathlib import Path

DATA_DIR = Path("data/raw/testing")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_kaggle_dataset(dataset_name: str, output_folder: str):
    """
    Requires Kaggle API:
    1. pip install kaggle
    2. Put kaggle.json in:
       Windows: C:/Users/YOUR_NAME/.kaggle/kaggle.json
       Linux/Mac: ~/.kaggle/kaggle.json
    """

    output_path = DATA_DIR / output_folder
    output_path.mkdir(parents=True, exist_ok=True)

    command = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        dataset_name,
        "-p",
        str(output_path),
        "--unzip"
    ]

    subprocess.run(command, check=True)
    print(f"Downloaded Kaggle dataset: {dataset_name}")


def download_github_repo(repo_url: str, output_folder: str):
    output_path = DATA_DIR / output_folder

    if output_path.exists():
        print(f"{output_folder} already exists.")
        return

    command = [
        "git",
        "clone",
        repo_url,
        str(output_path)
    ]

    subprocess.run(command, check=True)
    print(f"Cloned GitHub repo: {repo_url}")


def download_all_testing_data():
    # Brake testing dataset from Kaggle
    download_kaggle_dataset(
        dataset_name="rituparnaghosh18/braking-distance-data",
        output_folder="brake_testing"
    )

    # Gearbox fault diagnosis dataset from Kaggle
    download_kaggle_dataset(
        dataset_name="brjapon/gearbox-fault-diagnosis",
        output_folder="gearbox_testing"
    )

    # Engine fault detection dataset from Kaggle
    download_kaggle_dataset(
        dataset_name="ziya07/engine-fault-detection-data",
        output_folder="engine_testing"
    )

    # Public gearbox benchmark dataset from GitHub
    download_github_repo(
        repo_url="https://github.com/liuzy0708/MCC5-THU-Gearbox-Benchmark-Datasets.git",
        output_folder="gearbox_benchmark_github"
    )

    # Public engine fault dataset from GitHub
    download_github_repo(
        repo_url="https://github.com/Leo-Thomas/EngineFaultDB.git",
        output_folder="engine_fault_db_github"
    )


if __name__ == "__main__":
    download_all_testing_data()