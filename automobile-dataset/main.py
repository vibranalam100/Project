from src.download_data import download_data
from src.data_cleaning import clean_data
from src.download_testing_data import download_all_testing_data
from src.eda import run_eda
from src.model_training import train_model

def main():
    download_data()
    clean_data()
    run_eda()
    train_model()
    download_all_testing_data()

if __name__ == "__main__":
    main()