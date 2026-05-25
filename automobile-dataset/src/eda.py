import os
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import CLEAN_DATA_PATH
from src.data_loader import load_clean_data

def run_eda():
    os.makedirs("reports/figures", exist_ok=True)

    df = load_clean_data(CLEAN_DATA_PATH)

    print(df.head())
    print(df.info())
    print(df.describe())

    plt.figure(figsize=(10, 6))
    sns.histplot(df["price"], kde=True)
    plt.title("Automobile Price Distribution")
    plt.savefig("reports/figures/price_distribution.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="horsepower", y="price", hue="fuel_type")
    plt.title("Horsepower vs Price")
    plt.savefig("reports/figures/horsepower_vs_price.png")
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="body_style", y="price")
    plt.title("Price by Body Style")
    plt.savefig("reports/figures/price_by_body_style.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig("reports/figures/correlation_heatmap.png")
    plt.close()

if __name__ == "__main__":
    run_eda()