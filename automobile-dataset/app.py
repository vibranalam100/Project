import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

from src.config import CLEAN_DATA_PATH, MODEL_PATH
from src.data_loader import load_clean_data
from src.feature_engineering import add_features


st.set_page_config(
    page_title="Automobile Analytics Platform",
    layout="wide"
)

st.title("🚗 Automobile Analytics + Testing Data GUI")

TESTING_DATA_DIR = Path("data/raw/testing")


@st.cache_data
def load_data():
    return load_clean_data(CLEAN_DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def find_csv_files(folder):
    if not folder.exists():
        return []
    return list(folder.rglob("*.csv"))


def load_testing_csv(file_path):
    return pd.read_csv(file_path)


tab1, tab2, tab3, tab4 = st.tabs([
    "Predict Price",
    "Automobile Dataset",
    "Visual Analysis",
    "Testing Datasets"
])


with tab1:
    st.header("Predict Car Price")

    try:
        model = load_model()

        col1, col2, col3 = st.columns(3)

        with col1:
            make = st.selectbox("Make", [
                "toyota", "nissan", "mazda", "mitsubishi", "honda",
                "volkswagen", "subaru", "peugot", "volvo", "dodge",
                "bmw", "mercedes-benz", "audi", "saab", "porsche"
            ])

            fuel_type = st.selectbox("Fuel Type", ["gas", "diesel"])
            aspiration = st.selectbox("Aspiration", ["std", "turbo"])
            num_doors = st.selectbox("Number of Doors", ["two", "four"])
            body_style = st.selectbox("Body Style", [
                "sedan", "hatchback", "wagon", "hardtop", "convertible"
            ])

        with col2:
            drive_wheels = st.selectbox("Drive Wheels", ["fwd", "rwd", "4wd"])
            engine_location = st.selectbox("Engine Location", ["front", "rear"])
            engine_type = st.selectbox("Engine Type", [
                "ohc", "ohcf", "ohcv", "dohc", "l", "rotor"
            ])
            num_cylinders = st.selectbox("Cylinders", [
                "two", "three", "four", "five", "six", "eight", "twelve"
            ])
            fuel_system = st.selectbox("Fuel System", [
                "mpfi", "2bbl", "idi", "1bbl", "spdi", "4bbl", "mfi", "spfi"
            ])

        with col3:
            symboling = st.slider("Insurance Risk Symboling", -3, 3, 0)
            normalized_losses = st.number_input("Normalized Losses", 50, 300, 120)
            wheel_base = st.number_input("Wheel Base", 80.0, 130.0, 100.0)
            length = st.number_input("Length", 140.0, 230.0, 175.0)
            width = st.number_input("Width", 55.0, 80.0, 65.0)
            height = st.number_input("Height", 45.0, 65.0, 54.0)

        st.subheader("Engine and Performance")

        col4, col5, col6 = st.columns(3)

        with col4:
            curb_weight = st.number_input("Curb Weight", 1400, 4500, 2500)
            engine_size = st.number_input("Engine Size", 60, 350, 130)

        with col5:
            bore = st.number_input("Bore", 2.0, 4.5, 3.2)
            stroke = st.number_input("Stroke", 2.0, 4.5, 3.3)
            compression_ratio = st.number_input("Compression Ratio", 7.0, 25.0, 9.0)

        with col6:
            horsepower = st.number_input("Horsepower", 40, 300, 120)
            peak_rpm = st.number_input("Peak RPM", 3000, 7000, 5000)
            city_mpg = st.number_input("City MPG", 10, 60, 24)
            highway_mpg = st.number_input("Highway MPG", 10, 70, 30)

        car_data = {
            "symboling": symboling,
            "normalized_losses": normalized_losses,
            "make": make,
            "fuel_type": fuel_type,
            "aspiration": aspiration,
            "num_doors": num_doors,
            "body_style": body_style,
            "drive_wheels": drive_wheels,
            "engine_location": engine_location,
            "wheel_base": wheel_base,
            "length": length,
            "width": width,
            "height": height,
            "curb_weight": curb_weight,
            "engine_type": engine_type,
            "num_cylinders": num_cylinders,
            "engine_size": engine_size,
            "fuel_system": fuel_system,
            "bore": bore,
            "stroke": stroke,
            "compression_ratio": compression_ratio,
            "horsepower": horsepower,
            "peak_rpm": peak_rpm,
            "city_mpg": city_mpg,
            "highway_mpg": highway_mpg
        }

        if st.button("Predict Price"):
            input_df = pd.DataFrame([car_data])
            input_df = add_features(input_df)

            prediction = model.predict(input_df)[0]

            st.success(f"Predicted Automobile Price: ${prediction:,.2f}")

    except FileNotFoundError:
        st.error("Model not found. Run `python main.py` first.")


with tab2:
    st.header("Clean Automobile Dataset")

    try:
        df = load_data()

        st.dataframe(df, use_container_width=True)

        st.subheader("Dataset Shape")
        st.write(f"Rows: {df.shape[0]}")
        st.write(f"Columns: {df.shape[1]}")

        st.subheader("Dataset Summary")
        st.dataframe(df.describe(), use_container_width=True)

    except FileNotFoundError:
        st.error("Clean dataset not found. Run `python main.py` first.")


with tab3:
    st.header("Visual Analysis")

    try:
        df = load_data()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Price Distribution")
            st.bar_chart(df["price"])

        with col2:
            st.subheader("Average Price by Body Style")
            avg_price = df.groupby("body_style")["price"].mean()
            st.bar_chart(avg_price)

        st.subheader("Average Price by Make")
        avg_make_price = df.groupby("make")["price"].mean().sort_values()
        st.bar_chart(avg_make_price)

        st.subheader("Horsepower vs Price")
        st.scatter_chart(
            df,
            x="horsepower",
            y="price"
        )

    except FileNotFoundError:
        st.error("Clean dataset not found. Run `python main.py` first.")


with tab4:
    st.header("Automotive Testing Datasets")

    csv_files = find_csv_files(TESTING_DATA_DIR)

    if not csv_files:
        st.warning("No testing CSV files found. Run `python main.py` first.")
    else:
        selected_file = st.selectbox(
            "Select testing dataset",
            csv_files,
            format_func=lambda x: str(x).replace("\\", "/")
        )

        df_test = load_testing_csv(selected_file)

        st.subheader("Dataset Preview")
        st.dataframe(df_test, use_container_width=True)

        st.subheader("Dataset Shape")
        st.write(f"Rows: {df_test.shape[0]}")
        st.write(f"Columns: {df_test.shape[1]}")

        st.subheader("Columns")
        st.write(list(df_test.columns))

        st.subheader("Missing Values")
        missing_df = df_test.isnull().sum().reset_index()
        missing_df.columns = ["column", "missing_values"]
        st.dataframe(missing_df, use_container_width=True)

        numeric_columns = df_test.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()

        if numeric_columns:
            st.subheader("Numerical Summary")
            st.dataframe(df_test[numeric_columns].describe(), use_container_width=True)

        if len(numeric_columns) >= 2:
            st.subheader("Scatter Plot")

            x_axis = st.selectbox("X Axis", numeric_columns, key="testing_x")
            y_axis = st.selectbox("Y Axis", numeric_columns, key="testing_y")

            st.scatter_chart(df_test, x=x_axis, y=y_axis)

        if len(numeric_columns) >= 1:
            st.subheader("Bar Chart")

            selected_numeric = st.selectbox(
                "Select numeric column",
                numeric_columns,
                key="testing_bar"
            )

            st.bar_chart(df_test[selected_numeric])