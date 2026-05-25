import joblib
import pandas as pd

from src.config import MODEL_PATH
from src.feature_engineering import add_features

def predict_price(car_data):
    model = joblib.load(MODEL_PATH)

    df = pd.DataFrame([car_data])
    df = add_features(df)

    prediction = model.predict(df)[0]
    return round(prediction, 2)

if __name__ == "__main__":
    sample_car = {
        "symboling": 1,
        "normalized_losses": 120,
        "make": "bmw",
        "fuel_type": "gas",
        "aspiration": "std",
        "num_doors": "four",
        "body_style": "sedan",
        "drive_wheels": "rwd",
        "engine_location": "front",
        "wheel_base": 101.2,
        "length": 176.8,
        "width": 64.8,
        "height": 54.3,
        "curb_weight": 2500,
        "engine_type": "ohc",
        "num_cylinders": "six",
        "engine_size": 164,
        "fuel_system": "mpfi",
        "bore": 3.31,
        "stroke": 3.19,
        "compression_ratio": 9.0,
        "horsepower": 121,
        "peak_rpm": 4250,
        "city_mpg": 21,
        "highway_mpg": 28
    }

    price = predict_price(sample_car)
    print(f"Predicted price: ${price}")