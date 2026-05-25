import os
import joblib
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import CLEAN_DATA_PATH, MODEL_PATH, RANDOM_STATE, TEST_SIZE
from src.data_loader import load_clean_data
from src.feature_engineering import add_features

def train_model():
    os.makedirs("models", exist_ok=True)

    df = load_clean_data(CLEAN_DATA_PATH)
    df = add_features(df)

    target = "price"

    features = [
        "symboling",
        "normalized_losses",
        "make",
        "fuel_type",
        "aspiration",
        "num_doors",
        "body_style",
        "drive_wheels",
        "engine_location",
        "wheel_base",
        "length",
        "width",
        "height",
        "curb_weight",
        "engine_type",
        "num_cylinders",
        "engine_size",
        "fuel_system",
        "bore",
        "stroke",
        "compression_ratio",
        "horsepower",
        "peak_rpm",
        "city_mpg",
        "highway_mpg",
        "car_volume",
        "power_to_weight",
        "engine_power_ratio",
        "avg_mpg",
        "mpg_difference"
    ]

    X = df[features]
    y = df[target]

    categorical_features = X.select_dtypes(include="object").columns.tolist()
    numeric_features = X.select_dtypes(exclude="object").columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", StandardScaler(), numeric_features)
        ]
    )

    models = {
        "Ridge Regression": Ridge(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            random_state=RANDOM_STATE
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE
        )
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    best_model = None
    best_score = -999

    for name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model)
            ]
        )

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        r2 = r2_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        cv_scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=5,
            scoring="r2"
        )

        print("\n==============================")
        print(name)
        print("==============================")
        print("R2 Score:", round(r2, 4))
        print("MAE:", round(mae, 2))
        print("RMSE:", round(rmse, 2))
        print("CV R2 Mean:", round(cv_scores.mean(), 4))

        if r2 > best_score:
            best_score = r2
            best_model = pipeline

    joblib.dump(best_model, MODEL_PATH)
    print(f"\nBest model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_model()