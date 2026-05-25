RAW_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"

RAW_DATA_PATH = "data/raw/imports_85.csv"
CLEAN_DATA_PATH = "data/processed/automobile_clean.csv"
MODEL_PATH = "models/automobile_price_model.pkl"

RANDOM_STATE = 42
TEST_SIZE = 0.2

COLUMNS = [
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
    "price"
]