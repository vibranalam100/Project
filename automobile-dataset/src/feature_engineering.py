def add_features(df):
    df = df.copy()

    df["car_volume"] = df["length"] * df["width"] * df["height"]
    df["power_to_weight"] = df["horsepower"] / df["curb_weight"]
    df["engine_power_ratio"] = df["horsepower"] / df["engine_size"]
    df["avg_mpg"] = (df["city_mpg"] + df["highway_mpg"]) / 2
    df["mpg_difference"] = df["highway_mpg"] - df["city_mpg"]

    return df