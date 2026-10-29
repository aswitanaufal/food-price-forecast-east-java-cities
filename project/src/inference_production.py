
from pathlib import Path

import mlflow
import pandas as pd


TRACKING_URI = "sqlite:///mlflow.db"

MODEL_URI = (
    "models:/EastJavaFoodPriceForecast/Production"
)

# MODEL_URI = "models:/EastJavaFoodPriceForecast@champion"

DATA_FILE = Path(
    "project/data/processed/food_price_features.csv"
)

NUMERIC_FEATURES = [
    "price_lag_1",
    "price_lag_7",
    "rolling_mean_7",
    "price_change",
    "day_of_week",
    "month",
    "temperature_mean",
    "temperature_max",
    "precipitation_sum",
    "humidity_mean",
    "wind_speed_mean",
]

CATEGORICAL_FEATURES = [
    "city",
    "commodity",
    "weather_code",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def normalize_weather_code(value):
    if pd.isna(value):
        return float("nan")

    try:
        return str(int(float(value)))
    except (ValueError, TypeError):
        return str(value)


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_registry_uri(TRACKING_URI)

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    data = pd.read_csv(DATA_FILE)

    missing_columns = [
        col for col in FEATURE_COLUMNS
        if col not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing feature columns: {missing_columns}"
        )

    if data.empty:
        raise ValueError("Feature dataset is empty.")

    # Choose the latest complete input row for each city and commodity.
    X = data[["date"] + FEATURE_COLUMNS].copy()
    X["date"] = pd.to_datetime(X["date"], errors="coerce")

    for col in NUMERIC_FEATURES:
        X[col] = pd.to_numeric(
            X[col],
            errors="coerce",
        ).astype("float64")

    X["weather_code"] = X["weather_code"].apply(
        normalize_weather_code
    )

    # A complete row makes the inference test easier to interpret.
    X = X.dropna(subset=["date"] + FEATURE_COLUMNS)

    if X.empty:
        raise ValueError(
            "No complete input row is available for inference."
        )

    X_sample = (
        X.sort_values("date")
        .groupby(["city", "commodity"], sort=True, observed=True)
        .tail(1)
        .sort_values(["city", "commodity"])
        .drop(columns="date")
        .reset_index(drop=True)
    )

    # Load the currently registered Production model.
    model = mlflow.pyfunc.load_model(MODEL_URI)

    prediction = model.predict(X_sample)

    print("\n=== PRODUCTION INFERENCE ===")
    print(f"Model URI: {MODEL_URI}")
    print(f"Input rows: {len(X_sample)}")
    for row, value in zip(
        X_sample.itertuples(index=False),
        prediction,
    ):
        print(
            f"City: {row.city} | "
            f"Commodity: {row.commodity} | "
            f"Prediction: {value:.2f}"
        )


if __name__ == "__main__":
    main()
