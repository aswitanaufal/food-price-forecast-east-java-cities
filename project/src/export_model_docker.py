
from pathlib import Path
import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd


TRACKING_URI = "sqlite:///mlflow.db"

MODEL_URI = os.getenv(
    "MODEL_URI",
    "models:/EastJavaFoodPriceForecast@champion",
)

OUTPUT_DIR = Path("project/deployment")
OUTPUT_FILE = OUTPUT_DIR / "model.joblib"

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


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_registry_uri(TRACKING_URI)

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {DATA_FILE}"
        )

    # Load the selected Ridge Pipeline from MLflow
    print(f"[INFO] Loading model: {MODEL_URI}")

    model = mlflow.sklearn.load_model(MODEL_URI)

    # Export the entire pipeline, including preprocessing
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        model,
        OUTPUT_FILE,
        compress=3,
    )

    print(f"[SUCCESS] Model exported to: {OUTPUT_FILE}")

    # Verify that the exported file can be loaded
    loaded_model = joblib.load(OUTPUT_FILE)

    data = pd.read_csv(DATA_FILE)

    missing = [
        col for col in FEATURE_COLUMNS
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing feature columns: {missing}"
        )

    sample = data[FEATURE_COLUMNS].copy()

    for col in NUMERIC_FEATURES:
        sample[col] = pd.to_numeric(
            sample[col],
            errors="coerce",
        )

    sample["weather_code"] = sample["weather_code"].apply(
        lambda value: (
            str(int(float(value)))
            if pd.notna(value)
            else None
        )
    )

    sample = sample.dropna(subset=FEATURE_COLUMNS)

    if sample.empty:
        raise ValueError(
            "No complete feature row is available for testing."
        )

    # Test a single valid input row
    prediction = loaded_model.predict(sample.tail(1))

    print("[SUCCESS] Exported model loaded successfully.")
    print(f"[INFO] Test prediction: {float(prediction[0]):.2f}")


if __name__ == "__main__":
    main()
