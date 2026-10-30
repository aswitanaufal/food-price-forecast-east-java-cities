
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"

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

app = Flask(__name__)


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

# Load model
model = joblib.load(MODEL_PATH)


def normalize_weather_code(value):
    """Match the categorical representation used during training."""
    if value is None or pd.isna(value):
        return None

    try:
        return str(int(float(value)))
    except (ValueError, TypeError):
        return str(value)


def prepare_input(instances):
    """Validate JSON records and create the model input DataFrame."""
    if not isinstance(instances, list) or not instances:
        raise ValueError(
            "'instances' must be a non-empty list of JSON objects."
        )

    if not all(isinstance(row, dict) for row in instances):
        raise ValueError(
            "Every item in 'instances' must be a JSON object."
        )

    data = pd.DataFrame(instances)

    missing_columns = [
        col for col in FEATURE_COLUMNS
        if col not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required features: {missing_columns}"
        )

    # Keep only the columns that were used during training.
    data = data[FEATURE_COLUMNS].copy()

    for col in NUMERIC_FEATURES:
        data[col] = pd.to_numeric(
            data[col],
            errors="coerce",
        )

    for col in ["city", "commodity"]:
        data[col] = data[col].apply(
            lambda value: (
                str(value) if pd.notna(value) else None
            )
        )

    data["weather_code"] = data["weather_code"].apply(
        normalize_weather_code
    )

   
    if data[["city", "commodity"]].isna().any().any():
        raise ValueError(
            "'city' and 'commodity' cannot be empty."
        )

    numeric_array = data[NUMERIC_FEATURES].to_numpy(
        dtype=float
    )

    if np.isinf(numeric_array).any():
        raise ValueError(
            "Numeric features cannot contain infinity."
        )

    return data


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
    }), 200


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return jsonify({
            "error": "Request body must be a JSON object."
        }), 400

    instances = payload.get("instances")

    try:
        input_data = prepare_input(instances)
        predictions = model.predict(input_data)

        return jsonify({
            "predictions": [
                float(value) for value in predictions
            ],
            "count": len(predictions),
        }), 200

    except (ValueError, TypeError) as error:
        return jsonify({
            "error": str(error)
        }), 400


if __name__ == "__main__":
    # 0.0.0.0 to make flask service reachable outside the container
    app.run(host="0.0.0.0", port=8000)
