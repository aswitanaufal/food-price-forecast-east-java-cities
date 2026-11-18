from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"
FEATURE_DATA_PATH = BASE_DIR / "food_price_features.csv"
LOCAL_FEATURE_DATA_PATH = (
    BASE_DIR.parent / "data" / "processed" / "food_price_features.csv"
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
CATEGORICAL_FEATURES = ["city", "commodity", "weather_code"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)
app = FastAPI(title="East Java Food Price Forecast API", version="1.0.0")


def normalize_weather_code(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    try:
        return str(int(float(value)))
    except (ValueError, TypeError):
        return str(value)


def prepare_input(instances: Any) -> pd.DataFrame:
    if not isinstance(instances, list) or not instances:
        raise ValueError("'instances' must be a non-empty list of JSON objects.")
    if not all(isinstance(row, dict) for row in instances):
        raise ValueError("Every item in 'instances' must be a JSON object.")

    data = pd.DataFrame(instances)
    missing_columns = [column for column in FEATURE_COLUMNS if column not in data]
    if missing_columns:
        raise ValueError(f"Missing required features: {missing_columns}")

    data = data[FEATURE_COLUMNS].copy()
    for column in NUMERIC_FEATURES:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    for column in ["city", "commodity"]:
        data[column] = data[column].apply(
            lambda value: str(value) if pd.notna(value) else None
        )
    data["weather_code"] = data["weather_code"].apply(normalize_weather_code)

    if data[["city", "commodity"]].isna().any().any():
        raise ValueError("'city' and 'commodity' cannot be empty.")
    if data[NUMERIC_FEATURES].isna().any().any():
        raise ValueError("Numeric features must be valid numbers.")
    if np.isinf(data[NUMERIC_FEATURES].to_numpy(dtype=float)).any():
        raise ValueError("Numeric features cannot contain infinity.")
    return data


def predict_instances(instances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    input_data = prepare_input(instances)
    values = model.predict(input_data)
    return [
        {
            "city": row.city,
            "commodity": row.commodity,
            "prediction": float(value),
        }
        for row, value in zip(input_data.itertuples(index=False), values)
    ]


def latest_instances() -> list[dict[str, Any]]:
    data_path = (
        FEATURE_DATA_PATH
        if FEATURE_DATA_PATH.exists()
        else LOCAL_FEATURE_DATA_PATH
    )
    if not data_path.exists():
        raise FileNotFoundError(f"Feature dataset not found: {FEATURE_DATA_PATH}")

    data = pd.read_csv(data_path)
    required_columns = ["date", *FEATURE_COLUMNS]
    missing_columns = [column for column in required_columns if column not in data]
    if missing_columns:
        raise ValueError(f"Missing dataset columns: {missing_columns}")

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date"] + FEATURE_COLUMNS)
    if data.empty:
        raise ValueError("No complete input row is available for inference.")

    latest = (
        data.sort_values("date")
        .groupby(["city", "commodity"], sort=True, observed=True)
        .tail(1)
        .sort_values(["city", "commodity"])
    )
    return latest[FEATURE_COLUMNS].to_dict(orient="records")


@app.get("/")
def index() -> dict[str, Any]:
    return {
        "service": "food-price-api",
        "status": "running",
        "endpoints": {
            "health": "GET /health",
            "predict": "POST /predict",
            "predict_latest": "GET /predict/latest",
        },
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        results = predict_instances(payload.get("instances"))
    except (ValueError, TypeError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"predictions": results, "count": len(results)}


@app.get("/predict/latest")
def predict_latest() -> dict[str, Any]:
    try:
        results = predict_instances(latest_instances())
    except (FileNotFoundError, ValueError, TypeError) as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    return {"predictions": results, "count": len(results)}
