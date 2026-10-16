from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
import argparse

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_FILE = Path("project/data/processed/food_price_features.csv")

# simpan di mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Food Price Forecasting - East Java")

yaml.add_representer(
    np.dtype,
    lambda dumper, data: dumper.represent_str(str(data)),
    Dumper=yaml.SafeDumper,
)
yaml.add_representer(
    type,
    lambda dumper, data: dumper.represent_str(str(data)),
    Dumper=yaml.SafeDumper,
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

def parse_args():
    parser = argparse.ArgumentParser(description="Training Regression Model")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Regularization strength for Regression (default: 0.1)",
    )
    return parser.parse_args()

# hitung metrik evaluasi: MAE, RMSE, MAPE
def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # handle 0 division
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)

    nonzero = y_true_array != 0

    if nonzero.any():
        mape = np.mean(
            np.abs(
                (y_true_array[nonzero] - y_pred_array[nonzero])
                / y_true_array[nonzero]
            )
        ) * 100
    else:
        mape = float("nan")

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
    }


# baca dataset split data train dan test
def load_and_split_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError("Dataset not found. Please run feature_engineering.py first!")

    data = pd.read_csv(DATA_FILE)
    data["date"] = pd.to_datetime(data["date"])

    required = [
        "date",
        "city",
        "commodity",
        "price",
        "price_next_day",
    ] + NUMERIC_FEATURES

    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f"Row not found: {missing}")

    if "weather_code" not in data.columns:
        raise ValueError("Column 'weather_code' not found in dataset!")

    data = data.sort_values("date").reset_index(drop=True)

    dates = sorted(data["date"].dropna().unique())
    if len(dates) < 5:
        raise ValueError("Dataset too small for time-based split, need to collect more data!")

    split_index = int(len(dates) * 0.8)
    split_index = min(max(split_index, 1), len(dates) - 1)
    split_date = dates[split_index]

    train_data = data[data["date"] < split_date].copy()
    test_data = data[data["date"] >= split_date].copy()

    if train_data.empty or test_data.empty:
        raise ValueError("Train or test dataset is empty after split. Check the date range of the dataset!")

    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    X_train = train_data[feature_columns].copy()
    X_test = test_data[feature_columns].copy()
    y_train = train_data["price_next_day"]
    y_test = test_data["price_next_day"]

    # pastikan numeric features bertipe float64 agar aman jika ada missing values
    for col in NUMERIC_FEATURES:
        X_train[col] = X_train[col].astype("float64")
        X_test[col] = X_test[col].astype("float64")

    # weather_code diasumsikan sebagai kategori
    X_train["weather_code"] = X_train["weather_code"].astype("string")
    X_test["weather_code"] = X_test["weather_code"].astype("string")

    print(f"[INFO] Training rows: {len(train_data)}")
    print(f"[INFO] Testing rows: {len(test_data)}")
    print(f"[INFO] Test starts at: {split_date}")

    return X_train, X_test, y_train, y_test, train_data, test_data


# regresi dengan pipeline preprocessing
def create_model(alpha):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("regressor", Ridge(alpha=alpha)),
    ])

    return model


def log_metrics(metrics):
    for name, value in metrics.items():
        if np.isfinite(value):
            mlflow.log_metric(name, value)


def main():

    args = parse_args()
    alpha = args.alpha

    X_train, X_test, y_train, y_test, train_data, test_data = load_and_split_data()

    # baseline (hanya run 1x)
    with mlflow.start_run(
        run_name="Food_Price_Forecast",
        tags={
            "task": "forecasting",
            "model_type": "baseline",
            "framework": "numpy",
        },
    ):
        mlflow.log_param("model", "Food Price orecast")
        mlflow.log_param("strategy", "last_observed_price")
        mlflow.log_param("split", "time_based_80_20")

        baseline_predictions = test_data["price"].to_numpy()
        baseline_metrics = calculate_metrics(y_test, baseline_predictions)

        log_metrics(baseline_metrics)
        print("\nForecast:", baseline_metrics)

    # run regression
    with mlflow.start_run(
        run_name=f"Regression_alpha_{alpha}",
        tags={
            "task": "forecasting",
            "model_type": "regression",
            "algorithm": "ridge",
            "framework": "scikit-learn",
        },
    ):
        model = create_model(alpha)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        metrics = calculate_metrics(y_test, predictions)

        # simpan parameter
        mlflow.log_param("model", "Regression")
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("split", "time_based_80_20")
        mlflow.log_param("training_rows", len(X_train))
        mlflow.log_param("testing_rows", len(X_test))

        # simpan metrik
        log_metrics(metrics)

        # simpan model artifact
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model_dir",
            input_example=X_train.head(3),
            serialization_format="cloudpickle",
        )

        print(f"\n[RESULT] Regression alpha={alpha}:")
        print(metrics)

    print("\n[SUCCESS] Experiment completed, check MLflow UI for details")


if __name__ == "__main__":
    main()
