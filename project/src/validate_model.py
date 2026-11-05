import argparse
import json
import os

import mlflow
from mlflow import MlflowClient


DEFAULT_THRESHOLDS = {
    "MAE": 3500.0,
    "RMSE": 5000.0,
    "MAPE": 30.0,
}


def main():
    parser = argparse.ArgumentParser(
        description="Validate a trained model against performance thresholds."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--metrics-output", required=True)
    parser.add_argument("--max-mae", type=float, default=DEFAULT_THRESHOLDS["MAE"])
    parser.add_argument("--max-rmse", type=float, default=DEFAULT_THRESHOLDS["RMSE"])
    parser.add_argument("--max-mape", type=float, default=DEFAULT_THRESHOLDS["MAPE"])
    args = parser.parse_args()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or "sqlite:///mlflow.db"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(os.getenv("MLFLOW_REGISTRY_URI") or tracking_uri)

    run = MlflowClient().get_run(args.run_id)
    metrics = {
        "MAE": run.data.metrics.get("MAE"),
        "RMSE": run.data.metrics.get("RMSE"),
        "MAPE": run.data.metrics.get("MAPE"),
    }
    thresholds = {
        "MAE": args.max_mae,
        "RMSE": args.max_rmse,
        "MAPE": args.max_mape,
    }

    missing = [name for name, value in metrics.items() if value is None]
    if missing:
        raise ValueError(f"Missing required metrics in run {args.run_id}: {missing}")

    failures = {
        name: {"value": value, "maximum": thresholds[name]}
        for name, value in metrics.items()
        if value > thresholds[name]
    }
    result = {
        "run_id": args.run_id,
        "metrics": metrics,
        "thresholds": thresholds,
        "passed": not failures,
        "failures": failures,
    }
    with open(args.metrics_output, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit("Model validation failed.")


if __name__ == "__main__":
    main()
