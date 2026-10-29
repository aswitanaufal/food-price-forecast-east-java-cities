
from pathlib import Path
import argparse
import re

import mlflow
from mlflow import MlflowClient


TRACKING_URI = "sqlite:///mlflow.db"
MODEL_NAME = "EastJavaFoodPriceForecast"

DATASET_DVC_POINTER = Path(
    "project/data/processed/food_price_features.csv.dvc"
)


def get_dataset_dvc_hash():
    if not DATASET_DVC_POINTER.exists():
        print(
            "[WARNING] DVC pointer not found. "
            "Dataset lineage tag will be omitted."
        )
        return None

    content = DATASET_DVC_POINTER.read_text(
        encoding="utf-8"
    )

    match = re.search(
        r"(?m)^\s*md5:\s*([^\s]+)",
        content,
    )

    return match.group(1) if match else None


def main():
    parser = argparse.ArgumentParser(
        description="Register a model logged by MLflow."
    )

    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--description",
        default="Ridge Regression food price forecasting model",
    )

    args = parser.parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_registry_uri(TRACKING_URI)

    client = MlflowClient()
    run = client.get_run(args.run_id)

    if run.info.status != "FINISHED":
        raise ValueError(
            f"Run status is {run.info.status}; "
            "the run must be FINISHED."
        )

    # MLflow 3: find models stored independently from run artifacts.
    if not hasattr(mlflow, "search_logged_models"):
        raise RuntimeError(
            "This script expects an MLflow version that supports "
            "search_logged_models()."
        )

    logged_models = mlflow.search_logged_models(
        experiment_ids=[run.info.experiment_id],
        output_format="list",
    )

    matches = [
        model for model in logged_models
        if model.source_run_id == args.run_id
    ]

    if not matches:
        raise ValueError(
            "No Logged Model was found for this Run ID. "
            "Check the MLflow UI Logged Models page, or rerun "
            "training using mlflow.sklearn.log_model(name='model_dir')."
        )

    # Prefer the model named model_dir, matching the training script.
    named_matches = [
        model for model in matches
        if model.name == "model_dir"
    ]

    if len(named_matches) == 1:
        logged_model = named_matches[0]
    elif len(matches) == 1:
        logged_model = matches[0]
    else:
        details = [
            f"{model.name}: {model.model_id}"
            for model in matches
        ]
        raise ValueError(
            "Multiple logged models were found. "
            "Select the intended model explicitly: "
            + ", ".join(details)
        )

    print("[INFO] Selected Logged Model")
    print("Name:", logged_model.name)
    print("Model ID:", logged_model.model_id)
    print("Status:", logged_model.status)

    # Register the logged model by model ID.
    model_uri = f"models:/{logged_model.model_id}"

    registered = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME,
        await_registration_for=300,
    )

    version = str(registered.version)

    client.set_model_version_tag(
        MODEL_NAME,
        version,
        "source_run_id",
        args.run_id,
    )

    client.set_model_version_tag(
        MODEL_NAME,
        version,
        "source_model_id",
        logged_model.model_id,
    )

    client.set_model_version_tag(
        MODEL_NAME,
        version,
        "description",
        args.description,
    )

    dataset_hash = get_dataset_dvc_hash()

    if dataset_hash:
        client.set_model_version_tag(
            MODEL_NAME,
            version,
            "dataset_dvc_md5",
            dataset_hash,
        )

    for metric_name, metric_value in run.data.metrics.items():
        client.set_model_version_tag(
            MODEL_NAME,
            version,
            f"metric_{metric_name.lower()}",
            str(metric_value),
        )

    print("\n=== MODEL REGISTERED ===")
    print("Model name:", MODEL_NAME)
    print("Version:", version)
    print("Run ID:", args.run_id)
    print("Logged Model ID:", logged_model.model_id)
    print("Registered URI:", f"models:/{MODEL_NAME}/{version}")


if __name__ == "__main__":
    main()
