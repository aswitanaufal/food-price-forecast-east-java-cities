
import argparse

import mlflow
from mlflow import MlflowClient


TRACKING_URI = "sqlite:///mlflow.db"
MODEL_NAME = "EastJavaFoodPriceForecast"


def main():
    parser = argparse.ArgumentParser(
        description="Promote an MLflow model version."
    )

    parser.add_argument(
        "--version",
        required=True,
        help="Registered model version, such as 1 or 2.",
    )

    parser.add_argument(
        "--stage",
        required=True,
        choices=["Staging", "Production"],
    )

    args = parser.parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_registry_uri(TRACKING_URI)

    client = MlflowClient()

    version = client.get_model_version(
        MODEL_NAME,
        args.version,
    )

    print(
        f"[INFO] Current stage: {version.current_stage}"
    )

    if not hasattr(
        client,
        "transition_model_version_stage",
    ):
        raise RuntimeError(
            "Stage transitions are unavailable in this MLflow "
            "version. Use the candidate/champion aliases and "
            "document the limitation."
        )

    # When promoting to Production, archive other Production versions
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=str(args.version),
        stage=args.stage,
        archive_existing_versions=(
            args.stage == "Production"
        ),
    )

    # Also update a modern model alias
    alias = (
        "candidate"
        if args.stage == "Staging"
        else "champion"
    )

    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias=alias,
        version=str(args.version),
    )

    updated = client.get_model_version(
        MODEL_NAME,
        args.version,
    )

    print("\n=== MODEL PROMOTION COMPLETE ===")
    print(f"Model   : {MODEL_NAME}")
    print(f"Version : {args.version}")
    print(f"Stage   : {updated.current_stage}")
    print(f"Alias   : {alias}")


if __name__ == "__main__":
    main()
