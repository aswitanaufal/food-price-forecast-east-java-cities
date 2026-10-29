
from pathlib import Path
import shutil
import subprocess

import mlflow
from mlflow import MlflowClient
import yaml


TRACKING_URI = "sqlite:///mlflow.db"
MODEL_NAME = "EastJavaFoodPriceForecast"

DATASET_DVC_POINTER = Path(
    "project/data/processed/food_price_features.csv.dvc"
)

RELEASES_DIR = Path("project/models/releases")


def read_dataset_dvc_metadata():
    if not DATASET_DVC_POINTER.exists():
        raise FileNotFoundError(
            f"DVC pointer missing: {DATASET_DVC_POINTER}"
        )

    pointer = yaml.safe_load(
        DATASET_DVC_POINTER.read_text(encoding="utf-8")
    )

    outputs = pointer.get("outs", [])

    if not outputs:
        raise ValueError(
            "The DVC pointer does not contain output metadata."
        )

    dvc_hash = outputs[0].get("md5")

    if not dvc_hash:
        raise ValueError(
            "The dataset hash could not be read from the DVC pointer."
        )

    return dvc_hash


def get_git_commit():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )

    return result.stdout.strip()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Export a registered model release for DVC."
    )

    parser.add_argument("--run-id", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument(
        "--stage",
        required=True,
        choices=["Staging", "Production"],
    )

    args = parser.parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_registry_uri(TRACKING_URI)

    client = MlflowClient()

    registered_version = client.get_model_version(
        MODEL_NAME,
        args.version,
    )

    # Verify that the selected run is the version source.
    if registered_version.run_id != args.run_id:
        raise ValueError(
            "The supplied Run ID does not match the "
            "registered model version."
        )

    current_stage = registered_version.current_stage

    if current_stage.lower() != args.stage.lower():
        raise ValueError(
            f"Model version {args.version} is currently "
            f"{current_stage}, not {args.stage}."
        )

    run = client.get_run(args.run_id)
    dataset_hash = read_dataset_dvc_metadata()

    release_dir = RELEASES_DIR / f"v{args.version}"
    model_dir = release_dir / "model"

    if release_dir.exists():
        raise FileExistsError(
            f"Release already exists: {release_dir}. "
            "Use a new release directory or inspect the existing one."
        )

    release_dir.mkdir(parents=True, exist_ok=False)

    try:
        # Download the model artifact from the exact training run.
        artifact_uri = (
            f"runs:/{args.run_id}/model_dir"
        )

        downloaded_path = mlflow.artifacts.download_artifacts(
            artifact_uri=artifact_uri
        )

        source_dir = Path(downloaded_path)

        if not source_dir.is_dir():
            raise ValueError(
                f"Model artifact directory was not found: {source_dir}"
            )

        shutil.copytree(source_dir, model_dir)

        manifest = {
            "model_name": MODEL_NAME,
            "registry_version": str(args.version),
            "stage_at_export": current_stage,
            "alias_at_export": (
                "champion"
                if current_stage == "Production"
                else "candidate"
            ),
            "run_id": args.run_id,
            "model_uri": (
                f"models:/{MODEL_NAME}/{args.version}"
            ),
            "metrics": dict(run.data.metrics),
            "parameters": dict(run.data.params),
            "dataset": {
                "path": (
                    "project/data/processed/"
                    "food_price_features.csv"
                ),
                "dvc_pointer": (
                    DATASET_DVC_POINTER.as_posix()
                ),
                "dvc_md5": dataset_hash,
            },
            "git_commit": get_git_commit(),
        }

        manifest_path = release_dir / "model_release.yaml"

        with open(
            manifest_path,
            "w",
            encoding="utf-8",
        ) as file:
            yaml.safe_dump(
                manifest,
                file,
                sort_keys=False,
                allow_unicode=True,
            )

        print("\n=== MODEL RELEASE EXPORTED ===")
        print(f"Model name : {MODEL_NAME}")
        print(f"Version    : {args.version}")
        print(f"Stage      : {current_stage}")
        print(f"Run ID     : {args.run_id}")
        print(f"Release dir: {release_dir}")
        print(f"Manifest   : {manifest_path}")
        print(f"Dataset MD5: {dataset_hash}")

    except Exception:
        shutil.rmtree(release_dir, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
