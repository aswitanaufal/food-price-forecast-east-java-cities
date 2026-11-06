import os

import mlflow


DEFAULT_OWNER = "aswitanaufal"
DEFAULT_REPOSITORY = "food-price-forecast-east-java-cities"


def configure_mlflow():
    """Configure MLflow for DagsHub in CI or a local fallback."""
    token = os.getenv("DAGSHUB_TOKEN") or os.getenv("DAGSHUB_USER_TOKEN")
    if token:
        import dagshub

        owner = os.getenv("DAGSHUB_REPO_OWNER", DEFAULT_OWNER)
        os.environ["DAGSHUB_USER_TOKEN"] = token
        os.environ.setdefault("MLFLOW_TRACKING_USERNAME", owner)
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
        dagshub.auth.add_app_token(token)
        dagshub.init(
            repo_owner=owner,
            repo_name=os.getenv("DAGSHUB_REPO_NAME", DEFAULT_REPOSITORY),
            mlflow=True,
        )

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        tracking_uri = (
            mlflow.get_tracking_uri()
            if token
            else "sqlite:///mlflow.db"
        )
    registry_uri = os.getenv("MLFLOW_REGISTRY_URI") or tracking_uri
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri)
    return tracking_uri
