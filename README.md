# food-price-forecast-east-java-cities

## Project Overview

This project is an MLOps-based system for forecasting the prices of popular food commodities in cities across East Java, Indonesia. The project uses food price data from **PIHPS (Bank Indonesia)** and weather data from **Open-Meteo** as the primary data sources.

The main objective is to build a reproducible machine learning pipeline that can continuously ingest new data, process and evaluate the data, train forecasting models, and support continuous training as new data becomes available or the underlying data distribution changes.

The initial machine learning task is **regression-based forecasting**, where the system aims to predict the next-day price of selected food commodities using historical price information and weather-related features.

## Project Objectives

The main objectives of this project are:

- Collect periodically updated food price data from PIHPS.
- Collect weather data from Open-Meteo.
- Process and combine food price and weather data.
- Develop a model for next-day food price forecasting.
- Evaluate model performance using MAE, RMSE, and MAPE.
- Build a reproducible MLOps pipeline.
- Implement a continuous training strategy so the model can be updated when new data becomes available.

## Project Structure

The repository uses the following directory structure:

food-price-forecast-east-java-cities/
- .devcontainer/
  - devcontainer.json
- .github/
- project/
  - configs/
  - data/
    - raw
    - processed
  - docs/
  - models/
  - notebooks/
  - src/
  - tests/

## Running the Project with GitHub Codespaces

1. Open the Repository on GitHub
2. Create or Open a Codespace
    From the repository page, select: Code → Codespaces → Create codespace on main
    If a Codespace has already been created, open the existing Codespace instead of creating a new one.
    The .devcontainer/devcontainer.json file is used to configure the development environment automatically.
3. Verify the Python Environment
    After the Codespace has finished loading, open the terminal and run: python --version
    You can also verify Git: git --version
4. Install project dependencies, for now you can do: pip install pandas scikit-learn jupyter
5. Run the Project
    The project components will be executed from the project/ directory as the implementation develops.
6. Git Workflow
    This project follows a simplified GitHub Flow workflow.
    The main branch is used as the stable branch. New features or experiments should be developed in separate branches.

## Running Auto Ingest and Preprocess Data
The workflow is configured via .github/workflows/data_ingestion.yml and performs the following:
- Automated Schedule: Runs automatically every weekday (Monday to Friday) at 13:30 WIB (06:30 UTC).

Manual Trigger: Can be manually executed at any time from the GitHub interface:
1. Go to the Actions tab in this repository.
2. Select Automatic Ingestion and Preprocessing Data from the left sidebar.
3. Click Run workflow, select the target branch, and confirm.

Output: For now will be generating updates automatically committed and pushed to the ingestion-experiment branch.

Otherwise, this process can be run manually on local environment:
1. Install the requirement and dependencies
2. Run ingestion file by running: python project/src/ingest_data.py
3. Wait until ingesting process success and data will be saved in project/data/raw/
4. Run preprocessing file by running: python project/src/preprocess.py
5. Preprocessed data will be saved in project/data/processed/

## MLOps Automation (Code as a Trigger)

The workflow [.github/workflows/mlops-automation.yaml](.github/workflows/mlops-automation.yaml)
runs pytest on every pull request and on pushes to `main`. A push to `main` continues
automatically through DVC data pull, model training, metric validation, and MLflow
registration as `Staging`. A pull request intentionally stops after testing so that
repository secrets are not exposed to untrusted code.

### Public MLflow setup

This project uses the public DagsHub repository as both its DVC remote and MLflow
tracking/registry backend. Add `DAGSHUB_TOKEN` under **Settings > Secrets and
variables > Actions**. The workflows authenticate DVC with the DagsHub username
and token, while `project/src/mlflow_config.py` calls `dagshub.init(...)` before
training, validation, and registration.

For CI authentication, the token is also registered with
`dagshub.auth.add_app_token()` and exported as `DAGSHUB_USER_TOKEN`. This avoids
the interactive OAuth flow, which cannot complete on a headless GitHub Actions
runner.

The local `sqlite:///mlflow.db` file remains only as a local-development fallback
when `DAGSHUB_TOKEN` is not set. It is not used by the GitHub Actions jobs.

The secret name is case-sensitive and must be exactly `DAGSHUB_TOKEN`. Create it
under **Repository secrets**, not only as a local Codespaces secret. If it was
created under an Actions environment, add that environment name to the
`train-evaluate-register` job. A successful DVC datasource/storage entry alone
does not upload files; the ingestion workflow must run `dvc push -r origin`.

The workflow does not place credentials in source code. The values `MAX_MAE`,
`MAX_RMSE`, and `MAX_MAPE` are currently set in the workflow to `3500`, `5000`,
and `30`; replace them with the exact thresholds from LK-01 before submission.

### Demonstrating the chain

1. Commit and push a small change to `main`, such as changing `--alpha 0.1` in
   the workflow or a preprocessing parameter.
2. Open **Actions > MLOps Automation** and show the successful `Automated tests`
   and `Train, evaluate, and register` jobs.
3. Download the `mlops-training-evidence` artifact. It contains the run ID,
   metrics, thresholds, and validation result.
4. Open the public MLflow UI, locate experiment `Food Price Forecasting - East Java`,
   and show the registered `EastJavaFoodPriceForecast` version with stage `Staging`.

The Ridge hyperparameter is named `alpha`. It defaults
to `0.1` for push and pull-request-triggered runs. To test another value without
editing the workflow, open **Actions > MLOps Automation > Run workflow**, enter
the desired `alpha`, and start the run. The selected value is logged in MLflow
as the `alpha` parameter.

## Data Versioning
For now, this project uses DVC (Data Version Control) to track changes in raw datasets generated from PIHPS and Open-Meteo data ingestion process. DVC allows each dataset update to be recorded as a new version while keeping the actual dataset separate from Git source code. This method used some workflow:
1. Data Ingestion: new raw data will automatically collected from PIHPS and Open-Meteo based on Auto Ingest above
2. After new data is collected, raw data will tracked using DVC:
   dvc add project/data/raw/pihps
   dvc add project/data/raw/open_meteo
  This generates .dvc pointer files that store metadata and hashes for the tracked datasets.
3. DVC can be used to check whether the dataset has changed:
   dvc status
   dvc diff
4. The updated dataset is pushed to the configured local DVC storage:
   dvc push
5. Commit the dataset version metadata to GitHub
6. To restore dataset version use:
   dvc pull

## MLflow
This project uses DagsHub as the shared MLflow tracking and model registry backend.
The GitHub Actions workflows authenticate with `DAGSHUB_TOKEN`, and
`project/src/mlflow_config.py` calls `dagshub.init` for the configured DagsHub
repository. Local execution without that token falls back to
`sqlite:///mlflow.db`.

### Workflow pipeline:
To run a complete end-to-end experiment, execute the pipeline components in sequence:
1. Ingest raw data
2. Preprocess raw data
3. Generate features/feature engineering
4. Train and track model

Every execution of train.py automatically logs the following information under the "Food Price Forecasting - East Java" experiment:
1. Parameters: model, alpha, split (80/20 time-based), training_rows, testing_rows.
2. Metrics: MAE, RMSE, MAPE.
3. Tags:
    - task: forecasting
    - model_type: regression
    - algorithm: ridge
    - framework: scikit-learn
4. Artifacts: Full Scikit-learn pipeline (preprocessing + Ridge regressor) saved under model_dir/ using cloudpickle serialization along with model input examples.

To view the experiments:
1. Open the DagsHub repository page and select the MLflow experiments/runs view.
2. Select `Food Price Forecasting - East Java`.
3. Analyze runs, metrics, parameters, and model artifacts.
    - Compare runs side-by-side by selecting individual runs and clicking Compare.
    - Customize displayed table columns to sort models by metrics (MAE, RMSE, MAPE).
    - Inspect saved artifacts, input schemas, and environment dependencies directly inside the UI under model_dir.


## Model Registry, Versioning, and Inference

### Registered Model

This project uses MLflow Model Registry to manage the lifecycle of Ridge Regression models for East Java food price forecasting.

The registered model name is `EastJavaFoodPriceForecast`. Each registered version is connected to its source MLflow run, including its hyperparameters and evaluation metrics.

### Model Evaluation and Selection

Models are evaluated using MAE, RMSE, and MAPE on a chronological test set.

RMSE is the primary selection metric, while MAE and MAPE are used as supporting metrics. A model is promoted to Production only after reviewing its test results.

- Registered model: `EastJavaFoodPriceForecast`
- Active version: `v[actual version]`
- Active Run ID: `[actual Run ID]`
- Test RMSE: `[actual RMSE]`
- Test MAE: `[actual MAE]`
- Selection rationale: `[explain why this version was selected]`

### Model Versioning with MLflow

MLflow Model Registry maintains the model version history. The initial registered model becomes the first version if the registered model name has not been used before. Subsequent registrations under the same name create additional versions.

The selected model transitions through the Staging and Production stages. The
`champion` alias identifies the version selected for inference when the registry
backend supports alias lookups.

### Artifact and Dataset Lineage with DVC

The selected model artifact and its `model_release.yaml` manifest are stored in a release directory under `project/models/releases/`.

The manifest records the model Registry version, source Run ID, metrics, parameters, dataset path, and DVC hash of the training feature dataset.

DVC tracks the release directory through a pointer file such as:

`project/models/releases/v1.dvc`

Git stores the DVC pointer and project source files, while the configured DagsHub
DVC remote stores the corresponding model release artifacts.

### Inference Verification

The model is loaded programmatically through `mlflow.pyfunc.load_model()` and tested using the feature columns expected by the Ridge Regression pipeline.

The inference script uses the Production model URI:

`models:/EastJavaFoodPriceForecast/Production`

The project also assigns the `champion` alias on promotion. If the DagsHub
registry returns `INVALID_PARAMETER_VALUE` while resolving that alias, use the
Production URI instead.


## Model Inference API and Docker

### Overview

This project provides a FastAPI REST API for serving the trained Ridge Regression model used for East Java food price forecasting.

The API loads `models:/EastJavaFoodPriceForecast@champion` directly from the
DagsHub MLflow Model Registry when `DAGSHUB_TOKEN` is configured. Without that
token, local development falls back to `model.joblib`. The model includes its
preprocessing pipeline, including numerical imputation, standardization, and
categorical encoding.

The standalone FastAPI container is separate from the Compose deployment.
The Compose instructions use the `Production` stage URI because DagsHub returned
`INVALID_PARAMETER_VALUE` for the `@champion` URI in the MLflow CLI test.

The inference API exposes two endpoints:

- `GET /health`: Checks whether the service is running and the model is loaded.
- `POST /predict`: Accepts food price and weather features as JSON and returns each prediction together with its city and commodity.
- `GET /predict/latest`: Uses the latest complete feature row for every city and commodity in the bundled dataset.

These endpoints describe the standalone FastAPI container. The Compose
deployment below serves the DagsHub-registered MLflow model directly through
`/invocations`.

### Project Structure

project/deployment/
├── app.py
├── Dockerfile
├── nginx.conf
├── requirements.txt
├── sample_request.json
├── .dockerignore
└── model.joblib

### Exporting the Model

The selected registered model is exported from MLflow before building the Docker image.

```bash
python project/src/export_model_docker.py
```

The exported model is saved to `project/deployment/model.joblib`.

### Build the Docker Image

Run the following command from the repository root:

docker build \
  -f project/deployment/Dockerfile \
  -t food-price-api:1.0 \
  .

### Run the Container

docker run -d \
  --name food-price-api \
  -p 8000:8000 \
  food-price-api:1.0

Check the running container using: docker ps

### Health Check

curl http://localhost:8000/health

### Inference Request

Send a POST request to `/predict` from terminal/bash:

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [
      {
        "price_lag_1": 30500,
        "price_lag_7": 29000,
        "rolling_mean_7": 30000,
        "price_change": 500,
        "day_of_week": 3,
        "month": 10,
        "temperature_mean": 28.5,
        "temperature_max": 33.2,
        "precipitation_sum": 2.0,
        "humidity_mean": 78.0,
        "wind_speed_mean": 8.5,
        "city": "Kota Surabaya",
        "commodity": "Cabai Rawit",
        "weather_code": "61"
      }
    ]
  }'

The API returns a JSON response containing each predicted food price and its city/commodity label.

### Running the orchestrated system

The standalone FastAPI instructions above use `/predict`. The root
`docker-compose.yaml` instead serves the Production model from the DagsHub
MLflow Registry with MLflow's `/invocations` endpoint. It does not use the
local SQLite registry or `model.joblib`.

Set the DagsHub tracking URI and credentials in the terminal. If
`DAGSHUB_TOKEN` is not already available as a Codespaces secret, enter it
without echoing it:

```bash
if [ -z "${DAGSHUB_TOKEN:-}" ]; then
  read -rsp "DagsHub token: " DAGSHUB_TOKEN
  printf '\n'
  export DAGSHUB_TOKEN
fi
export DAGSHUB_REPO_OWNER=aswitanaufal
export DAGSHUB_REPO_NAME=food-price-forecast-east-java-cities
export MLFLOW_TRACKING_URI="https://dagshub.com/${DAGSHUB_REPO_OWNER}/${DAGSHUB_REPO_NAME}.mlflow"
export MLFLOW_REGISTRY_URI="$MLFLOW_TRACKING_URI"
export MLFLOW_TRACKING_USERNAME="$DAGSHUB_REPO_OWNER"
export MLFLOW_TRACKING_PASSWORD="$DAGSHUB_TOKEN"
```

The automated training workflow registers each passing version as `Staging`;
it does not promote models to `Production` automatically. Review the
registered versions and their stages in DagsHub:

```bash
python - <<'PY'
from mlflow import MlflowClient

versions = MlflowClient().search_model_versions(
    "name='EastJavaFoodPriceForecast'"
)
for model in sorted(versions, key=lambda item: int(item.version)):
    print(
        f"version={model.version}, stage={model.current_stage}, "
        f"status={model.status}, run_id={model.run_id}"
    )
PY
```

After reviewing the metrics, promote the selected version in the DagsHub
repository's MLflow **Models** page: open `EastJavaFoodPriceForecast`, select
the chosen version, and change its stage to `Production`. The automated
workflow only registers models as `Staging`.

The CLI promotion script is also available:

```bash
python project/src/promote_model.py --version VERSION --stage Production
```

Replace `VERSION` with the reviewed version number. If DagsHub responds with
HTTP 403 on the stage-transition request, verify that the token belongs to a
user with write access to this repository. Use the DagsHub Models page to
promote the version if the API continues to deny the transition. Check the
version's stage again before proceeding.

Confirm the promotion before serving:

```bash
python - <<'PY'
from mlflow import MlflowClient

versions = MlflowClient().search_model_versions(
    "name='EastJavaFoodPriceForecast'"
)
production = [model for model in versions if model.current_stage == "Production"]
if not production:
    raise SystemExit("No EastJavaFoodPriceForecast version is in Production.")
for model in production:
    print(f"version={model.version}, stage={model.current_stage}")
PY
```

To test the registered model locally before containerizing it, start MLflow
serving in one terminal. The DagsHub model URI verified for this deployment is
`models:/Production/latest`; use the same URI when building the container image:

```bash
mlflow models serve \
  --model-uri models:/Production/latest \
  --host 127.0.0.1 \
  --port 5001 \
  --env-manager local
```

In another terminal, capture the API test output for the report:

```bash
curl -i http://localhost:5001/ping
curl -i http://localhost:5001/invocations \
  -H "Content-Type: application/json" \
  --data-binary @project/deployment/sample_request.json
```

Stop the local server with `Ctrl+C`, then build the Docker image from the
DagsHub Production model and start the three-replica Compose service:

```bash
mlflow models build-docker \
  --model-uri models:/Production/latest \
  --name east-java-food-price-model:dagshub-production
docker compose up -d --wait
```

Capture the running replica evidence and test the Compose endpoint:

```bash
docker compose ps mlflow-model
curl -i http://localhost:8000/ping
curl -i http://localhost:8000/invocations \
  -H "Content-Type: application/json" \
  --data-binary @project/deployment/sample_request.json
```

The MLflow model image contains the Production artifact resolved from DagsHub
at build time, so the token is not passed into the serving containers. When a
new version is promoted to `Production`, rerun `mlflow models build-docker`
with the same tag and recreate the containers to serve it.

The Compose file sets `deploy.replicas: 3`. To simulate a higher load, override
the number of replicas and verify the result in `docker compose ps`:

```bash
docker compose up -d --scale mlflow-model=5 --wait
docker compose ps mlflow-model
```

Replace `5` with the desired count. Return to three instances with
`docker compose up -d --scale mlflow-model=3 --wait`. This uses Docker Compose
scaling, not Docker Swarm.

### Model Artifact Versioning

The exported `model.joblib` is tracked using DVC for the standalone FastAPI
image. Git stores the DVC pointer file, while the configured DVC remote stores
the corresponding model artifact.

Before building the image in a fresh working environment, restore the model artifact if necessary:

dvc pull project/deployment/model.joblib.dvc


The Docker image contains the application, the required Python dependencies, and the exported model. The running API therefore does not require access to the MLflow UI or its tracking database for every prediction request.
