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
This project leverages **MLflow** with a local SQLite backend (`sqlite:///mlflow.db`) to track hyperparameter configurations, evaluation metrics, pipeline metadata, and serialized model artifacts across training runs.

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

To view the experiments in MLflow UI:
1. Launch MLflow UI from the root directory: 
mlflow ui --backend-store-uri sqlite:///mlflow.db
2. Access the Dashboard:
    - Local Environment: Open http://127.0.0.1:5000 in your web browser.
    - GitHub Codespaces: Open the Ports tab in the bottom panel and click the forwarded link/globe icon for port 5000.
3. Analyze Results:
    - Compare runs side-by-side by selecting individual runs and clicking Compare.
    - Customize displayed table columns to sort models by metrics (MAE, RMSE, MAPE).
    - Inspect saved artifacts, input schemas, and environment dependencies directly inside the UI under model_dir.