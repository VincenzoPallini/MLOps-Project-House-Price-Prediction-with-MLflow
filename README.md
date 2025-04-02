# MLOps Project: House Price Prediction

## Project Overview

This project demonstrates an advanced application of MLOps principles to predict house prices using the Kaggle House Prices dataset.

**Dataset:** [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) from Kaggle.

## Key Features & Enhanced MLOps Practices

* **End-to-End Orchestrated Workflow (Prefect):** The entire ML pipeline (data loading, preprocessing, training, evaluation) is defined and orchestrated as a Prefect flow (`run_pipeline.py`), managing task dependencies and execution state.
* **Configuration Management (Hydra):** Replaced command-line arguments with Hydra for flexible and structured configuration management using YAML files (`conf/`), allowing easy parameter overrides and composition.
* **Experiment Tracking (MLflow Tracking):** Systematically logged parameters (including full Hydra config), metrics (RMSE, MAE, R2 on log and original scales), source code versions (Git commits), and model artifacts (Scikit-learn Pipelines) for each training run using MLflow.
* **Data Version Control (DVC):** Integrated DVC to version control the raw dataset (`data/raw`), ensuring data reproducibility alongside code reproducibility. Uses local remote storage for this implementation.
* **Containerization (Docker):** Provided a `Dockerfile` to build a container image with all necessary dependencies (OS, Python, Conda, libraries), ensuring a consistent and portable execution environment. Includes `.dockerignore` for optimized image builds.
* **Continuous Integration (GitHub Actions - Basics):** Implemented a basic CI pipeline using GitHub Actions (`.github/workflows/ci.yaml`) that automatically runs code linting (Flake8) on push/pull requests to the main branch. Includes `pytest` structure for future test expansion.
* **Modular Code Structure:** Refactored the codebase into distinct, reusable Python modules within the `src/` directory (`load_data.py`, `preprocess.py`, `train.py`, `evaluate.py`) executed as tasks within the Prefect flow.
* **Model Management (MLflow Models & Registry):** Saved trained models (Scikit-learn pipelines) in MLflow's standard format. Implemented **programmatic registration** of models to the MLflow Model Registry upon successful training, including adding descriptions and tags via `MlflowClient`. Manual lifecycle management (staging) via UI is still possible.
* **Prediction Statistics Logging (Basic Monitoring):** Included a script (`src/predict.py`) to load a registered model, perform predictions, and log descriptive statistics of the output distribution to a dedicated MLflow experiment, forming a basis for future prediction drift monitoring.
* **Version Control (Git):** Utilized Git extensively for versioning all code, configuration files, DVC metafiles, Docker files, and CI workflows.

## Technologies Used

* **Core Language:** Python
* **MLOps & Workflow:** MLflow (Tracking, Models, Registry), Prefect (Orchestration), Hydra (Configuration), DVC (Data Versioning), Docker (Containerization), GitHub Actions (CI - Linting)
* **Machine Learning:** Scikit-learn (Pipeline, Ridge, RobustScaler, StandardScaler, train_test_split, metrics)
* **Utilities:** SciPy (skewness), OmegaConf (Hydra config), flatten-dict (logging)
* **Testing/Linting:** Pytest, Flake8
* **Environment Management:** Conda


## Orchestrated Workflow Summary (via `run_pipeline.py`)

1.  **Initialization:** The Prefect flow starts, manually initializing Hydra to load the configuration from YAML files (`conf/`).
2.  **Load Data (`load_data` task):** Reads raw `train.csv` and `test.csv`.
3.  **Preprocess Data (`preprocess_data` task):** Takes raw dataframes, performs outlier removal, target separation/transformation, NaN imputation, feature engineering, skewness correction, encoding, and saves processed dataframes (and target) to Parquet files in `data/processed/`.
4.  **Train Model (`training_task` task):**
    * Receives Hydra configuration and processed training data/target.
    * Performs train/validation split.
    * Starts an MLflow run.
    * Logs flattened Hydra configuration as MLflow parameters.
    * Builds and trains a Scikit-learn `Pipeline` (Scaler + Model specified in config, e.g., Ridge).
    * Evaluates the pipeline on the validation set using functions from `evaluate.py` (calculating RMSE, MAE, R2 on log and original scales).
    * Logs calculated metrics to MLflow.
    * Logs the trained pipeline (with signature and input example) as an MLflow artifact.
    * **Programmatically registers** the logged model artifact to the specified MLflow Model Registry name (`HousePricePredictor`), adding descriptive tags and description.
5.  **Flow Completion:** Prefect marks the flow run as completed (or failed if any task fails).

## Results

* Successfully implemented and orchestrated a complex, reproducible MLOps pipeline.
* Demonstrated effective use of Hydra for configuration and Prefect for task orchestration.
* Utilized MLflow Tracking to systematically compare experiments run via the orchestrated pipeline.
* Integrated DVC for data versioning and Docker for environment containerization.
* Set up basic CI linting checks with GitHub Actions.
* Automated model registration into the MLflow Model Registry.
