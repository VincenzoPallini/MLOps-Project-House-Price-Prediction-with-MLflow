# MLOps Project: House Price Prediction with MLflow

## Project Overview

This project demonstrates practical MLOps principles applied to a real-world regression task: predicting house prices using the Kaggle House Prices dataset. The core focus is **not just on building a predictive model, but on establishing a robust, reproducible, and trackable machine learning workflow using MLflow**.

**Dataset:** [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) from Kaggle.

## Key Features & MLOps Practices 

* **End-to-End ML Workflow:** Implementation covering data preprocessing, feature engineering, model training (Ridge Regression), evaluation, and experiment tracking.
* **Experiment Tracking (MLflow Tracking):** Systematically logged parameters (like regularization strength `alpha`), performance metrics (Validation RMSE) and model artifacts for each training run, enabling comparison and reproducibility.
* **Reproducible Runs (MLflow Projects):** Packaged the ML workflow with a defined Conda environment (`conda.yaml`) and entry points (`MLproject`), allowing anyone to reproduce the training runs reliably using a single `mlflow run` command.
* **Model Management (MLflow Models & Registry UI):** Saved trained models (Scikit-learn pipelines including scaler + model) in MLflow's standard format. Utilized the MLflow UI's Model Registry features to version models and manage their lifecycle stages (e.g., Staging, Production) locally.

## Technologies Used

* **Core Language:** Python 
* **MLOps & Experiment Tracking:** MLflow (Tracking, Projects, Models, Registry)
* **Machine Learning:** Scikit-learn (Pipeline, Ridge, RobustScaler, train_test_split, mean_squared_error)
* **Data Manipulation & Analysis:** Pandas, NumPy, SciPy (for skewness calculation)
* **Environment Management:** Conda

## Workflow Summary

1.  **Data Preprocessing (`src/preprocess.py`):** Loaded raw data, handled missing values using various strategies (mean/median/mode/constant), performed feature engineering (e.g., TotalSF, HouseAge), corrected skewed numerical features (log transform), and applied one-hot encoding to categorical features.
2.  **Model Training & Tracking (`src/train.py`):**
    * Split processed data into training and validation sets.
    * Defined a Scikit-learn `Pipeline` including `RobustScaler` and `Ridge` regression.
    * Started an MLflow run for each training execution.
    * Logged hyperparameters (e.g., `alpha`), input feature count, and data paths.
    * Trained the pipeline.
    * Evaluated the model on the validation set using Root Mean Squared Error (RMSE) on the log-transformed target.
    * Logged the validation RMSE metric to MLflow.
    * Logged the entire trained pipeline as an MLflow artifact.
3.  **Experimentation & Packaging:**
    * Ran the training script multiple times with different `alpha` values using `mlflow run . -P alpha=...`.
    * Used the MLflow UI (`mlflow ui`) to compare runs and identify the best hyperparameters based on validation RMSE.
    * Defined the project structure and environment in `MLproject` and `conda.yaml` for reproducible execution.
    * (Locally) Registered promising model versions in the MLflow Model Registry via the UI.

## Results

* Successfully implemented a reproducible ML pipeline for training a Ridge regression model on the House Prices dataset.
* Utilized MLflow Tracking to systematically compare experiments with varying regularization strengths (`alpha`).
* Identified the best performing Ridge model based on the validation set RMSE.
* Demonstrated the ability to package the project using MLflow Projects for easy and reliable re-execution.

