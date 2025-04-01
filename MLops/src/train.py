# src/train.py
import argparse
import os
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

import mlflow
import mlflow.sklearn

from preprocess import preprocess_data

warnings.filterwarnings("ignore")

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    return rmse


def train_model(train_data_path, test_data_path, alpha):
    """
    Esegue preprocessing, training, valutazione e logging MLflow.
    """
    print(f"Avvio training con alpha={alpha}")
    print(f"Dati raw: train='{train_data_path}', test='{test_data_path}'")

    # --- 1. Preprocessing ---

    X_train_proc, X_test_proc, y_train_log = preprocess_data(train_data_path, test_data_path)

    if X_train_proc is None:
        print("Errore durante il preprocessing. Interruzione.")
        return

    # --- 2. Split Train/Validation ---

    X_subtrain, X_val, y_subtrain, y_val = train_test_split(
        X_train_proc, y_train_log, test_size=0.2, random_state=42
    )
    print(f"Split Train/Validation: Subtrain={X_subtrain.shape}, Validation={X_val.shape}")

    # --- 3. MLflow Run ---

    with mlflow.start_run():
        print("MLflow Run avviata.")


        pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('ridge', Ridge(alpha=alpha, random_state=42))
        ])

        print("Addestramento pipeline (Scaler + Ridge)...")
        pipeline.fit(X_subtrain, y_subtrain)
        print("Addestramento completato.")

        y_pred_val = pipeline.predict(X_val)
        rmse_val = eval_metrics(y_val, y_pred_val)
        print(f"  RMSE su Validation Set: {rmse_val:.5f}")

        # --- 6. Logging MLflow ---

        print("Logging parametri su MLflow...")
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("train_data_path", train_data_path)
        mlflow.log_param("test_data_path", test_data_path)
        mlflow.log_param("validation_split_ratio", 0.2)
        mlflow.log_param("input_features", X_train_proc.shape[1])


        print("Logging metrica (RMSE) su MLflow...")
        mlflow.log_metric("rmse_validation", rmse_val)


        print("Logging del modello (pipeline) su MLflow...")
        mlflow.sklearn.log_model(pipeline, "ridge-model-pipeline")

        print(f"MLflow Run ID: {mlflow.active_run().info.run_id} completata.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-data", type=str, required=True,
        help="Percorso al file train.csv raw"
    )
    parser.add_argument(
        "--test-data", type=str, required=True,
        help="Percorso al file test.csv raw"
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0,
        help="Parametro di regolarizzazione alpha per Ridge"
    )

    args = parser.parse_args()

    train_model(
        train_data_path=args.train_data,
        test_data_path=args.test_data,
        alpha=args.alpha
    )

    print("\nScript train.py terminato.")