import argparse
import pandas as pd
import numpy as np
import mlflow
import os
import warnings
import sys 

warnings.filterwarnings("ignore")

def calculate_stats(data_series, prefix):
    """Calcola statistiche descrittive per una Serie Pandas."""
    stats = {
        f"{prefix}_mean": data_series.mean(),
        f"{prefix}_std": data_series.std(),
        f"{prefix}_min": data_series.min(),
        f"{prefix}_max": data_series.max(),
        f"{prefix}_p25": data_series.quantile(0.25),
        f"{prefix}_p50": data_series.median(), 
        f"{prefix}_p75": data_series.quantile(0.75),
        f"{prefix}_count": len(data_series)
    }
    return stats

def predict_and_log(model_uri, input_data_path, experiment_name="Prediction_Logs"):
    """
    Carica un modello, fa predizioni su nuovi dati e logga le statistiche
    delle predizioni in una nuova run MLflow.
    """
    print(f"--- Avvio Predizione e Logging ---")
    print(f"Modello URI: {model_uri}")
    print(f"Dati Input (processati): {input_data_path}") 

    # --- 1. Caricamento Modello ---
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        print("Modello caricato con successo.")
    except Exception as e:
        print(f"Errore durante il caricamento del modello: {e}")
        return

    # --- 2. Caricamento Dati Input (Processati) ---

    if not os.path.exists(input_data_path):
        print(f"Errore: File dati input non trovato: {input_data_path}")
        print("Assicurati di aver eseguito prima il preprocessing (es. python MLOps/src/preprocess.py) per generarlo.")

        return 
    try:
        input_data = pd.read_parquet(input_data_path)
        print(f"Dati di input caricati: {input_data.shape}")
    except Exception as e:
        print(f"Errore durante il caricamento dei dati di input: {e}")
        return

    # --- 3. Esecuzione Predizioni ---
    try:
        print("Esecuzione predizioni...")
        y_pred_log = model.predict(input_data)
        y_pred_log_series = pd.Series(y_pred_log, name="predictions_log")
        y_pred_orig_series = pd.Series(np.expm1(y_pred_log), name="predictions_original")
        print(f"Predizioni generate per {len(y_pred_log_series)} campioni.")
    except Exception as e:
        print(f"Errore durante la predizione: {e}")
        return

    # --- 4. Calcolo Statistiche Predizioni ---
    print("Calcolo statistiche delle predizioni...")
    pred_log_stats = calculate_stats(y_pred_log_series, "pred_log")
    pred_orig_stats = calculate_stats(y_pred_orig_series, "pred_orig")

    # --- 5. Logging MLflow ---
    try:
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run() as run:
            print(f"Avviata run MLflow per logging predizioni: Run ID {run.info.run_id}")
            mlflow.log_param("model_uri_used", model_uri)
            mlflow.log_param("input_data_path", input_data_path) 
            mlflow.log_param("num_predictions", len(y_pred_log_series))

            print("Logging statistiche predizioni su MLflow...")
            mlflow.log_metrics(pred_log_stats)
            mlflow.log_metrics(pred_orig_stats)



            print("Logging completato.")
    except Exception as e:
        print(f"Errore durante il logging MLflow: {e}")

    print(f"--- Fine Predizione e Logging ---")

if __name__ == "__main__":
    try:
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        MLOPS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
        DEFAULT_INPUT_PATH = os.path.join(MLOPS_DIR, "data", "processed", "test_features_processed.parquet")
        print(f"Percorso di default calcolato per i dati di input: {DEFAULT_INPUT_PATH}")
    except NameError:

        print("Attenzione: Impossibile determinare dinamicamente il percorso dello script.")
        print("Il percorso di default per --input-data potrebbe non essere corretto se non specificato.")
        DEFAULT_INPUT_PATH = "data/processed/test_features_processed.parquet" 


    # --- Definizione Argomenti ---
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-uri", type=str, required=True,
        help="MLflow Model URI (es. 'models:/HousePricePredictor/1' o 'runs:/RUN_ID/artifact_path')"
    )
    parser.add_argument(
        "--input-data", type=str,
        default=DEFAULT_INPUT_PATH, 
        help=f"Percorso al file Parquet con i dati di test processati (default: {DEFAULT_INPUT_PATH})"
    )
    parser.add_argument(
        "--experiment-name", type=str, default="Prediction_Logs",
        help="Nome dell'esperimento MLflow dove loggare le statistiche"
    )
    args = parser.parse_args()

    predict_and_log(
        model_uri=args.model_uri,
        input_data_path=args.input_data, 
        experiment_name=args.experiment_name
    )