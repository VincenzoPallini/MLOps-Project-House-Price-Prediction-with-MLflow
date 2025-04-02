# src/train.py (Refactored with Hydra)
import hydra
from omegaconf import DictConfig, OmegaConf # Importa da Hydra/OmegaConf
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler # Importa entrambi se vuoi sceglierli da config
from sklearn.linear_model import Ridge # Potresti importare altri modelli qui
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
import os
from flatten_dict import flatten # Per loggare parametri annidati

# Importa i moduli refactored
from load_data import load_raw_data
from preprocess import preprocess_data
from evaluate import calculate_regression_metrics, calculate_metrics_on_original_scale
from hydra.utils import to_absolute_path # Utility per gestire i path
from mlflow.models import infer_signature # Per la signature (opzionale ma buono)


warnings.filterwarnings("ignore")


# La funzione principale ora è decorata con @hydra.main
# config_path: dice a Hydra dove cercare la cartella 'conf' (relativa a questo script)
# config_name: nome del file di configurazione principale (senza .yaml)
# version_base=None: raccomandato per compatibilità futura di Hydra
@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main_training_workflow(cfg: DictConfig): # Riceve l'oggetto config 'cfg'
    """
    Orchestra il workflow usando la configurazione fornita da Hydra (cfg).
    """
    # Hydra cambia la directory di lavoro, quindi risolviamo i path
    # relativi alla directory originale del progetto usando to_absolute_path
    raw_train_path = to_absolute_path(cfg.data.raw_train_path)
    raw_test_path = to_absolute_path(cfg.data.raw_test_path)
    processed_data_dir = to_absolute_path(cfg.data.processed_dir)

    print(f"--- Avvio Workflow di Training (Hydra) ---")
    print(f"Configurazione Caricata:\n{OmegaConf.to_yaml(cfg)}") # Stampa la config usata

    # --- 1. Caricamento Dati Raw ---
    df_train_raw, df_test_raw = load_raw_data(raw_train_path, raw_test_path)
    if df_train_raw is None: return

    # --- 2. Preprocessing ---
    X_train_proc, X_test_proc, y_train_log = preprocess_data(
        df_train_raw, df_test_raw, processed_data_dir
    )
    if X_train_proc is None: return

    # --- 3. Split Train/Validation ---
    X_subtrain, X_val, y_subtrain, y_val = train_test_split(
        X_train_proc, y_train_log,
        test_size=cfg.validation_split_ratio, # Usa parametro da config
        random_state=cfg.random_state         # Usa parametro da config
    )
    print(f"Split Train/Validation: Subtrain={X_subtrain.shape}, Validation={X_val.shape}")

    # --- 4. MLflow Run ---
    mlflow.set_experiment(cfg.mlflow_experiment_name) # Usa nome esperimento da config
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"MLflow Run avviata: {run_id}")
        print(f"MLflow Run URL: {mlflow.get_tracking_uri()}/#/experiments/{run.info.experiment_id}/runs/{run_id}")


        # --- 5. Log Parametri (dalla config Hydra) ---
        print("Logging parametri (flattened config) su MLflow...")
        # Converti l'oggetto OmegaConf in dizionario e appiattiscilo
        flat_cfg = flatten(OmegaConf.to_container(cfg, resolve=True), reducer='dot')
        mlflow.log_params(flat_cfg)

        # --- 6. Definizione e Training Pipeline ---
        # Scegli lo scaler dalla configurazione
        if cfg.model.scaler == 'robust':
            scaler = RobustScaler()
        elif cfg.model.scaler == 'standard':
            scaler = StandardScaler()
        else:
            print(f"Warning: Scaler '{cfg.model.scaler}' non riconosciuto, uso RobustScaler.")
            scaler = RobustScaler()

        # Istanzia il modello dalla configurazione (per ora solo Ridge)
        if cfg.model.name == 'ridge':
            model = Ridge(alpha=cfg.model.alpha, random_state=cfg.random_state)
            print(f"Usando Ridge model con alpha={cfg.model.alpha}")
        # Potresti aggiungere qui if/else per altri modelli (es. cfg.model.name == 'lasso')
        else:
            print(f"Errore: Modello '{cfg.model.name}' non supportato.")
            return

        pipeline = Pipeline([('scaler', scaler), (cfg.model.name, model)])

        print("Addestramento pipeline...")
        pipeline.fit(X_subtrain, y_subtrain)
        print("Addestramento completato.")

        # --- 7. Valutazione ---
        print("Valutazione su Validation Set...")
        y_pred_val_log = pipeline.predict(X_val)
        metrics_log = calculate_regression_metrics(y_val, y_pred_val_log)
        metrics_orig = calculate_metrics_on_original_scale(y_val, y_pred_val_log)
        mlflow.log_metrics(metrics_log)
        mlflow.log_metrics(metrics_orig)

        # --- 8. Logging Modello (con Signature) ---
        artifact_path = f"{cfg.model.name}-model-pipeline" # Nome artefatto dinamico
        print(f"Logging del modello (pipeline) come '{artifact_path}' con signature...")
        input_sample = X_subtrain.head()
        output_sample = pipeline.predict(input_sample)
        signature = infer_signature(input_sample, output_sample)
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path=artifact_path,
            signature=signature,
            input_example=input_sample
        )

        print(f"MLflow Run {run_id} completata.")

    print(f"--- Fine Workflow di Training (Hydra) ---")


# Entry point: Hydra si occupa di chiamare questa funzione
if __name__ == "__main__":
    main_training_workflow()
    print("\nScript train.py (Hydra) terminato.")