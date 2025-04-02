import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
import os
from flatten_dict import flatten
from omegaconf import DictConfig, OmegaConf
from prefect import task
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient 

from src.evaluate import calculate_regression_metrics, calculate_metrics_on_original_scale

warnings.filterwarnings("ignore")

@task
def training_task(cfg: DictConfig, X_train_proc, y_train_log):
    """
    Esegue split, training, valutazione, logging MLflow E REGISTRAZIONE MODELLO.
    Restituisce l'ID della run MLflow e le metriche calcolate.
    """
    print(f"--- Esecuzione training_task TASK ---")
    X_subtrain, X_val, y_subtrain, y_val = train_test_split(
        X_train_proc, y_train_log,
        test_size=cfg.validation_split_ratio,
        random_state=cfg.random_state
    )
    print(f"Split interno: Subtrain={X_subtrain.shape}, Validation={X_val.shape}")

    # --- MLflow Run ---
    mlflow.set_experiment(cfg.mlflow_experiment_name)
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"MLflow Run avviata per training: {run_id}")
        all_metrics = {} 

        try:
            # --- Log Parametri ---
            print("Logging parametri su MLflow...")
            flat_cfg = flatten(OmegaConf.to_container(cfg, resolve=True), reducer='dot')
            mlflow.log_params(flat_cfg)

            # --- Definizione e Training Pipeline ---
            if cfg.model.scaler == 'robust': scaler = RobustScaler()
            elif cfg.model.scaler == 'standard': scaler = StandardScaler()
            else: scaler = RobustScaler()

            if cfg.model.name == 'ridge':
                model_alpha = float(cfg.model.alpha)
                model = Ridge(alpha=model_alpha, random_state=cfg.random_state)
                print(f"Usando Ridge model con alpha={model_alpha}")
            else: raise ValueError(f"Modello '{cfg.model.name}' non supportato.")

            pipeline = Pipeline([('scaler', scaler), (cfg.model.name, model)])
            print("Addestramento pipeline...")
            pipeline.fit(X_subtrain, y_subtrain)
            print("Addestramento completato.")

            # --- Valutazione ---
            print("Valutazione su Validation Set...")
            y_pred_val_log = pipeline.predict(X_val)
            metrics_log = calculate_regression_metrics(y_val, y_pred_val_log)
            metrics_orig = calculate_metrics_on_original_scale(y_val, y_pred_val_log)
            mlflow.log_metrics(metrics_log)
            mlflow.log_metrics(metrics_orig)
            all_metrics = {**metrics_log, **metrics_orig}

            # --- Logging Modello ---
            artifact_path = f"{cfg.model.name}-model-pipeline"
            print(f"Logging del modello come '{artifact_path}'...")
            input_sample = X_subtrain.head()
            output_sample = pipeline.predict(input_sample)
            signature = infer_signature(input_sample, output_sample)
            mlflow.sklearn.log_model(
                sk_model=pipeline, artifact_path=artifact_path,
                signature=signature, input_example=input_sample
            )


            print("Avvio registrazione programmatica modello...")
            model_uri = f"runs:/{run_id}/{artifact_path}"
            model_name_registry = cfg.model_registry_name # Nome dal config


            model_version_details = mlflow.register_model(
                model_uri=model_uri,
                name=model_name_registry
            )
            version_number = model_version_details.version
            print(f"Modello registrato: Nome='{model_name_registry}', Versione={version_number}")

            client = MlflowClient()
            description = (f"Modello {cfg.model.name.capitalize()} (alpha={cfg.model.alpha}, "
                           f"scaler={cfg.model.scaler}) addestrato nella run {run_id}. "
                           f"Validation RMSE: {all_metrics.get('rmse', 'N/A'):.5f}")
            client.update_model_version(
                name=model_name_registry,
                version=version_number,
                description=description
            )
            print(f"Aggiunta descrizione alla Versione {version_number}.")

            client.set_model_version_tag(model_name_registry, version_number, "validation_rmse", f"{all_metrics.get('rmse', 'N/A'):.5f}")
            client.set_model_version_tag(model_name_registry, version_number, "run_id", run_id)
            client.set_model_version_tag(model_name_registry, version_number, "model_type", cfg.model.name)
            client.set_model_version_tag(model_name_registry, version_number, "alpha", str(cfg.model.alpha))
            print(f"Aggiunti tag alla Versione {version_number}.")

            print(f"MLflow Run {run_id} completata con successo.")

        except Exception as train_eval_e:
            print(f"ERRORE durante training/valutazione/logging/registrazione: {train_eval_e}")
            mlflow.log_param("error", str(train_eval_e))
            raise train_eval_e 

    return {"run_id": run_id, "metrics": all_metrics}

