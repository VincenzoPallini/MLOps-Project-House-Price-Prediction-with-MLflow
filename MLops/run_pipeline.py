print("--- Script run_pipeline.py Iniziato ---")

import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from hydra.core.global_hydra import GlobalHydra
from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf
from prefect import flow, get_run_logger
from src.load_data import load_raw_data
from src.preprocess import preprocess_data
from src.train import training_task


print("--- Import completati ---")

print("--- Prima del decorator @flow ---")

@flow(name="House Price Prediction Pipeline")
def pipeline_flow():
    """
    Flow Prefect che orchestra la pipeline.
    Carica manualmente la configurazione Hydra all'inizio.
    """
    print("--- !!! DENTRO pipeline_flow (Inizio Esecuzione Flow) !!! ---")
    logger = get_run_logger()
    logger.info("--- Avvio Pipeline Flow (Manuale Hydra Init) ---")


    GlobalHydra.instance().clear()

    try:
        initialize(config_path="conf", version_base=None)

        cfg = compose(config_name="config")
        logger.info(f"Configurazione Hydra caricata manualmente:\n{OmegaConf.to_yaml(cfg)}")
    except Exception as hydra_e:
        logger.error(f"Errore durante l'inizializzazione manuale di Hydra: {hydra_e}")
        raise hydra_e 


    current_dir = os.getcwd() 
    logger.info(f"Directory corrente: {current_dir}")
    raw_train_path = os.path.join(current_dir, cfg.data.raw_train_path)
    raw_test_path = os.path.join(current_dir, cfg.data.raw_test_path)
    processed_data_dir = os.path.join(current_dir, cfg.data.processed_dir)
    logger.info(f"Path dati raw train (risolto): {raw_train_path}")
    logger.info(f"Path dati processati (risolto): {processed_data_dir}")

    logger.info("Avvio task: load_raw_data")
    df_train_raw, df_test_raw = load_raw_data(raw_train_path, raw_test_path)

    logger.info("Avvio task: preprocess_data")
    preprocess_result_tuple = preprocess_data(
        df_train_raw=df_train_raw, 
        df_test_raw=df_test_raw,   
        processed_data_dir=processed_data_dir,
    )

    try:
        logger.info("Ottenuti risultati da preprocess_data...")
        X_train_proc, _, y_train_log = preprocess_result_tuple
        logger.info("Risultati preprocess_data ottenuti e unpackati.")
    except Exception as e:
         logger.error(f"Errore durante l'unpacking dei risultati di preprocess_data: {e}")
         logger.error(f"Tipo di preprocess_result_tuple: {type(preprocess_result_tuple)}")
         if isinstance(preprocess_result_tuple, tuple):
             logger.error(f"Lunghezza della tupla: {len(preprocess_result_tuple)}")
         raise e

    logger.info("Avvio task: training_task")

    train_result_future = training_task.submit(
        cfg=cfg,
        X_train_proc=X_train_proc,
        y_train_log=y_train_log,
    )

    try:
        logger.info("Attesa risultati training_task...")
        final_output = train_result_future.result() 
        logger.info(f"\n--- Risultato finale dal training_task ---")
        logger.info(f"  MLflow Run ID: {final_output.get('run_id')}")
        logger.info(f"  Metriche: {final_output.get('metrics')}")
    except Exception as e:
        logger.error(f"Task training_task fallito: {e}")
        raise

    logger.info("--- Fine Pipeline Flow ---")

if __name__ == "__main__":
    print("--- Chiamata a pipeline_flow dal blocco main ---")
    pipeline_flow()
    print("\nScript run_pipeline.py terminato.")