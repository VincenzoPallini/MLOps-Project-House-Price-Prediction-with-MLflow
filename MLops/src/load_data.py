import pandas as pd
import os
from prefect import task 
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

@task 
def load_raw_data(train_path, test_path):
    """
    Carica i file CSV raw di training e test, rimuove la colonna 'Id'.
    Restituisce due DataFrame pandas.
    """
    print("--- Esecuzione load_raw_data TASK ---") # Aggiornato print per chiarezza
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"Errore: File non trovati in {train_path} o {test_path}")
        raise FileNotFoundError(f"Raw data files not found at {train_path} or {test_path}")

    try:
        df_train_raw = pd.read_csv(train_path)
        df_test_raw = pd.read_csv(test_path)
        print(f"Dati raw caricati: Train={df_train_raw.shape}, Test={df_test_raw.shape}")

        if 'Id' in df_train_raw.columns:
            df_train_raw = df_train_raw.drop('Id', axis=1)
        if 'Id' in df_test_raw.columns:
            df_test_raw = df_test_raw.drop('Id', axis=1)
        print("Colonna 'Id' rimossa dai dataframe raw (se presente).")

        return df_train_raw, df_test_raw

    except Exception as e:
        print(f"Errore durante il caricamento dei dati raw: {e}")
        raise e

