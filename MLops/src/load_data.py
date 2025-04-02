# src/load_data.py
import pandas as pd
import os

def load_raw_data(train_path, test_path):
    """
    Carica i file CSV raw di training e test, rimuove la colonna 'Id'.
    Restituisce due DataFrame pandas.
    """
    print("--- Esecuzione load_raw_data ---")
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"Errore: File non trovati in {train_path} o {test_path}")
        return None, None

    try:
        df_train_raw = pd.read_csv(train_path)
        df_test_raw = pd.read_csv(test_path)
        print(f"Dati raw caricati: Train={df_train_raw.shape}, Test={df_test_raw.shape}")

        # Rimuovi ID (salva test_id se necessario esternamente)
        # test_id = df_test_raw['Id'].copy() # Rimosso da qui, può essere gestito esternamente se serve
        df_train_raw = df_train_raw.drop('Id', axis=1)
        df_test_raw = df_test_raw.drop('Id', axis=1)
        print("Colonna 'Id' rimossa dai dataframe raw.")

        return df_train_raw, df_test_raw

    except Exception as e:
        print(f"Errore durante il caricamento dei dati raw: {e}")
        return None, None

if __name__ == '__main__':
    # Test della funzione
    TRAIN_CSV = '../data/raw/train.csv'
    TEST_CSV = '../data/raw/test.csv'
    df_train, df_test = load_raw_data(TRAIN_CSV, TEST_CSV)
    if df_train is not None:
        print("\nTest load_raw_data completato con successo.")
        print("Head df_train_raw:\n", df_train.head())
    else:
        print("\nTest load_raw_data fallito.")