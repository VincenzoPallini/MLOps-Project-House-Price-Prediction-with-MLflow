# tests/test_preprocess.py
import pytest
import pandas as pd
import os

# Assumi che pytest venga eseguito dalla radice del progetto
# Altrimenti, potresti dover aggiustare l'import o il sys.path
try:
    from src.preprocess import preprocess_data
except ImportError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from src.preprocess import preprocess_data


# Definisci percorsi relativi alla radice del progetto
# Questi test ASSUMONO che i dati siano presenti (via dvc pull)
# Potrebbero fallire in un ambiente CI pulito senza dvc pull
TRAIN_PATH = "data/raw/train.csv"
TEST_PATH = "data/raw/test.csv"

# Verifica se i file di dati esistono prima di eseguire i test che li usano
# Pytest eseguirà questo check prima di collezionare i test nella classe
@pytest.mark.skipif(not (os.path.exists(TRAIN_PATH) and os.path.exists(TEST_PATH)),
                    reason="Data files not found (run 'dvc pull'?)")
class TestPreprocessDataWithData:

    def test_preprocess_data_runs(self):
        """ Test if preprocess_data runs without throwing errors on actual data """
        try:
            X_train, X_test, y_train = preprocess_data(TRAIN_PATH, TEST_PATH)
            assert X_train is not None
            assert X_test is not None
            assert y_train is not None
        except Exception as e:
            pytest.fail(f"preprocess_data failed with exception: {e}")

    def test_preprocess_data_output_shapes(self):
        """ Test the output shapes (adjust numbers based on previous runs) """
        X_train, X_test, y_train = preprocess_data(TRAIN_PATH, TEST_PATH)

        # Verifica consistenza righe (train può variare leggermente per outlier)
        assert abs(X_train.shape[0] - len(y_train)) == 0
        assert abs(X_train.shape[0] - 1458) <= 5 # Num righe train atteso (circa)
        assert X_test.shape[0] == 1459          # Num righe test fisso

        # Verifica consistenza colonne
        assert X_train.shape[1] == X_test.shape[1]
        # Verifica che il numero di colonne sia quello atteso dopo OHE
        # Sostituisci 322 con il numero effettivo osservato nel tuo output precedente
        expected_cols = 322
        assert X_train.shape[1] == expected_cols

    def test_preprocess_data_no_nan_output(self):
        """ Test that the processed dataframes have no NaN values """
        X_train, X_test, y_train = preprocess_data(TRAIN_PATH, TEST_PATH)
        assert X_train.isnull().sum().sum() == 0
        assert X_test.isnull().sum().sum() == 0
        assert y_train.isnull().sum() == 0

# Aggiungi qui altri test che NON richiedono i dati reali, se possibile
# Esempio: Testare una funzione helper isolata se ne avessi una in preprocess.py