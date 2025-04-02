import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from prefect import task 
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

@task 
def calculate_regression_metrics(y_true, y_pred):
    """
    Calcola le metriche di regressione (RMSE, MAE, R2) sulla scala fornita.
    Restituisce un dizionario di metriche.
    """
    print("--- Esecuzione calculate_regression_metrics TASK ---") # Aggiorna print
    try:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        metrics = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }
        print(f"  Metriche (scala attuale): RMSE={rmse:.5f}, MAE={mae:.5f}, R2={r2:.5f}")
        return metrics
    except Exception as e:
        print(f"Errore nel calcolo metriche: {e}")
        raise e


@task 
def calculate_metrics_on_original_scale(y_true_log, y_pred_log):
    """
    Riconverte le predizioni e i valori veri alla scala originale
    e calcola le metriche di regressione.
    Restituisce un dizionario di metriche con suffisso '_orig'.
    """
    print("--- Esecuzione calculate_metrics_on_original_scale TASK ---") 
    try:
        y_true_orig = np.expm1(y_true_log)
        y_pred_orig = np.expm1(y_pred_log)

        rmse_orig = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
        mae_orig = mean_absolute_error(y_true_orig, y_pred_orig)
        r2_orig = r2_score(y_true_orig, y_pred_orig)

        orig_metrics_suffixed = {
             "rmse_orig": rmse_orig,
             "mae_orig": mae_orig,
             "r2_orig": r2_orig
        }
        print(f"  Metriche (scala originale): RMSE={rmse_orig:.2f}, MAE={mae_orig:.2f}, R2={r2_orig:.5f}")
        return orig_metrics_suffixed
    except Exception as e:
        print(f"Errore nel calcolo metriche su scala originale: {e}")

        return {}

