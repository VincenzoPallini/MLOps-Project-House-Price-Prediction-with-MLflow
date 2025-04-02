# src/evaluate.py
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_regression_metrics(y_true, y_pred):
    """
    Calcola le metriche di regressione (RMSE, MAE, R2) sulla scala fornita.
    Restituisce un dizionario di metriche.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }
    print(f"  Metriche calcolate: RMSE={rmse:.5f}, MAE={mae:.5f}, R2={r2:.5f}")
    return metrics

def calculate_metrics_on_original_scale(y_true_log, y_pred_log):
    """
    Riconverte le predizioni e i valori veri alla scala originale
    e calcola le metriche di regressione.
    Restituisce un dizionario di metriche con suffisso '_orig'.
    """
    try:
        y_true_orig = np.expm1(y_true_log)
        y_pred_orig = np.expm1(y_pred_log)
        orig_metrics = calculate_regression_metrics(y_true_orig, y_pred_orig)
        # Aggiungi suffisso per distinguerle
        orig_metrics_suffixed = {f"{k}_orig": v for k, v in orig_metrics.items()}
        print("  Metriche calcolate su scala originale.")
        return orig_metrics_suffixed
    except Exception as e:
        print(f"Errore nel calcolo metriche su scala originale: {e}")
        return {}

if __name__ == '__main__':
    # Test funzioni
    y_t = np.log1p(np.array([200000, 250000, 180000]))
    y_p = np.log1p(np.array([210000, 245000, 190000]))
    print("Test calculate_regression_metrics (log scale):")
    log_metrics = calculate_regression_metrics(y_t, y_p)
    print(log_metrics)
    print("\nTest calculate_metrics_on_original_scale:")
    orig_metrics = calculate_metrics_on_original_scale(y_t, y_p)
    print(orig_metrics)