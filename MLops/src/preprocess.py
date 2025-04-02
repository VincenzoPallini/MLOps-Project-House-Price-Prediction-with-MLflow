# src/preprocess.py (Corretto)
import pandas as pd
import numpy as np
from scipy.stats import skew
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def preprocess_data(df_train_raw, df_test_raw, processed_data_dir="../data/processed"):
    """
    Applica preprocessing e feature engineering ai DataFrame raw.
    Salva i risultati processati in formato Parquet e restituisce i
    DataFrame delle feature processate (train e test) e la variabile
    target log-trasformata (y_train_log).
    """
    if df_train_raw is None or df_test_raw is None:
        print("Errore: DataFrame di input mancanti in preprocess_data.")
        return None, None, None

    print("--- Esecuzione preprocess_data (Corretto) ---")

    # --- 1. Gestione Outlier (su df_train_raw) ---
    print(f"Dimensione Train raw prima rimozione outlier: {df_train_raw.shape}")
    outlier_index = df_train_raw[(df_train_raw['GrLivArea'] > 4000) & (df_train_raw['SalePrice'] < 300000)].index
    if not outlier_index.empty:
        df_train_raw = df_train_raw.drop(outlier_index)
        df_train_raw.reset_index(drop=True, inplace=True)
        print(f"Dimensione Train raw dopo rimozione outlier GrLivArea: {df_train_raw.shape}")
    else:
        print("Nessun outlier GrLivArea/SalePrice trovato da rimuovere.")

    # --- 2. Separare y_train e Trasformazione Log ---
    if 'SalePrice' not in df_train_raw.columns:
        print("Errore: Colonna 'SalePrice' non trovata nel DataFrame di training.")
        return None, None, None
    y_train_log = np.log1p(df_train_raw['SalePrice'].copy())
    df_train_features = df_train_raw.drop('SalePrice', axis=1)
    print(f"Separato e trasformato y_train_log ({y_train_log.shape[0]} elementi).")

    # --- 3. Combinare Feature per preprocessing coerente ---
    ntrain = df_train_features.shape[0]
    all_data = pd.concat((df_train_features, df_test_raw)).reset_index(drop=True) # Usa df_test_raw
    print(f"Dataset combinato (all_data - solo features) dimensioni: {all_data.shape}")

    # --- 4. Gestione Valori Mancanti (NaN) ---
    print("Gestione Valori Mancanti...")
    # Feature dove NaN significa 'None'
    cols_fillna_none = [
        'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType',
        'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond',
        'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType'
    ]
    for col in cols_fillna_none:
        if col in all_data.columns: all_data[col] = all_data[col].fillna('None')

    # Feature numeriche dove NaN probabilmente significa 0
    cols_fillna_zero = [
        'GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
        'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'
    ]
    for col in cols_fillna_zero:
         if col in all_data.columns: all_data[col] = all_data[col].fillna(0)

    # LotFrontage: Imputazione con la mediana per quartiere
    if 'LotFrontage' in all_data.columns:
        all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(
            lambda x: x.fillna(x.median()))
        if all_data['LotFrontage'].isnull().any():
             all_data['LotFrontage'] = all_data['LotFrontage'].fillna(all_data['LotFrontage'].median())
        print(" - Imputato 'LotFrontage'.")

    # Feature categoriche rimanenti: Imputazione con la moda
    cols_fillna_mode = ['MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st',
                        'Exterior2nd', 'SaleType', 'Functional', 'Utilities']
    for col in cols_fillna_mode:
        if col in all_data.columns:
            if all_data[col].isnull().any():
                mode_val = all_data[col].mode()[0]
                all_data[col] = all_data[col].fillna(mode_val)

    # Verifica finale NaN
    missing_after = all_data.isnull().sum().sum()
    if missing_after == 0:
        print("Nessun valore mancante rimasto in all_data.")
    else:
        print(f"ATTENZIONE: Rimangono {missing_after} valori mancanti!")
        print(all_data.isnull().sum()[all_data.isnull().sum() > 0])

    # --- 5. Feature Engineering (con controlli corretti)---
    print("Creazione Nuove Feature...")
    # Combinazione Superfici Totali
    required_sf_cols = ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']
    if all(col in all_data.columns for col in required_sf_cols):
        all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
    else: print(f"Warning: Mancano una o più colonne per TotalSF: {required_sf_cols}")

    # Età della casa e ristrutturazione
    required_yr_cols = ['YrSold', 'YearBuilt', 'YearRemodAdd']
    if all(col in all_data.columns for col in required_yr_cols):
        all_data['HouseAge'] = all_data['YrSold'] - all_data['YearBuilt']
        all_data['RemodAge'] = all_data['YrSold'] - all_data['YearRemodAdd']
        all_data['HouseAge'] = all_data['HouseAge'].apply(lambda x: max(x, 0))
        all_data['RemodAge'] = all_data['RemodAge'].apply(lambda x: max(x, 0))
        all_data['IsRemodeled'] = (all_data['YearRemodAdd'] != all_data['YearBuilt']).astype(int)
    else: print(f"Warning: Mancano una o più colonne per HouseAge/RemodAge: {required_yr_cols}")

    # Numero totale bagni
    required_bath_cols = ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']
    if all(col in all_data.columns for col in required_bath_cols):
        all_data['TotalBath'] = all_data['FullBath'] + 0.5 * all_data['HalfBath'] + \
                               all_data['BsmtFullBath'] + 0.5 * all_data['BsmtHalfBath']
    else: print(f"Warning: Mancano una o più colonne per TotalBath: {required_bath_cols}")

    # Superficie totale portici
    required_porch_cols = ['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
    if all(col in all_data.columns for col in required_porch_cols):
        all_data['TotalPorchSF'] = all_data['OpenPorchSF'] + all_data['EnclosedPorch'] + \
                                  all_data['3SsnPorch'] + all_data['ScreenPorch']
    else: print(f"Warning: Mancano una o più colonne per TotalPorchSF: {required_porch_cols}")
    print(" - Create feature ingegnerizzate.")

    # --- 6. Gestione della Skewness ---
    print("Correzione Skewness Feature Numeriche...")
    numeric_feats = all_data.select_dtypes(include=np.number).columns
    potential_exclude = ['YrSold', 'MoSold', 'YearBuilt', 'YearRemodAdd', 'GarageYrBlt',
                         'OverallQual', 'OverallCond']
    numeric_feats = numeric_feats.drop([col for col in potential_exclude if col in numeric_feats], errors='ignore')

    if not numeric_feats.empty:
        skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
        skewness_df = pd.DataFrame({'Skew': skewed_feats})
        high_skew = skewness_df[abs(skewness_df['Skew']) > 0.75].index
        print(f"Applicazione log1p a {len(high_skew)} feature numeriche con skewness > 0.75.")
        for feat in high_skew:
            if feat in all_data.columns:
                all_data[feat] = np.log1p(all_data[feat])
    else:
        print("Nessuna feature numerica selezionata per la correzione della skewness.")


    # --- 7. Conversione Tipi (MSSubClass) e Encoding ---
    print("Encoding Feature Categoriche...")
    if 'MSSubClass' in all_data.columns:
        all_data['MSSubClass'] = all_data['MSSubClass'].astype(str)
        print(" - Convertito 'MSSubClass' a stringa.")

    all_data = pd.get_dummies(all_data)
    print(f"Dimensioni dataset dopo get_dummies: {all_data.shape}")

    # --- 8. Separare nuovamente Train e Test ---
    if ntrain > all_data.shape[0]:
         print(f"Errore: ntrain ({ntrain}) è maggiore delle righe in all_data ({all_data.shape[0]})")
         return None, None, None
    train_features_processed = all_data[:ntrain]
    test_features_processed = all_data[ntrain:]
    print(f"Dataset separati nuovamente: Train={train_features_processed.shape}, Test={test_features_processed.shape}")
    if train_features_processed.shape[0] != len(y_train_log):
         print(f"ATTENZIONE: Righe df_train_processed ({train_features_processed.shape[0]}) != lunghezza y_train_log ({len(y_train_log)})!")

    # --- 9. Salvataggio Dati Processati (con correzione per y_train_log) ---
    print(f"Salvataggio dati processati in {processed_data_dir}...")
    os.makedirs(processed_data_dir, exist_ok=True)
    try:
        train_features_processed.to_parquet(os.path.join(processed_data_dir, "train_features_processed.parquet"))
        test_features_processed.to_parquet(os.path.join(processed_data_dir, "test_features_processed.parquet"))
        # Correzione: Converti Series in DataFrame prima di salvare
        pd.DataFrame(y_train_log, columns=["SalePrice_log"]).to_parquet(os.path.join(processed_data_dir, "y_train_log.parquet"))
        print("Dati processati salvati come file Parquet.")
    except Exception as e:
        print(f"Errore durante il salvataggio dei dati processati: {e}")

    print("--- Fine preprocess_data (Corretto) ---")
    return train_features_processed, test_features_processed, y_train_log