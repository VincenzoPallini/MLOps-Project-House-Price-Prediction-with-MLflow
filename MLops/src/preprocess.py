# src/preprocess.py
import pandas as pd
import numpy as np
from scipy.stats import skew


def preprocess_data(train_path, test_path):
    """
    Carica i dati raw, applica preprocessing e feature engineering.
    Restituisce i dataframe delle feature processate (train e test)
    e la variabile target log-trasformata (y_train_log).
    Lo scaling NON viene applicato qui.
    """
    print("--- Esecuzione preprocess_data ---")

    try:
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
        print(f"Dati caricati: Train={df_train.shape}, Test={df_test.shape}")
    except FileNotFoundError:
        print(f"Errore: File non trovati in {train_path} o {test_path}")
        return None, None, None

    test_id = df_test['Id'].copy()
    df_train = df_train.drop('Id', axis=1)
    df_test = df_test.drop('Id', axis=1)
    print("Colonna 'Id' rimossa dai dataframe.")

    print(f"Dimensione Train prima rimozione outlier: {df_train.shape}")
    outlier_index = df_train[(df_train['GrLivArea'] > 4000) & (df_train['SalePrice'] < 300000)].index
    df_train = df_train.drop(outlier_index)
    df_train.reset_index(drop=True, inplace=True)
    print(f"Dimensione Train dopo rimozione outlier GrLivArea: {df_train.shape}")

    y_train_log = np.log1p(df_train['SalePrice'].copy()) 
    df_train_features = df_train.drop('SalePrice', axis=1)
    print(f"Separato e trasformato y_train_log ({y_train_log.shape[0]} elementi).")
    print(f"Dimensioni df_train_features: {df_train_features.shape}")

    ntrain = df_train_features.shape[0]
    all_data = pd.concat((df_train_features, df_test)).reset_index(drop=True)
    print(f"Dataset combinato (all_data - solo features) dimensioni: {all_data.shape}")

    print("Gestione Valori Mancanti...")
    cols_fillna_none = [
        'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType',
        'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond',
        'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType'
    ]
    for col in cols_fillna_none:
        if col in all_data.columns: all_data[col] = all_data[col].fillna('None')

    cols_fillna_zero = [
        'GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
        'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'
    ]
    for col in cols_fillna_zero:
        if col in all_data.columns: all_data[col] = all_data[col].fillna(0)

    if 'LotFrontage' in all_data.columns:
        all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(
            lambda x: x.fillna(x.median()))
        all_data['LotFrontage'] = all_data['LotFrontage'].fillna(all_data['LotFrontage'].median())

    cols_fillna_mode = ['MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType', 'Functional', 'Utilities']
    for col in cols_fillna_mode:
        if col in all_data.columns:
            mode_val = all_data[col].mode()[0]
            all_data[col] = all_data[col].fillna(mode_val)

    missing_after = all_data.isnull().sum().sum()
    if missing_after == 0:
        print("Nessun valore mancante rimasto in all_data.")
    else:
        print(f"ATTENZIONE: Rimangono {missing_after} valori mancanti!")
        print(all_data.isnull().sum()[all_data.isnull().sum() > 0])

    print("Creazione Nuove Feature...")
    all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
    all_data['HouseAge'] = all_data['YrSold'] - all_data['YearBuilt']
    all_data['RemodAge'] = all_data['YrSold'] - all_data['YearRemodAdd']
    all_data['HouseAge'] = all_data['HouseAge'].apply(lambda x: max(x, 0))
    all_data['RemodAge'] = all_data['RemodAge'].apply(lambda x: max(x, 0))
    all_data['IsRemodeled'] = (all_data['YearRemodAdd'] != all_data['YearBuilt']).astype(int)
    all_data['TotalBath'] = all_data['FullBath'] + 0.5 * all_data['HalfBath'] + \
                           all_data['BsmtFullBath'] + 0.5 * all_data['BsmtHalfBath']
    all_data['TotalPorchSF'] = all_data['OpenPorchSF'] + all_data['EnclosedPorch'] + \
                              all_data['3SsnPorch'] + all_data['ScreenPorch']

    print("Correzione Skewness Feature Numeriche...")
    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
    numeric_feats = numeric_feats.drop(['YrSold', 'MoSold', 'YearBuilt', 'YearRemodAdd', 'GarageYrBlt'], errors='ignore')
    skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    skewness_df = pd.DataFrame({'Skew': skewed_feats})
    high_skew = skewness_df[abs(skewness_df['Skew']) > 0.75].index
    print(f"Applicazione log1p a {len(high_skew)} feature numeriche con skewness > 0.75.")
    for feat in high_skew:
        if feat in all_data.columns:
            all_data[feat] = np.log1p(all_data[feat])

    print("Encoding Feature Categoriche...")
    if 'MSSubClass' in all_data.columns:
        all_data['MSSubClass'] = all_data['MSSubClass'].astype(str)

    all_data = pd.get_dummies(all_data)
    print(f"Dimensioni dataset dopo get_dummies: {all_data.shape}")

    train_features_processed = all_data[:ntrain]
    test_features_processed = all_data[ntrain:]

    print(f"Dataset separati nuovamente:")
    print(f"Train processato (solo features): {train_features_processed.shape}")
    print(f"Test processato (solo features): {test_features_processed.shape}")
    if train_features_processed.shape[0] != len(y_train_log):
         print(f"ATTENZIONE: Numero righe train_features_processed ({train_features_processed.shape[0]})"
               f" non corrisponde a lunghezza y_train_log ({len(y_train_log)})!")

    print("--- Fine preprocess_data ---")
    return train_features_processed, test_features_processed, y_train_log

if __name__ == "__main__":

    TRAIN_CSV = '../data/raw/train.csv'
    TEST_CSV = '../data/raw/test.csv'

    print("Testing preprocess_data function...")
    X_train_proc, X_test_proc, y_train_proc = preprocess_data(TRAIN_CSV, TEST_CSV)

    if X_train_proc is not None:
        print("\nOutput del test:")
        print("X_train_proc shape:", X_train_proc.shape)
        print("X_test_proc shape:", X_test_proc.shape)
        print("y_train_proc shape:", y_train_proc.shape)
        print("\nX_train_proc head:\n", X_train_proc.head())
        print("\ny_train_proc head:\n", y_train_proc.head())
    else:
        print("Test fallito: dati non caricati.")