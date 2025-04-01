import mlflow
import pandas as pd 
import numpy as np  

run_id = "f43e1c7ea0644c178211b2461bc474aa"

artifact_path = "ridge-model-pipeline"


model_uri = f"runs:/{run_id}/{artifact_path}"


print(f"Caricamento modello sklearn da: {model_uri}")
sklearn_model = mlflow.sklearn.load_model(model_uri)
print("Modello sklearn caricato:", type(sklearn_model))

print(f"\nCaricamento modello pyfunc da: {model_uri}")
pyfunc_model = mlflow.pyfunc.load_model(model_uri)
print("Modello pyfunc caricato:", type(pyfunc_model))


try:
    from src.preprocess import preprocess_data  
    _, X_test_proc_unscaled, _ = preprocess_data("data/raw/train.csv", "data/raw/test.csv") 
    sample_data = X_test_proc_unscaled.head()

    print("\nDati di esempio (prima dello scaling interno alla pipeline):\n", sample_data)

    predictions = pyfunc_model.predict(sample_data)
    print("\nPredizioni (scala logaritmica):\n", predictions)


    original_scale_predictions = np.expm1(predictions)
    print("\nPredizioni (scala originale approssimata):\n", original_scale_predictions)

except Exception as e:
    print(f"\nErrore durante la creazione dati di esempio o predizione: {e}")
    print("Potrebbe essere necessario aggiustare il caricamento dati per l'esempio.")