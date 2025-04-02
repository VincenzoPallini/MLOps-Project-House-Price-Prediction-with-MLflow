# Dockerfile

# 1. Scegli un'immagine base con Conda preinstallato
FROM continuumio/miniconda3:latest

# 2. Imposta la directory di lavoro all'interno del container
WORKDIR /app

# 3. Copia i file di definizione dell'ambiente
COPY conda.yaml .

# 4. Installa Git (necessario a MLflow per loggare il commit) e DVC
#    Aggiorna Conda e installa l'ambiente dal file YAML
#    Pulizia cache Conda per ridurre dimensione immagine
RUN conda update -n base -c defaults conda -y && \
    conda install git -y && \
    conda env create -f conda.yaml && \
    conda clean -afy

# 5. Imposta la shell per usare l'ambiente Conda attivato per i comandi successivi
#    Il nome dell'ambiente è 'mlops-houseprices' come definito in conda.yaml
SHELL ["conda", "run", "-n", "mlops-houseprices", "/bin/bash", "-c"]

# 6. Copia tutto il codice sorgente del progetto nella directory di lavoro /app
#    (Rispetta le regole in .dockerignore)
COPY . /app

# 7. (Opzionale) Esponi la porta se dovessi eseguire un servizio (es. mlflow ui)
#    EXPOSE 5000

# 8. (Opzionale) Comando di default all'avvio del container
#    Potrebbe essere per lanciare mlflow run o un'API
# ENTRYPOINT ["python", "src/train.py"] # Esempio
# Per ora lo lasciamo commentato per eseguire comandi manualmente

# Verifica che l'ambiente sia attivo e le librerie disponibili
RUN echo "Ambiente Conda 'mlops-houseprices' creato e attivato." && \
    python --version && \
    conda list mlflow && \
    dvc --version