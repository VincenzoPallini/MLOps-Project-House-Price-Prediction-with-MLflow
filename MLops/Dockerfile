FROM continuumio/miniconda3:latest

WORKDIR /app

COPY conda.yaml .


RUN conda update -n base -c defaults conda -y && \
    conda install git -y && \
    conda env create -f conda.yaml && \
    conda clean -afy


SHELL ["conda", "run", "-n", "mlops-houseprices", "/bin/bash", "-c"]


COPY . /app

RUN echo "Ambiente Conda 'mlops-houseprices' creato e attivato." && \
    python --version && \
    conda list mlflow && \
    dvc --version