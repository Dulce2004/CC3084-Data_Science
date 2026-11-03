# Imagen del flujo MLOps (pipeline + monitoreo). No incluye datos institucionales:
# se montan como volumen en tiempo de ejecución (ver docker-compose.yml).
FROM python:3.12-slim

WORKDIR /app

COPY 06_mlops/requirements.txt 06_mlops/requirements.txt
RUN pip install --no-cache-dir -r 06_mlops/requirements.txt

# Solo lo que el pipeline reutiliza de los Laboratorios 1 y 2 (ver 06_mlops/src/modelo.py).
COPY 01_limpieza_eda/src 01_limpieza_eda/src
COPY 03_modelado_prediccion/src 03_modelado_prediccion/src
COPY 06_mlops 06_mlops

CMD ["python", "06_mlops/src/pipeline.py", "--config", "06_mlops/config/pipeline.yaml"]
