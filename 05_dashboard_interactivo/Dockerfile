# Imagen del tablero. Lleva su propio data/resultados precalculado (Lab. 10, se
# versiona a propósito); la versión aprobada del destino (Lab. 11) se monta como
# volumen desde 06_mlops/data/aprobado, no se hornea en la imagen.
FROM python:3.12-slim

WORKDIR /app

COPY 05_dashboard_interactivo/requirements.txt 05_dashboard_interactivo/requirements.txt
RUN pip install --no-cache-dir -r 05_dashboard_interactivo/requirements.txt

COPY 05_dashboard_interactivo 05_dashboard_interactivo

WORKDIR /app/05_dashboard_interactivo
EXPOSE 8501
CMD ["streamlit", "run", "src/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
