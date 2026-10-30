# Observatorio de Inteligencia Turística (CC3084 Data Science, UVG 2026-II)

Tablero interactivo y flujo MLOps sobre el **ingreso mensual de viajeros internacionales a Guatemala** (ene-2009 a
jun-2026). El **Laboratorio 10** construyó el tablero para directivos; el **Laboratorio 11** agrega un flujo
reproducible que estima el ingreso de viajeros al destino asignado al equipo, **Petén / Mundo Maya**, y publica una
versión aprobada para que el tablero la consulte. Ambos reutilizan la limpieza, las series y los modelos de los
Laboratorios 1 (series de tiempo) y 2 (LSTM).

*Datos de uso exclusivamente académico; no son cifras oficiales del INGUAT ni del IGM.*

## Estructura del repositorio

```
├── data/
│   ├── raw/            # Base_Migracion_2009-2026jun.xlsx  (inmutable)
│   └── processed/      # migracion_limpia.csv  (generado, no versionado)
├── 01_limpieza_eda/          # src/ de limpieza (Lab. 1)
├── 02_series_tiempo/         # src/ de construcción de series (Lab. 1)
├── 03_modelado_prediccion/   # src/ de los modelos clásicos (Lab. 1)
├── 04_deep_learning_lstm/    # src/ del LSTM (Lab. 2)
├── 05_dashboard_interactivo/ # tablero Streamlit + Plotly (Lab. 10)  ← ver su README
│   ├── src/                  # app.py, comun.py, app_pages/, preparar_datos.py
│   ├── notebooks/
│   ├── data/resultados/      # predicciones y métricas precalculadas
│   ├── tests/
│   ├── requirements.txt
│   ├── codebook.md
│   └── README.md
├── 06_mlops/                 # flujo reproducible + controles de calidad (Lab. 11)  ← ver su README
│   ├── config/pipeline.yaml  # rutas, períodos, parámetros y umbrales
│   ├── src/                  # pipeline.py y las etapas del flujo
│   ├── tests/
│   ├── data/                 # expedientes de ejecución y versiones aprobadas
│   ├── logs/                 # bitácora por ejecución
│   ├── requirements.txt
│   ├── codebook.md
│   └── README.md
├── reports/
│   ├── Documento de diseño Laboratorio 10.pdf
│   ├── Avance Laboratorio 11.docx
│   └── figuras/              # diagrama del flujo del Lab. 11
├── docs/                     # enunciados de los Laboratorios 10 y 11
└── README.md
```

Las etapas 01–04 contienen solo el código de los Labs 1 y 2 que el tablero reutiliza; sus cuadernos e informes
están en las ramas `Lab1-SeriesdeTiempo` y `Lab2-ST-LSTM`. Cada etapa sigue la convención del repositorio de referencia del curso: `src/`, `notebooks/`, `data/`, un
`requirements.txt`, un `README.md` y un `codebook.md`.

## Cómo correr el tablero

```bash
cd 05_dashboard_interactivo
pip install -r requirements.txt
streamlit run src/app.py
```

Los datos que usa el tablero ya vienen precalculados en `05_dashboard_interactivo/data/resultados/`. Para regenerarlos,
ver `05_dashboard_interactivo/README.md`.

## Cómo correr el flujo MLOps

```bash
pip install -r 06_mlops/requirements.txt
python 06_mlops/src/pipeline.py --config 06_mlops/config/pipeline.yaml
python -m pytest 06_mlops/tests -q
```

Un solo comando ejecuta las siete etapas: lectura, validación, preparación, ajuste, evaluación,
almacenamiento y publicación de la versión aprobada. Ver `06_mlops/README.md` para el alcance, el
diagrama y los seis controles de calidad.

## Equipo

Daniel Chet (231177) · Dulce Ambrosio (231143) · Cristian Tunchez (231359) · Javier Linares (231135)
