# Laboratorio 10 — Visualización interactiva (CC3084 Data Science, UVG 2026-II)

Tablero interactivo del **ingreso mensual de viajeros internacionales a Guatemala** (ene-2009 a jun-2026) para el
observatorio de turismo. Reutiliza la limpieza, las series y los modelos de los Laboratorios 1 (series de tiempo) y
2 (LSTM). *Datos de uso exclusivamente académico; no son cifras oficiales del INGUAT ni del IGM.*

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
├── reports/
│   └── Documento de diseño Laboratorio 10.pdf
├── docs/                     # enunciado del Laboratorio 10
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

## Equipo

Daniel Chet (231177) · Dulce Ambrosio (231143) · Cristian Tunchez (231359) · Javier Linares (231135)
