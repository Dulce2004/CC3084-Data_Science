# Laboratorio 2 — Deep Learning (CC3084 Data Science, UVG 2026-II)

Modelos LSTM y análisis de similitud entre series con *catch22*, sobre el **ingreso mensual de
viajeros internacionales a Guatemala** (ene-2009 a jun-2026). *Datos de uso exclusivamente
académico; no son cifras oficiales del INGUAT ni del IGM.*

Este repositorio reutiliza el código de limpieza, construcción de series y modelado clásico
desarrollado para el Laboratorio 1 (carpetas `01`-`03`), porque el Laboratorio 2 pide explícitamente
usar los mismos conjuntos de entrenamiento/prueba y comparar contra esos modelos. Los entregables
propios del Laboratorio 1 (informe, notebooks de EDA/series/modelado) ya no forman parte del
repositorio.

## Estructura del repositorio

```
├── data/
│   ├── raw/            # Base_Migracion_2009-2026jun.xlsx  (inmutable)
│   └── processed/      # migracion_limpia.csv  (generado, no versionado)
├── 01_limpieza_eda/
│   ├── src/            # config.py, limpieza.py — usados por 04_deep_learning_lstm
│   ├── requirements.txt
│   └── README.md
├── 02_series_tiempo/
│   ├── src/            # series.py (construcción de series, split, ADF) — usado por 04_deep_learning_lstm
│   ├── requirements.txt
│   └── README.md
├── 03_modelado_prediccion/
│   ├── src/            # modelado.py (ARIMA/SARIMA, Prophet, Holt-Winters, SES, seasonal naive)
│   ├── requirements.txt
│   └── README.md
├── 04_deep_learning_lstm/
│   ├── src/            # lstm_utils.py (LSTM, tuneo, LSTM+catch22), catch22_utils.py (22 features)
│   ├── notebooks/      # 04_lstm_modelos.ipynb  (Laboratorio 2 completo: ejercicios 1 y 2)
│   ├── requirements.txt
│   └── README.md
├── reports/
│   ├── figuras/         # figuras exportadas para el informe del Lab. 2
│   └── informe_laboratorio2.pdf   # informe final (sin código), para entregar
├── docs/               # enunciado del laboratorio (PDF)
└── README.md
```

## Cómo reproducir

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r 04_deep_learning_lstm/requirements.txt

# 1) Generar el dataset limpio
python 01_limpieza_eda/src/limpieza.py

# 2) Ejecutar el cuaderno del Laboratorio 2
jupyter nbconvert --to notebook --execute --inplace 04_deep_learning_lstm/notebooks/04_lstm_modelos.ipynb
```

## Laboratorio 2 — Deep Learning (LSTM y catch22)

Ver `04_deep_learning_lstm/README.md` para el detalle. Resumen: al menos 2 modelos LSTM tuneados por
serie para *Total mensual* y *Vía Aérea* (mismos conjuntos de entrenamiento/prueba del Lab. 1),
comparados contra los mejores modelos clásicos; y un análisis de similitud entre las 8 series del
Lab. 1 con las 22 características de *catch22* (PCA, clustering, heatmap, correlación, distancias).
`pycatch22` no compila en esta máquina (Windows sin MSVC/gcc, sin *wheels* en PyPI), así que las 22
características se reimplementaron en Python puro en `04_deep_learning_lstm/src/catch22_utils.py`.
