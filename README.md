# Laboratorio 2 — Deep Learning (CC3084 Data Science, UVG 2026-II)

Continuación del **Laboratorio 1 — Series de Tiempo** sobre el ingreso mensual de viajeros
internacionales a Guatemala (ene-2009 a jun-2026). Este laboratorio construye modelos **LSTM**
para dos de las series ya trabajadas, y (en la entrega final) explora la similitud entre todas las
series construidas mediante **catch22**.

*Datos de uso exclusivamente académico; no son cifras oficiales del INGUAT ni del IGM.*

## Estructura del repositorio

```
Lab2/
├── data/
│   ├── raw/            # Base_Migracion_2009-2026jun.xlsx  (idéntico al Laboratorio 1, inmutable)
│   └── processed/      # migracion_limpia.csv  (generado, no versionado)
├── 01_limpieza_eda/     # (heredado del Laboratorio 1) limpieza.py, config.py, 01_eda.ipynb
├── 02_series_tiempo/    # (heredado del Laboratorio 1) series.py — construcción de series y split 70/30
├── 03_modelado_prediccion/  # (heredado del Laboratorio 1) modelado.py — ARIMA/SARIMA/Prophet/HW/SES,
│                            #   se reutiliza aquí solo para recalcular las tablas de referencia
├── 04_deep_learning_lstm/   # NUEVO — Ejercicio 1 del Laboratorio 2 (LSTM + tuneo)
│   ├── src/lstm_utils.py
│   ├── notebooks/04_lstm_modelos.ipynb
│   ├── requirements.txt
│   └── README.md
├── reports/
│   └── figuras/         # figuras exportadas (incluye las nuevas lstm_*.png)
├── docs/                # enunciado del Laboratorio 2 (PDF)
└── README.md
```

Las carpetas `01_limpieza_eda`, `02_series_tiempo` y `03_modelado_prediccion` son las mismas del
Laboratorio 1 (se copiaron sin cambios) porque el enunciado pide **usar los mismos conjuntos de
entrenamiento y prueba**; `04_deep_learning_lstm` es el trabajo nuevo de este laboratorio.

## Estado de entrega

- **(AVANCE, 30/07/2026 17:20) ✅ Completado:** Ejercicio 1 — al menos 2 modelos LSTM con tuneo de
  parámetros, para 2 series (`Total mensual` y `Vía Aérea`), con comparación contra el Laboratorio 1.
- **(ENTREGA FINAL, 02/08/2026) ⏳ Pendiente:** Ejercicio 2 completo (extracción de características
  catch22 para las 8 series, matriz estandarizada, PCA, clustering, heatmap, correlaciones, mapa de
  distancias, y las preguntas de análisis 2.6–2.14, incluyendo un modelo LSTM adicional con las
  variables de catch22).

## Cómo reproducir

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r 01_limpieza_eda/requirements.txt -r 03_modelado_prediccion/requirements.txt \
            -r 04_deep_learning_lstm/requirements.txt

# 1) Generar el dataset limpio (idéntico al Laboratorio 1)
python 01_limpieza_eda/src/limpieza.py

# 2) Ejecutar el cuaderno del Ejercicio 1 (LSTM)
jupyter nbconvert --to notebook --execute --inplace 04_deep_learning_lstm/notebooks/04_lstm_modelos.ipynb
```
