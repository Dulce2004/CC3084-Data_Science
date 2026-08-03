# 04 — Deep Learning: Modelos LSTM (Laboratorio 2)

Contiene el **Ejercicio 1** del Laboratorio 2: al menos 2 modelos LSTM (con tuneo de
hiperparámetros) por serie, para **Total mensual** y **Vía Aérea**, usando los mismos conjuntos de
entrenamiento/prueba (split temporal 70/30) construidos en el Laboratorio 1
(`02_series_tiempo/src/series.py`).

## Contenido

- `src/lstm_utils.py` — utilidades reutilizables:
  - `preparar_datos_lstm`: transformación log1p + escalado min-max (ajustado solo con
    entrenamiento) + ventanas supervisadas.
  - `construir_modelo_lstm`: arquitectura LSTM configurable (nº de capas, unidades, dropout,
    bidireccional, learning rate).
  - `entrenar_modelo`: entrenamiento con `EarlyStopping`.
  - `pronostico_recursivo`: pronóstico multi-paso *walk-forward* sobre el conjunto de prueba
    (el modelo no ve los valores reales de prueba, igual que ARIMA/Prophet/HW en el Lab. 1).
  - `tunear_lstm`: entrena y evalúa una lista de configuraciones, devuelve tabla ordenada por RMSE.
- `notebooks/04_lstm_modelos.ipynb` — cuaderno del Ejercicio 1: selección de series, preparación de
  datos, tuneo (4 configuraciones por serie), selección del mejor modelo, predicción, y comparación
  contra los mejores modelos clásicos del Laboratorio 1 (recalculados aquí para tener cifras
  exactas y comparables).

## Cómo reproducir

```bash
pip install -r 04_deep_learning_lstm/requirements.txt
jupyter nbconvert --to notebook --execute --inplace 04_deep_learning_lstm/notebooks/04_lstm_modelos.ipynb
```

Requiere que `data/processed/migracion_limpia.csv` ya exista (generarlo con
`python 01_limpieza_eda/src/limpieza.py` si hace falta).

## Resultados (resumen)

| Serie | Mejor modelo Lab. 1 | RMSE Lab. 1 | Mejor LSTM (este lab.) | RMSE LSTM |
|---|---|---|---|---|
| Total mensual | Prophet | 107,270 | LSTM-B (2 capas 64→32, dropout 0.2, look_back 12) | 181,007 |
| Vía Aérea | SES | 41,350 | LSTM-D (2 capas 32→16, dropout 0.1, look_back 18, lr 5e-4) | 55,424 |

En ambas series, los modelos clásicos del Laboratorio 1 siguen superando a los LSTM tuneados aquí
(ver la discusión completa del ejercicio 1.4 en el notebook). Pendiente para la entrega final:
Ejercicio 2 (catch22) y la comparación de algoritmos consolidada.
