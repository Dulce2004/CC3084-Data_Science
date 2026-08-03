# 04 — Deep Learning: LSTM y catch22 (Laboratorio 2)

Laboratorio completo: **Ejercicio 1** (modelos LSTM para dos series del Laboratorio 1, con tuneo de
hiperparámetros y comparación contra los mejores modelos clásicos) y **Ejercicio 2** (similitud
entre las 8 series construidas en el Laboratorio 1 usando las 22 características de *catch22*: PCA,
clustering, heatmap, correlación entre características, mapa de distancias, y un LSTM adicional que
incorpora esas características).

## Contenido

- `src/lstm_utils.py` — utilidades reutilizables:
  - `preparar_datos_lstm`: transformación log1p + escalado min-max (ajustado solo con
    entrenamiento) + ventanas supervisadas.
  - `construir_modelo_lstm` / `entrenar_modelo` / `pronostico_recursivo` / `tunear_lstm`: LSTM
    "puro" (ejercicio 1), con pronóstico *walk-forward* (igual que ARIMA/Prophet/HW en el Lab. 1).
  - `construir_modelo_lstm_catch22` / `entrenar_modelo_catch22` / `pronostico_recursivo_catch22`:
    variante de dos ramas (secuencia + vector catch22) para el ejercicio 2.14.
- `src/catch22_utils.py` — reimplementación en Python puro (numpy/scipy) de las 22 características
  de *catch22* (Lubba et al., 2019). **Nota:** el paquete oficial `pycatch22` no tiene *wheels* para
  Windows y requiere compilar una extensión en C (no hay MSVC/gcc instalado en esta máquina), así
  que se reimplementaron las 22 características siguiendo las definiciones publicadas.
- `notebooks/04_lstm_modelos.ipynb` — cuaderno completo: selección de series, preparación de datos,
  tuneo (4 configuraciones por serie), selección del mejor modelo, predicción y comparación contra
  el Laboratorio 1 (ejercicio 1); extracción de catch22 para las 8 series, matriz estandarizada,
  PCA/clustering/heatmap/correlación/distancias, interpretación, y LSTM + catch22 (ejercicio 2).

## Cómo reproducir

```bash
pip install -r 04_deep_learning_lstm/requirements.txt
jupyter nbconvert --to notebook --execute --inplace 04_deep_learning_lstm/notebooks/04_lstm_modelos.ipynb
```

Requiere que `data/processed/migracion_limpia.csv` ya exista (generarlo con
`python 01_limpieza_eda/src/limpieza.py` si hace falta).

## Resultados (resumen)

**Ejercicio 1 — LSTM vs. Laboratorio 1**

| Serie | Mejor modelo Lab. 1 | RMSE Lab. 1 | Mejor LSTM (este lab.) | RMSE LSTM | LSTM vs. Lab.1 |
|---|---|---|---|---|---|
| Total mensual | Prophet | 107,270 | LSTM-A (1 capa, look_back 12) | 162,502 | +51.5% peor |
| Vía Aérea | SES | 41,350 | LSTM-D (2 capas 32→16, look_back 18, lr 5e-4) | 55,424 | +34.0% peor |

En ambas series los modelos clásicos del Laboratorio 1 siguen superando a los LSTM tuneados aquí,
aunque el LSTM es más estable numéricamente que el SARIMA del Lab. 1 en Vía Aérea (ver ejercicio 1.4
en el notebook).

**Ejercicio 2 — catch22**

Sobre las 8 series (Total, 3 vías, 4 tipos de viajero) el clustering jerárquico (Ward, k=3) sobre las
22 características catch22 encuentra tres grupos: (1) Total/Vía Aérea/Vía Terrestre/Tipo Turista —
alto volumen y tendencia fuerte; (2) Vía Marítima/Tipo Cruceristas — caída de pandemia total (-100%)
y volatilidad alta; (3) Tipo Viajero/Tipo Excursionista — bajo volumen, estacionalidad casi nula. Las
etiquetas categóricas del Laboratorio 1 (vía, tipo de viajero) **no** predicen bien estos
agrupamientos: las 3 vías y los 4 tipos quedan repartidos en clusters distintos. El LSTM con
características catch22 (2.14, sobre Total mensual) no mejoró al mejor LSTM "puro" — al ser un
vector fijo por serie, no aporta información que varíe entre ventanas de entrenamiento de una sola
serie. Ver la discusión completa en el notebook y en `reports/informe_laboratorio2.pdf`.
