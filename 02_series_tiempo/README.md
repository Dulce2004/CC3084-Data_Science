# 02 — Construcción y análisis preliminar de series

Segunda etapa: división entrenamiento/prueba, construcción de las series mensuales y su
análisis preliminar (incisos 2, 3 y 4.a–e del laboratorio).

## Contenido
- `src/series.py` — carga del CSV limpio, `serie_mensual()` (agregación mensual con frecuencia MS),
  `split_temporal()` (70/30), `resumen_serie()` (inicio/fin/frecuencia) y `adf_test()` (Dickey-Fuller aumentada).
- `notebooks/02_analisis_preliminar_series.ipynb` — para cada serie: gráfico, descomposición,
  ACF/PACF, prueba ADF y discusión de estacionariedad en media y varianza.

## Series construidas (a partir del entrenamiento)
- **Obligatoria:** Total mensual de viajeros.
- **Categoría 1 — Vías:** Aérea, Terrestre, Marítima.
- **Categoría 2 — Tipo de viajero:** Turista, Excursionista, Viajero, Cruceristas.

## Ejecutar
Requiere el CSV limpio (`python ../01_limpieza_eda/src/limpieza.py`).
```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/02_analisis_preliminar_series.ipynb
```

## Diagnóstico preliminar
Todas las series analizadas muestran tendencia, estacionalidad de período 12 y varianza
creciente con el nivel ⇒ no estacionarias en media ni en varianza; requieren transformación
logarítmica y una diferenciación (d = 1). Punto de partida para modelos **SARIMA(p,1,q)(P,1,Q)₁₂**.
