# 02 — Construcción de series

Módulo compartido: construcción de las series mensuales y su partición entrenamiento/prueba. El
análisis preliminar (notebook `02_analisis_preliminar_series.ipynb`) era del Laboratorio 1 y ya no
forma parte del repositorio; el código de esta carpeta se conserva porque `04_deep_learning_lstm`
lo importa directamente para reconstruir las 8 series y comparar contra los modelos del Lab. 1.

## Contenido
- `src/series.py` — carga del CSV limpio, `serie_mensual()` (agregación mensual con frecuencia MS),
  `split_temporal()` (70/30), `resumen_serie()` (inicio/fin/frecuencia) y `adf_test()` (Dickey-Fuller aumentada).

## Series construidas (a partir del entrenamiento)
- **Obligatoria:** Total mensual de viajeros.
- **Categoría 1 — Vías:** Aérea, Terrestre, Marítima.
- **Categoría 2 — Tipo de viajero:** Turista, Excursionista, Viajero, Cruceristas.

## Ejecutar
Requiere el CSV limpio (`python ../01_limpieza_eda/src/limpieza.py`).
