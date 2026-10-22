# Lab 10 · Tablero interactivo — Observatorio de Turismo Sostenible (Guatemala)

CC3084 Data Science, UVG 2026-II · Dulce Ambrosio · Daniel Chet · Javier Linares

Tablero en **Streamlit + Plotly** para directivos del sector turístico. Usa la Base de Migración
2009–jun 2026 y **reutiliza los modelos del Laboratorio 1** (no entrena modelos nuevos).

## Ejecutar (desde la raíz del repo)
```bash
pip install -r lab10_dashboard/requirements.txt
python 01_limpieza_eda/src/limpieza.py          # genera data/processed/migracion_limpia.csv
python lab10_dashboard/preparar_datos.py         # agrega datos y reproduce predicciones/métricas del Lab 1
streamlit run lab10_dashboard/app.py
```

## Archivos
| Archivo | Para qué sirve |
|---|---|
| `preparar_datos.py` | Agrega la base por mes/vía/tipo/región/frontera y reproduce los 5 modelos del Lab 1 (SARIMA, Holt-Winters, Prophet, SES, ingenuo estacional) sobre el mismo split 70/30. |
| `app.py` | El tablero: pestañas Panorama, Mercados y fronteras, Predicciones y Acerca. |
| `data/` | CSV generados (agregado mensual, predicciones, métricas). |
| `docs/` | Documento de diseño (avance). |
| `../.streamlit/config.toml` | Tema de colores de la app. |

## Visualizaciones (avance)
V1 barras por año ↔ V2 detalle mensual · V3 mapa de calor · V4 regiones ↔ V5 fronteras ·
V8 error por modelo ↔ V7 observado vs. estimado. Pendientes para la entrega final: V6 y V9.

> Datos de uso académico; no son cifras oficiales del INGUAT ni del IGM.
