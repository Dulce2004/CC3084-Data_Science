# 01 — Limpieza y Análisis Exploratorio (EDA)

Primera etapa: ingesta del Excel crudo, limpieza y análisis exploratorio (inciso 1 del laboratorio).

## Contenido
- `src/config.py` — rutas del proyecto, paleta de colores accesible y estilo de gráficas.
- `src/limpieza.py` — pipeline de limpieza (ingesta → normalización de texto → columna `fecha`
  → depuración de categorías inválidas → validación → escritura de `data/processed/migracion_limpia.csv`).
- `notebooks/01_eda.ipynb` — análisis exploratorio: comportamiento temporal, países, regiones,
  vías y fronteras, valores faltantes/duplicados/atípicos, estadísticas descriptivas e interpretación.

## Ejecutar
```bash
python src/limpieza.py                                   # genera el CSV limpio
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
```

## Notas de limpieza
- Se eliminan 21 filas con `Región dos` inválida (`"0"`, `"Cruceros"`).
- `Viajero` contiene valores fraccionarios (conteos estimados/prorrateados); se conservan.
- 210 meses consecutivos, sin nulos ni duplicados tras la limpieza.
