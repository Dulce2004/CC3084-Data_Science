# 01 — Limpieza de datos

Módulo compartido: ingesta del Excel crudo y limpieza. El análisis exploratorio (notebook
`01_eda.ipynb`) era del Laboratorio 1 y ya no forma parte del repositorio; el código de esta
carpeta se conserva porque `04_deep_learning_lstm` lo importa directamente (`config.py`, y
`limpieza.py` para generar el CSV limpio).

## Contenido
- `src/config.py` — rutas del proyecto, paleta de colores accesible y estilo de gráficas
  (usado por `04_deep_learning_lstm`).
- `src/limpieza.py` — pipeline de limpieza (ingesta → normalización de texto → columna `fecha`
  → depuración de categorías inválidas → validación → escritura de `data/processed/migracion_limpia.csv`).

## Ejecutar
```bash
python src/limpieza.py    # genera el CSV limpio, requerido por 04_deep_learning_lstm
```

## Notas de limpieza
- Se eliminan 21 filas con `Región dos` inválida (`"0"`, `"Cruceros"`).
- `Viajero` contiene valores fraccionarios (conteos estimados/prorrateados); se conservan.
- 210 meses consecutivos, sin nulos ni duplicados tras la limpieza.
