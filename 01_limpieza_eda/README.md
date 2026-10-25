# 01 — Limpieza de datos (del Laboratorio 1)

Etapa reutilizada por el tablero del Laboratorio 10 (`05_dashboard_interactivo/src/preparar_datos.py`).

## Contenido
- `src/config.py` — rutas del proyecto y proporción entrenamiento/prueba (70/30).
- `src/limpieza.py` — ingesta del Excel crudo (`data/raw/`), normalización de texto, columna `fecha`, depuración de
  categorías inválidas y validación.
- `codebook.md` — variables de la base y advertencias metodológicas.

## Notas de limpieza
- Se eliminan 21 filas con `Región dos` inválida (`"0"`, `"Cruceros"`).
- `Viajero` contiene valores fraccionarios (conteos estimados/prorrateados); se conservan.
- 210 meses consecutivos, sin nulos ni duplicados tras la limpieza.

El análisis exploratorio completo está en la rama `Lab1-SeriesdeTiempo`.
