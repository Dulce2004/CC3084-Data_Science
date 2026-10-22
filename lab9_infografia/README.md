# Lab 9 · Infografía "Guatemala en 17 años de viajeros" (CC3084 Data Science, UVG)

Autores: Dulce Rebeca Ambrosio Jiménez · Daniel Alejandro Chet Delgado · Javier Alexander Linares Chang

Los datos y los modelos salen del **Laboratorio 1 (Series de Tiempo)**:
https://github.com/Dulce2004/CC3084-Data_Science/tree/Lab1-SeriesdeTiempo
(Base de Migración 2009–jun 2026, INGUAT/IGM). Esta carpeta `lab9_infografia/` va dentro de ese mismo repo.

> Los datos son de uso estrictamente académico: el repositorio debe ser **privado** y la infografía no se publica sin consulta previa.

## Archivos
| Archivo | Para qué sirve |
|---|---|
| `calcular_datos.py` | Usa los módulos del Lab 1 (`series`, `modelado`) para armar la serie mensual de viajeros y el pronóstico de Prophet con intervalo de 80%. Guarda `tot.csv`, `te_ex.csv` y `prophet_fc.csv`. |
| `graficas_lab9.py` | Genera las 5 gráficas (PNG, 300 dpi, fondo blanco) en `graficas/`. |
| `graficas_lab9.ipynb` | Lo mismo en notebook, ya ejecutado (las gráficas están en las salidas de las celdas). |
| `tot.csv`, `prophet_fc.csv`, `te_ex.csv` | Series ya calculadas para poder correr las gráficas sin el repo completo. |
| `graficas/` | g1 línea de tiempo · g2 estacionalidad · g3 cuadrante · g4 modelos (RMSE/MAE) · g5 proyección de Prophet. |

## Cómo reproducir
```bash
# 1) desde la raíz del repo del Lab 1 (con data/processed/migracion_limpia.csv generado)
python lab9_infografia/calcular_datos.py      # regenera los CSV
# 2) gráficas
python lab9_infografia/graficas_lab9.py       # o abrir graficas_lab9.ipynb
```
La composición final de la página (textos, franjas de color, orden de los bloques) se armó en Canva.
Paleta: azul vivo #2889E2 · azul marino #1F3A5F · fondos #EAF0F7 / #D3DEEB · acento naranja #C4571A.
