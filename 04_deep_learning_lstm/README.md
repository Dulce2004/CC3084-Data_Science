# 04 — LSTM (del Laboratorio 2)

Etapa reutilizada por el tablero del Laboratorio 10.

## Contenido
- `src/lstm_utils.py` — preparación de datos (log1p, escalado min-max con entrenamiento, ventanas), construcción y
  entrenamiento del LSTM, pronóstico recursivo y `tunear_lstm()`.

## Modelos que usa el tablero
El mejor LSTM de cada serie en el Laboratorio 2, con la semilla `fijar_semillas()`:

| Serie | Configuración | RMSE |
|---|---|---|
| Total mensual | 1 capa de 32, ventana 12, lr 1e-3 | 162,502 |
| Vía Aérea | 2 capas 32→16, ventana 18, dropout 0.1, lr 5e-4 | 55,424 |

El cuaderno completo (tuneo y análisis catch22) está en la rama `Lab2-ST-LSTM`.
