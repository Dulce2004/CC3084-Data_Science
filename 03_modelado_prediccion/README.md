# 03 — Modelos de pronóstico (del Laboratorio 1)

Etapa reutilizada por el tablero del Laboratorio 10: `05_dashboard_interactivo/src/preparar_datos.py` vuelve a
correr estas funciones para guardar las predicciones y las métricas que muestra el tablero.

## Contenido
- `src/modelado.py` — ARIMA/SARIMA sobre `log1p(serie)` (manual y por búsqueda de AIC), Prophet, Holt-Winters,
  suavizamiento exponencial simple, seasonal naive, métricas MAE/RMSE y tabla comparativa.

## Decisiones de modelado
- Los SARIMA se ajustan sobre `log1p(serie)`; las predicciones se revierten con `expm1`.
- `d = 1` en las ocho series. `D = 1` si la fuerza estacional (STL) supera 0.3; Vía Terrestre, Excursionista y
  Viajero quedan con `D = 0`.
- Modelo manual `(1,d,1)(1,D,1)_12` contra búsqueda por AIC sobre p,q∈{0,1,2} y P,Q∈{0,1}.
- En **Vía Aérea** los SARIMA divergen al pronosticar 63 meses; el tablero los excluye de esa serie con un aviso.

El cuaderno de modelado está en la rama `Lab1-SeriesdeTiempo`.
