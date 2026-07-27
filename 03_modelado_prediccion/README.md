# 03 — Modelado y predicción

Tercera etapa: completa el análisis preliminar (inciso 4.a–e) de las cuatro series que faltaban
y hace el modelado completo (inciso 4.f–k) de las ocho series, más el análisis comparativo
(ejercicio 5).

## Contenido
- `src/modelado.py` — ajuste ARIMA/SARIMA sobre `log1p(serie)` (manual y por grid-search de AIC,
  equivalente casero de `auto_arima`), Prophet, Holt-Winters, suavizamiento exponencial simple,
  seasonal naive, métricas MAE/RMSE, Ljung-Box y tabla comparativa final.
- `notebooks/03_modelado.ipynb`:
  - **Parte A** — inciso 4.a–e de Vía Marítima, Tipo Excursionista, Tipo Viajero y Tipo
    Cruceristas (las otras cuatro series ya se cubrieron en `02_series_tiempo`).
  - **Parte B** — inciso 4.f–k de las ocho series: elección de d/D, lectura de ACF/PACF,
    modelo manual vs. grid-AIC, diagnóstico de residuos, comparación con Prophet/Holt-Winters/
    SES/seasonal naive, y pronóstico del mejor modelo sobre el conjunto de prueba.
  - **Ejercicio 5** — comparación de estacionalidad, tendencia, volatilidad e impacto de la
    pandemia entre las series de cada categoría, y hallazgos generales para INGUAT.

## Ejecutar
Requiere el CSV limpio y los `src/` de las etapas 01 y 02.
```bash
python ../01_limpieza_eda/src/limpieza.py
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/03_modelado.ipynb
```

## Decisiones de modelado
- Todos los modelos ARIMA/SARIMA se ajustan sobre `log1p(serie)` (no sobre la serie cruda) para
  estabilizar la varianza; las predicciones se revierten con `expm1` antes de comparar métricas.
- `d = 1` en las ocho series (confirmado por ADF: no rechaza en niveles, sí tras una diferencia).
- `D` (diferenciación estacional) se decide por la fuerza estacional de una descomposición STL:
  `D = 1` si supera 0.3, si no `D = 0`. Vía Terrestre y Tipo Viajero/Excursionista quedan con D=0
  (estacionalidad débil); el resto con D=1.
- El "modelo manual" de referencia es siempre `(1,d,1)(1,D,1)_12`, la lectura típica cuando ACF y
  PACF decaen gradualmente tras el primer rezago; se contrasta contra una búsqueda por AIC sobre
  p,q∈{0,1,2} y P,Q∈{0,1} (36 combinaciones), que hace de equivalente casero a `auto_arima`.

## Hallazgo a destacar
En **Vía Aérea** tanto el SARIMA manual como el de grid-AIC divergen numéricamente al pronosticar
(la doble diferenciación d=1,D=1 en escala log acumula deriva a lo largo de 63 meses de
pronóstico, y `expm1` de esa deriva explota). Se documenta en vez de descartarse en silencio:
es evidencia de que un buen AIC no garantiza un buen pronóstico a horizonte largo.
