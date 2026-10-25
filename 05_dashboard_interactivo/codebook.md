# Codebook — `data/resultados/` (Laboratorio 10)

Archivos generados por `src/preparar_datos.py` a partir de `data/raw/Base_Migracion_2009-2026jun.xlsx`, limpiado
con `01_limpieza_eda/src/limpieza.py` (ver `01_limpieza_eda/codebook.md` para las variables originales).

## `mercados.csv`: viajeros por mes, región, mercado, vía y tipo

| Variable | Tipo | Descripción |
|---|---|---|
| `fecha` | fecha | Primer día del mes (2009-01-01 a 2026-06-01). |
| `Región dos` | categórica | Continente o gran área de procedencia (11 categorías). |
| `País` | categórica | País de procedencia hasta 2022; desde 2023, en parte, agrupación de mercado. |
| `Vía` | categórica | Aérea, Terrestre, Marítima. |
| `Tipo de Viajero` | categórica | Turista, Excursionista, Viajero, Cruceristas. |
| `Viajero` | numérica | Suma de viajeros (puede tener decimales por estimación de la fuente). |

## `fronteras.csv`: viajeros por mes, vía, puesto de entrada y tipo

Las mismas `fecha`, `Vía`, `Tipo de Viajero` y `Viajero`, más `Frontera` (puesto de entrada, 22 categorías; p. ej.
`01 La Aurora`).

## `predicciones.csv`: pronósticos en el período de prueba

Período de prueba: el 30 % final de cada serie, de abr-2021 a jun-2026 (63 meses).

| Variable | Tipo | Descripción |
|---|---|---|
| `fecha` | fecha | Mes pronosticado. |
| `serie` | categórica | Total, Vía Aérea, Vía Terrestre, Vía Marítima, Turista, Excursionista, Viajero, Cruceristas. |
| `modelo` | categórica | Prophet, SES, Holt-Winters, Seasonal naive, SARIMA manual, SARIMA grid-AIC, LSTM (este último solo en Total y Vía Aérea). |
| `real` | numérica | Viajeros observados. |
| `pred` | numérica | Viajeros pronosticados. |
| `lo`, `hi` | numérica | Límites del intervalo de 80 % (solo Prophet; vacío en los demás). |

## `metricas.csv`: error de cada modelo

| Variable | Tipo | Descripción |
|---|---|---|
| `serie`, `modelo` | categórica | Como en `predicciones.csv`. |
| `detalle` | texto | Configuración (órdenes SARIMA, capas y ventana del LSTM, etc.). |
| `MAE` | numérica | Error absoluto medio en viajeros. |
| `RMSE` | numérica | Raíz del error cuadrático medio en viajeros. |

El tablero calcula además el **error relativo** = MAE / promedio real de la serie en el período de prueba.
