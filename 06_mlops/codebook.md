# Codebook — salidas del flujo MLOps (Laboratorio 11)

Archivos generados por `src/pipeline.py`. Cada ejecución escribe su expediente completo en
`data/resultados/<run_id>/`; si la versión se aprueba, los mismos archivos se copian a
`data/aprobado/<run_id>/`.

El `run_id` tiene la forma `AAAAMMDDTHHMMSSZ-xxxx-<commit>`: fecha-hora UTC de inicio, sufijo
aleatorio (para que dos ejecuciones del mismo segundo no coincidan) y los primeros 7 caracteres del
commit de Git que produjo la ejecución.

## `estimaciones.csv`: estimaciones del destino

Una fila por mes estimado.

| Variable | Tipo | Descripción |
|---|---|---|
| `destino` | texto | Nombre del destino (`Petén / Mundo Maya`). |
| `codigo_destino` | texto | Código corto del destino (`PETEN`). |
| `periodo` | texto `AAAA-MM` | Mes estimado. |
| `valor` | numérica | Viajeros estimados en el mes. |
| `unidad` | texto | Siempre `viajeros por mes`. |
| `tipo_valor` | texto | `estimado`. Distingue la estimación del dato observado. |
| `valor_observado` | numérica | Viajeros observados en el mes, si existen; vacío cuando no hay valor comparable. |
| `run_id` | texto | Identificador de la ejecución que lo produjo. |
| `fecha_generacion` | fecha-hora UTC | Cuándo se generó el archivo. |
| `version_procedimiento` | texto | Versión del procedimiento (`version_procedimiento` de la configuración). |

## `metricas.csv`: desempeño sobre el período de evaluación

Dos filas: el procedimiento y la referencia, evaluados sobre los mismos meses.

| Variable | Tipo | Descripción |
|---|---|---|
| `papel` | categórica | `procedimiento` o `referencia`. |
| `procedimiento` | categórica | `holt_winters`, `ses`, `seasonal_naive`, `ultimo_valor`, `promedio_historico`. |
| `n_meses` | entero | Meses con valor observado comparable. |
| `promedio_observado` | numérica | Promedio observado del período evaluado (denominador del error relativo). |
| `MAE` | numérica | Error absoluto medio, en viajeros. |
| `RMSE` | numérica | Raíz del error cuadrático medio, en viajeros. |
| `error_relativo` | numérica | `MAE` / `promedio_observado`. |
| `sesgo` | numérica | Error medio con signo: positivo sobreestima, negativo subestima. |
| `MAPE` | numérica | Error porcentual absoluto medio. **Vacío** si algún valor observado del período es cero, porque la división no está definida. |
| `nota_mape` | texto | Explica por qué `MAPE` quedó vacío, cuando corresponde. |

## `controles.csv`: resultado de los seis controles de calidad

Una fila por control, en el orden en que corren.

| Variable | Tipo | Descripción |
|---|---|---|
| `control` | texto | Número y nombre (`3. unicidad`). |
| `verifica` | texto | Qué comprueba el control. |
| `nivel` | categórica | `critico` o `advertencia`, según la configuración. |
| `estado` | categórica | `ok`, `advertencia` o `error`. |
| `detiene_ejecucion` | booleano | `True` solo cuando un control crítico queda en error. |
| `filas_afectadas` | entero | Filas (o meses, o puestos) involucrados en el problema. |
| `detalle` | texto | Descripción legible, con ejemplos cuando ayudan a corregir. |

## `metadatos.json`: trazabilidad de la ejecución

| Clave | Contenido |
|---|---|
| `run_id`, `destino`, `codigo_destino` | identificación de la ejecución y del destino |
| `version_procedimiento`, `commit_git` | versión del procedimiento y commit que la produjo (`-sucio` si el árbol tenía cambios sin confirmar) |
| `configuracion` | ruta del archivo de configuración y su huella SHA-256 |
| `fuente` | nombre, ruta, formato, hoja, huella SHA-256, tamaño, filas, columnas y granularidad de la fuente |
| `semilla`, `dependencias` | semilla usada y versiones de Python y de los paquetes |
| `procedimiento` | variable, definición, unidad, agregación, fórmula, puestos, tipos y factor de reparto |
| `periodo_ajuste`, `periodo_evaluacion` | períodos usados |
| `serie` | meses con y sin cobertura, promedio mensual y aporte de cada componente del estimador |
| `ajuste` | modelo, detalle, parámetros calculados, referencia y tramo de ajuste |
| `controles` | resumen del reporte: conteos por estado y controles en error |
| `metrica_validacion` | MAE, RMSE, error relativo, MAPE y meses evaluados |
| `aceptacion` | cada condición de la regla, si se cumplió y por qué; comparación con la referencia |
| `limitaciones_validacion` | limitaciones que impiden reproducir exactamente la situación de enero y febrero de 2026 |
| `detenido_en` / `error` | motivo, cuando la ejecución se detuvo o falló |

También se copia el archivo de configuración usado (`pipeline.yaml`): sin él el expediente no sería
reproducible.

## `data/aprobado/version_actual.json`: puntero a la versión vigente

Es el único archivo que el tablero necesita leer para saber qué versión consultar. Si no existe,
todavía no hay versión aprobada y el tablero debe informar que las estimaciones no están disponibles.

| Clave | Descripción |
|---|---|
| `run_id` | ejecución cuya salida está vigente |
| `carpeta` | ruta de la carpeta aprobada, relativa a la raíz del repositorio |
| `destino`, `version_procedimiento` | destino y versión del procedimiento publicados |
| `fecha_publicacion` | cuándo se publicó esta versión |
| `version_anterior` | `run_id` de la versión a la que reemplazó (o `null` si es la primera) |
| `periodo_estimado` | período que cubren las estimaciones |
| `metrica_validacion` | métrica de validación de esa versión, para mostrarla en el tablero |
| `restaurada` | presente y `true` cuando el puntero se reescribió con `--restaurar` |

## `data/historial_ejecuciones.jsonl` y `logs/<run_id>.log`

El historial tiene una línea JSON por ejecución, con `run_id`, `estado` (`aprobada`,
`aprobada_sin_publicar`, `rechazada`, `detenida`, `fallida`, `validado` o `restaurada`), horas de
inicio y fin, duración, etapas completadas, advertencias, errores y resultado de la publicación. Es
la entrada del monitoreo de la entrega final.

La bitácora repite lo mismo en texto legible, línea a línea y con marca de tiempo, incluyendo el
detalle de cada control y de cada condición de la regla de aceptación.
