# 06 — Flujo MLOps del Observatorio (Laboratorio 11)

Flujo reproducible que estima el ingreso mensual de viajeros internacionales al destino
**Petén / Mundo Maya** y publica una versión aprobada para que la consulte el tablero del
Laboratorio 10.

Este es el **avance** del Laboratorio 11 (entrega del 29-oct-2026): alcance, diagrama, pipeline
inicial y los seis controles de calidad funcionando. El registro en MLflow, el monitoreo
automático y la contenerización con Docker quedan para la entrega final (1-nov-2026); la
estructura ya los contempla.

*Datos de uso exclusivamente académico; no son cifras oficiales del INGUAT ni del IGM.*

## Alcance del flujo

| | |
|---|---|
| **Variable estimada** | `viajeros_destino`: viajeros internacionales que ingresan al destino en el mes |
| **Unidad** | viajeros por mes |
| **Nivel de agregación** | mes × destino |
| **Fuente** | `data/raw/Base_Migracion_2009-2026jun.xlsx`, hoja `Datos` (fuente mensual) |
| **Columnas usadas** | `Año`, `Mes cod`, `Vía`, `Frontera`, `País`, `Región dos`, `Tipo de Viajero`, `Viajero` |
| **Cobertura** | ene-2009 a jun-2026 (210 meses) |
| **Período de ajuste** | ene-2009 a dic-2025 (204 meses) |
| **Período de evaluación** | ene-2026 a jun-2026 (6 meses ocultos del ajuste) |
| **Archivo que lee el tablero** | `data/aprobado/version_actual.json` → `data/aprobado/<run_id>/estimaciones.csv` |

### El procedimiento de estimación

```
V(m) = D(m) + f · A(m)
```

- `D(m)` — entradas observadas en los puestos de entrada del propio destino: `02 Mundo Maya`,
  `17 El Naranjo`, `18 Bethel` y `20 Melchor de Mencos`, para los tipos *Turista* y *Excursionista*.
- `A(m)` — turistas que ingresan por `01 La Aurora` (vía aérea) y pueden continuar al destino.
- `f` — **factor de reparto**: proporción de esos turistas que visita el destino.

`D` y `A` se calculan sobre conjuntos de registros **disjuntos**, así que ningún viajero se cuenta
dos veces; el control 6 lo verifica en cada ejecución. Las conexiones domésticas La Aurora → Flores
no aparecen en la base de migración, de modo que sumarlas por separado no duplicaría registros,
pero sí duplicaría personas: por eso el término indirecto se estima con un factor menor que 1 en
lugar de sumar otra serie observada.

### Supuestos declarados

- **`factor_reparto = 0.12`** es un supuesto del observatorio, no un parámetro estimado de los
  datos. Es provisional y está pendiente de calibrar con el registro de visitantes del Parque
  Nacional Tikal (IDAEH). La configuración declara además un rango de sensibilidad (0.08 – 0.18) y
  el control 6 advierte si el valor usado queda fuera.
- Se excluye el tipo de viajero *Viajero*, cuyo criterio de registro cambió en 2023, y *Cruceristas*,
  que no ingresa por los puestos del destino.
- Un mes del rango de cobertura sin información **no se convierte en cero**: queda vacío y el
  control 2 lo reporta.

## Diagrama del flujo

![Flujo del pipeline](../reports/figuras/lab11_flujo.png)

Se genera con `python 06_mlops/src/diagrama.py`, para que la figura del informe corresponda siempre
al pipeline implementado. Marca los tres puntos que pide el enunciado: dónde se **validan los
datos** (etapa 2), dónde se **evalúa el procedimiento** (etapa 5) y dónde se **autoriza la
publicación** (regla de aceptación, antes de la etapa 7).

## Cómo ejecutarlo

```bash
pip install -r 06_mlops/requirements.txt

# flujo completo: lectura → validación → preparación → ajuste → evaluación → almacenamiento → publicación
python 06_mlops/src/pipeline.py --config 06_mlops/config/pipeline.yaml

# solo lectura y controles de calidad
python 06_mlops/src/pipeline.py --solo-validar

# ejecuta todo sin tocar la versión aprobada
python 06_mlops/src/pipeline.py --sin-publicar

# deja vigente una versión aprobada anterior
python 06_mlops/src/pipeline.py --restaurar 20261030T050644Z-6da9-942c1b0

# pruebas
python -m pytest 06_mlops/tests -q
```

`--config` es opcional: sin él se usa `06_mlops/config/pipeline.yaml`. Los datos institucionales se
pueden montar fuera del repositorio con la variable de entorno `OBS_DATOS_CRUDOS`, que tiene
prioridad sobre la ruta de la configuración (así no entran en la imagen Docker de la entrega final).

Nada se edita a mano entre etapas y nada depende del estado de una sesión: las únicas entradas son
el archivo de configuración y las fuentes que este declara. Cambiar un filtro en el tablero no
vuelve a ajustar el procedimiento, porque el tablero solo lee el archivo publicado.

## Los seis controles de calidad

| # | Control | Qué verifica | Advierte cuando | Detiene cuando |
|---|---|---|---|---|
| 1 | `columnas_tipos` | columnas indispensables presentes y con tipo compatible | aparece una columna no declarada (cambio de estructura) | falta una columna indispensable o una no admite su tipo |
| 2 | `fechas_cobertura` | `Año`/`Mes cod` forman fechas válidas; todos los meses de la cobertura traen información | hay meses fuera del rango, o meses sin cobertura dentro del umbral | hay fechas inválidas, o los meses sin cobertura superan `max_meses_sin_cobertura` |
| 3 | `unicidad` | sin filas repetidas según la clave que define la granularidad | hay duplicados por debajo de `max_prop_duplicados` | la proporción de duplicados supera ese umbral |
| 4 | `valores_faltantes` | sin ausencias en las columnas necesarias para estimar | una columna de ausencia permitida supera `max_prop_nulos_permitidos` | falta un valor en una columna indispensable |
| 5 | `valores_denominadores` | conteos no negativos y finitos; denominador del reparto distinto de cero | el denominador es cero en un mes fuera del período evaluado | hay conteos negativos o no finitos, o el denominador es cero o falta dentro del período evaluado |
| 6 | `regla_metodologica` | serie mensual de fuente mensual; sin duplicar visitantes; factor de reparto en (0, 1] | el factor queda fuera del rango de sensibilidad declarado | la fuente trae granularidad diaria, un año supera 12 meses, los puestos del destino y el del reparto se solapan o faltan, o el factor está fuera de (0, 1] |

El nivel de cada control (`critico` o `advertencia`) está en la configuración, no en el código. Los
seis corren siempre, aunque uno falle, para que el reporte describa el estado completo de la fuente;
quien decide detenerse es el pipeline. Un control que lanza una excepción queda en error: **un
control roto nunca aprueba una versión.**

El resultado se escribe en `controles.csv` de cada ejecución, con una fila por control: qué verifica,
nivel, estado, si detuvo la ejecución, filas afectadas y detalle.

## Regla de aceptación

Se evalúa en este orden y basta que una condición falle para rechazar la versión candidata:

1. ningún control crítico en error;
2. salidas completas: una fila por mes de evaluación, sin valores vacíos ni no finitos;
3. `MAE ≤ 12 000` viajeros/mes;
4. error relativo (`MAE` / promedio observado del período) `≤ 35 %`.

Superar a la referencia **no** es obligatorio: la condición existe en la configuración
(`mejor_o_igual_que_referencia`) y está en `false`. Los umbrales se justifican por la escala de la
serie: el destino promedia unos 24 000 viajeros/mes, así que 12 000 es la mitad del volumen típico y
35 % es holgado frente al 13 % que alcanza el procedimiento actual, pero rechaza una degradación
evidente. La regla se aplica tal como está escrita y no se modifica después de ver los resultados.

## Resultados del avance

Dos ejecuciones completas con los mismos datos y la misma configuración
(`20261030T050644Z-6da9-942c1b0` y `20261030T050712Z-3df2-942c1b0`):

| | Procedimiento (Holt-Winters) | Referencia (promedio histórico) |
|---|---|---|
| MAE | 3 110.0 | 4 264.4 |
| RMSE | 3 669.0 | 4 760.2 |
| Error relativo | 12.9 % | 17.8 % |

Las dos ejecuciones produjeron estimaciones **idénticas** (diferencia máxima 0.0, por debajo de la
tolerancia configurada de 1e-6): los procedimientos usados son deterministas y la semilla queda
fijada igual.

### Hallazgo: la clave declarada no coincidía con la granularidad de la fuente

El control 3 detuvo la primera ejecución con 42 filas duplicadas. No eran filas repetidas: el mismo
país aparece en dos valores distintos de `Agrupación Residencia` en 20 combinaciones de 2020–2022, así
que la clave de la granularidad incluye esa columna. Se corrigió la configuración, no los datos. Es
el tipo de problema que el control existe para detectar: sin él, agregar la serie habría sumado sin
avisar registros que la fuente distingue.

## Estructura

```
06_mlops/
├── config/pipeline.yaml   rutas, períodos, parámetros, umbrales y niveles de control
├── src/
│   ├── pipeline.py        orquestador: un comando corre las siete etapas
│   ├── config.py          carga y valida la configuración; huellas y commit de Git
│   ├── ingesta.py         etapa 1: lectura de las fuentes
│   ├── validaciones.py    etapa 2: los seis controles de calidad
│   ├── preparacion.py     etapa 3: serie del destino y factor de reparto
│   ├── modelo.py          etapa 4: ajuste (reutiliza el Laboratorio 1) y referencia
│   ├── evaluacion.py      etapa 5: métricas y regla de aceptación
│   ├── publicacion.py     etapas 6 y 7: expediente, publicación y restauración
│   ├── registro.py        run_id, bitácora por ejecución e historial
│   └── diagrama.py        genera la figura del flujo
├── tests/                 52 pruebas: controles, flujo, fallo controlado y recuperación
├── data/
│   ├── resultados/<run_id>/   expediente de cada ejecución (aprobada o no)
│   ├── aprobado/<run_id>/     versiones aprobadas conservadas
│   ├── aprobado/version_actual.json   puntero a la versión vigente
│   └── historial_ejecuciones.jsonl    una línea por ejecución
├── logs/<run_id>.log      bitácora: inicio, fin, etapas, advertencias, errores, publicación
├── requirements.txt
├── codebook.md
└── README.md
```

## Pendiente para la entrega final

- Registro de ejecuciones en MLflow Tracking (ejercicio 4).
- Integración visual con el tablero: distinguir observado de estimado, mostrar fecha de
  actualización y métrica de validación (ejercicio 5).
- Verificaciones de monitoreo —actualización, calidad y resultados— sobre el historial (ejercicio 6).
- Docker y Docker Compose con persistencia de MLflow y de las versiones aprobadas (ejercicio 7).

## Equipo

| Integrante | Carné |
|---|---|
| Daniel Chet | 231177 |
| Dulce Ambrosio | 231143 |
| Cristian Tunchez | 231359 |
| Javier Linares | 231135 |
