# 05 — Tablero interactivo (Laboratorio 10)

Observatorio de turismo para directivos: un tablero en **Streamlit + Plotly** sobre el ingreso mensual de viajeros
internacionales a Guatemala (ene-2009 a jun-2026). Sirve para explorar los datos y comparar los modelos de pronóstico
de los Laboratorios 1 y 2. No se entrenó ningún modelo nuevo.

*Datos de uso exclusivamente académico; no son cifras oficiales del INGUAT ni del IGM.*

## Contenido

- `src/app.py`: punto de entrada, navegación y filtros generales (período y tipo de viajero).
- `src/comun.py`: carga de datos, paleta con significado fijo y ayudas para los enlaces entre gráficas.
- `src/app_pages/`: una página por pregunta de gestión.

  | Página | Gráficas | Enlace |
  |---|---|---|
  | Panorama en el tiempo | llegadas por año, detalle mensual por vía, mapa de calor mes × año | clic en un año → sus meses |
  | Mercados de origen | treemap región → mercado, ranking de mercados, evolución del mercado por vía | clic en un mercado → su evolución |
  | Vías y puestos de entrada | mezcla anual por vía, puestos de entrada | clic en vía y año → sus puestos |
  | Modelos de pronóstico | pronóstico vs. real, precisión por modelo, error mes a mes, error modelo × serie | clic en un modelo → sus errores |
  | Inicio, Glosario | tarjetas, instrucciones, términos y limitaciones | — |

- `src/preparar_datos.py`: precálculo. Vuelve a correr los modelos del Lab 1 (Prophet, SES, Holt-Winters, seasonal
  naive, SARIMA manual y grid-AIC, en las 8 series) y el mejor LSTM del Lab 2 (Total y Vía Aérea), y comprueba que
  los RMSE coincidan con los publicados.
- `data/resultados/`: salida del precálculo (ver `codebook.md`). **Se versiona a propósito**: sin estos archivos el
  tablero necesitaría Prophet, TensorFlow y varios minutos de entrenamiento antes de abrir.
- `tests/test_app.py`: prueba de humo de todas las páginas con `streamlit.testing`.

## Ejecutar el tablero

```bash
cd 05_dashboard_interactivo
pip install -r requirements.txt
streamlit run src/app.py
python -m pytest tests -q        # prueba de humo
```

Hay que lanzarlo desde `05_dashboard_interactivo/` para que Streamlit lea el tema de `.streamlit/config.toml`.

## Regenerar `data/resultados/` (opcional)

Requiere Python 3.12, porque TensorFlow no publica wheels para 3.14. Se corre desde la raíz del repo y toma unos 10 minutos.

```bash
uv venv -p 3.12 .venv312 && uv pip install -p .venv312 -r 05_dashboard_interactivo/requirements-precalculo.txt
PYTHONIOENCODING=utf-8 .venv312/Scripts/python 05_dashboard_interactivo/src/preparar_datos.py
```

En Windows con sistema en español no use `PYTHONUTF8=1`: cmdstanpy (Prophet) decodifica la salida de `where.exe`
como UTF-8 y falla con `'Prophet' object has no attribute 'stan_backend'`.

## Documento de diseño

`reports/Documento de diseño Laboratorio 10.pdf`

## Equipo

| Integrante | Carné | Parte |
|---|---|---|
| Daniel Chet | 231177 | estructura, tema, Inicio, Panorama, documento de diseño |
| Dulce Ambrosio | 231143 | Mercados, Glosario |
| Cristian Tunchez | 231359 | Vías y puestos de entrada, README y codebook |
| Javier Linares | 231135 | precálculo de modelos, Pronósticos |
