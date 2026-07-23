# Laboratorio 1 — Series de Tiempo (CC3084 Data Science, UVG 2026-II)

Análisis de series de tiempo del **ingreso mensual de viajeros internacionales a Guatemala**
(ene-2009 a jun-2026). *Datos de uso exclusivamente académico; no son cifras oficiales del INGUAT ni del IGM.*

## Estructura del repositorio

```
Lab1/
├── data/
│   ├── raw/            # Base_Migracion_2009-2026jun.xlsx  (inmutable)
│   └── processed/      # migracion_limpia.csv  (generado, no versionado)
├── 01_limpieza_eda/
│   ├── src/            # config.py, limpieza.py (pipeline de limpieza)
│   ├── notebooks/      # 01_eda.ipynb  (análisis exploratorio)
│   ├── requirements.txt
│   └── README.md
├── 02_series_tiempo/
│   ├── src/            # series.py (construcción de series, ADF, helpers)
│   ├── notebooks/      # 02_analisis_preliminar_series.ipynb
│   ├── requirements.txt
│   └── README.md
├── reports/figuras/    # figuras exportadas para el informe
├── docs/               # enunciado del laboratorio (PDF)
└── README.md
```

El diseño sigue la convención por etapas del repositorio de referencia del curso
(`src/`, `notebooks/`, `data/raw`, `data/processed`, un `requirements.txt` y un `README.md` por etapa).

## Cómo reproducir

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r 01_limpieza_eda/requirements.txt

# 1) Generar el dataset limpio
python 01_limpieza_eda/src/limpieza.py

# 2) Ejecutar los cuadernos (EDA y series)
jupyter nbconvert --to notebook --execute --inplace 01_limpieza_eda/notebooks/01_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace 02_series_tiempo/notebooks/02_analisis_preliminar_series.ipynb
```

## Estado del avance (entrega 23-jul-2026)

- [x] **Análisis exploratorio** general de los datos (cuaderno `01_eda.ipynb`).
- [x] **División temporal** 70/30 (147 meses entrenamiento / 63 prueba).
- [x] **Construcción de series**: Total mensual + Vías (3) + Tipo de viajero (4).
- [x] **Análisis preliminar** de series: gráfico, descomposición, ACF/PACF, prueba ADF
      y discusión de estacionariedad (Total, Aérea, Terrestre, Turista).

### Pendiente para la entrega final (26-jul-2026)
Modelos ARIMA/SARIMA y comparación con Prophet, Holt-Winters, suavizamiento exponencial
y seasonal naïve; predicción sobre el conjunto de prueba; métricas MAE/RMSE/AIC/BIC;
y análisis comparativo entre categorías.

## Categorías seleccionadas
Además de la serie obligatoria (Total mensual), se analizan **Vías de ingreso** y **Tipo de viajero**.
