# CC3084-Data_Science

Proyecto de análisis de series de tiempo para la materia CC3084 - Data Science. El trabajo principal está en el notebook [notebooks/01_EDA.ipynb](notebooks/01_EDA.ipynb), donde se explora la temperatura de Guatemala, se prepara el conjunto de entrenamiento y prueba, y se entrenan modelos de pronóstico con SARIMA y métodos de suavizamiento.

## Estructura

- [notebooks/01_EDA.ipynb](notebooks/01_EDA.ipynb): análisis exploratorio, preparación de datos, diagnóstico de serie temporal y evaluación de modelos.
- [data/guatemala_temperatura.csv](data/guatemala_temperatura.csv): datos originales.
- [data/processed/train.csv](data/processed/train.csv): conjunto de entrenamiento generado por el notebook.
- [data/processed/test.csv](data/processed/test.csv): conjunto de prueba generado por el notebook.
- [requirements.txt](requirements.txt): dependencias del proyecto.

## Requisitos

Instala las dependencias con:

```bash
pip install -r requirements.txt
```

## Uso

1. Abre [notebooks/01_EDA.ipynb](notebooks/01_EDA.ipynb) en VS Code o Jupyter.
2. Ejecuta las celdas en orden.
3. La celda de preparación de datos generará `train.csv` y `test.csv` dentro de `data/processed/`.
4. Las celdas posteriores realizan el análisis de estacionariedad, el entrenamiento de modelos y la evaluación de predicciones.

## Contenido del notebook

- Exploración visual de temperaturas por capa.
- División temporal de datos en entrenamiento y prueba.
- Análisis de estacionariedad con media/varianza móvil y pruebas estadísticas.
- Entrenamiento y validación de modelos SARIMA.
- Comparación con Holt-Winters, Seasonal Naive y suavizamiento exponencial simple.

## Autor

Dulce Ambrosio - 231143