# Laboratorio 3 — Deep Learning: Reconocimiento de Lenguaje de Señas (ASL)

CC3084 – Data Science, UVG, Semestre II 2026.

**Alcance de esta carpeta: solo el avance** (Análisis Exploratorio, Preprocesamiento, y plan de
selección de modelos), correspondiente a los ejercicios 1-3 del enunciado. El entrenamiento y
evaluación de los modelos (ejercicios 4-10: CNNs, red fully-connected, modelo clásico, image
augmentation, pruebas con fotos propias, reflexión de accesibilidad e informe final) queda para la
entrega final y todavía no está implementado aquí.

## Dataset

[ASL Alphabet (Kaggle)](https://www.kaggle.com/datasets/grassknoted/asl-alphabet): 87,000 imágenes
de 200x200 px a color, 29 clases (A-Z, SPACE, DELETE, NOTHING), 3,000 img/clase.

Por el peso del dataset completo, se trabaja con una **submuestra aleatoria de 600 imágenes por
clase (17,400 en total)**, extraída directamente del .zip descargado de Kaggle (no se versiona el
dataset ni la submuestra, ver `.gitignore`).

## Estructura

```
05_deep_learning_asl/
├── src/
│   └── extraer_submuestra.py       # extrae la submuestra estratificada desde el .zip de Kaggle
├── notebooks/
│   ├── avance_eda_preprocesamiento.py     # fuente editable (jupytext, formato percent)
│   └── avance_eda_preprocesamiento.ipynb  # notebook ejecutado (informe del avance)
└── requirements/
    └── requirements.txt
```

Datos (fuera de esta carpeta, mismo patrón que el resto del repo):

```
data/
├── raw/asl_alphabet/
│   ├── train/<clase>/*.jpg     # submuestra, 600 img/clase
│   └── test/*.jpg              # set de prueba oficial de Kaggle (28 img, 1 por clase)
└── processed/asl_alphabet/
    ├── split_train.csv, split_val.csv, split_test.csv   # rutas del split propio 70/15/15
    ├── X_train.npy, y_train.npy  (y equivalentes val/test)  # imágenes preprocesadas 64x64x3, [0,1]
    └── clases.json               # mapeo clase -> índice
```

## Cómo reproducir

```bash
pip install -r 05_deep_learning_asl/requirements/requirements.txt

# 1) Descargar archive.zip de https://www.kaggle.com/datasets/grassknoted/asl-alphabet
#    y ajustar la ruta ZIP_PATH en extraer_submuestra.py

# 2) Extraer la submuestra
python 05_deep_learning_asl/src/extraer_submuestra.py

# 3) Ejecutar el notebook del avance
jupyter nbconvert --to notebook --execute --inplace \
  05_deep_learning_asl/notebooks/avance_eda_preprocesamiento.ipynb
```

## Resumen de hallazgos del avance

- Imágenes: JPG 200x200 RGB, confirmado sobre una muestra de cada clase.
- Dataset perfectamente balanceado (600 img/clase en la submuestra, 3,000 en el original).
- Los grupos de letras más confundibles visualmente (correlación de imagen promedio por clase):
  S/T/U/V/W/X/Y (>0.97 entre sí) y, en menor magnitud, M/N/S (0.93-0.95) y U/V/R (0.98-0.99) —
  confirmando los ejemplos del enunciado.
- Split propio 70/15/15 estratificado por clase (semilla fija), reservando el set oficial de Kaggle
  (28 img) como prueba adicional fuera de distribución.
- Preprocesamiento: resize a 64x64, normalización [0,1], se conserva color RGB, sin filtros
  adicionales (justificación en el notebook).
- Plan de modelos para la entrega final: 2 CNN (baseline y profunda+regularizada), 1 MLP
  fully-connected, 1 modelo clásico (Random Forest o SVM sobre features reducidas).
