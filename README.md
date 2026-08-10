# ASL Alphabet Classifier

Clasificador de imágenes que reconoce letras del alfabeto de Lenguaje de Señas Americano (ASL) a
partir de fotos de manos. Es el motor de reconocimiento (una letra a la vez) para un prototipo de
traductor de señas en tiempo real.

**Mejor modelo:** una CNN (`CNN_B_profunda_do0.3`) con **96.6% accuracy / F1 macro** sobre un test
set propio, prácticamente empatada con un Random Forest sobre features HOG (96.7%). Ver
[`../reports/INFORME_FINAL.md`](../reports/INFORME_FINAL.md) para el análisis completo.

## Dataset

[ASL Alphabet (Kaggle)](https://www.kaggle.com/datasets/grassknoted/asl-alphabet): ~87,000 fotos
de 200x200 px a color, 29 clases (A-Z + SPACE, DELETE, NOTHING).

Por costo computacional se entrena sobre una **submuestra aleatoria de 600 imágenes/clase (17,400
en total)**, con las imágenes reducidas a 64x64 y normalizadas a `[0,1]`. El dataset ni la
submuestra se versionan en el repo (ver `.gitignore`); se regeneran con
`src/extraer_submuestra.py`.

## Modelos entrenados

| Modelo | Tipo | Test accuracy | Test F1 macro |
|---|---|---|---|
| **CNN_B_profunda_do0.3** | CNN (BatchNorm + Dropout) | **96.6%** | **96.6%** |
| RF_300_default | Random Forest sobre features HOG | 96.7% | 96.7% |
| MLP_A_512_256 | Red fully-connected | 80.5% | 81.9% |

Tabla completa con 11 corridas (4 CNN, 2 MLP, 3 RF+HOG, 2 con image augmentation) en
[`../data/processed/asl_alphabet/resultados_modelos.csv`](../data/processed/asl_alphabet/resultados_modelos.csv).

**Hallazgos principales:**
- Las letras más confundibles visualmente son el grupo S-T-U-V-W-X-Y (forma de mano muy similar) y,
  en menor medida, M/N/S.
- El image augmentation se hizo **sin flip horizontal/vertical**, porque invierte la posición
  relativa de dedos/pulgar y cambia el significado de la seña; en este dataset no mejoró los
  resultados.
- Al probar el mejor modelo con fotos tomadas por personas distintas a las del dataset, la
  precisión cae a ~20% — evidencia de que el modelo generaliza mal fuera de las condiciones
  (cámara, fondo, mano) del dataset de entrenamiento.

## Estructura del repo

```
05_deep_learning_asl/
├── src/
│   └── extraer_submuestra.py         # arma la submuestra estratificada desde el .zip de Kaggle
├── notebooks/
│   ├── avance_eda_preprocesamiento.ipynb   # exploración de datos y preprocesamiento
│   ├── entrega_final_modelos.ipynb         # entrenamiento y comparación de modelos (CNN, MLP, RF)
│   └── ejercicio8_fotos_propias.ipynb      # prueba del modelo con fotos externas al dataset
├── models/
│   ├── mejor_modelo.keras            # mejor modelo, listo para inferencia
│   └── mejor_modelo_info.json        # nombre, tipo y métricas del modelo guardado
└── requirements/
    └── requirements.txt

data/
├── raw/asl_alphabet/
│   ├── train/<clase>/*.jpg           # submuestra de entrenamiento
│   └── test/*.jpg                    # test set oficial de Kaggle (fuera de distribución)
└── processed/asl_alphabet/
    ├── split_train.csv, split_val.csv, split_test.csv   # split propio 70/15/15
    └── clases.json                   # mapeo clase -> índice
```

## Cómo ejecutar

```bash
pip install -r 05_deep_learning_asl/requirements/requirements.txt

# 1) Descargar archive.zip de https://www.kaggle.com/datasets/grassknoted/asl-alphabet
#    y ajustar la ruta ZIP_PATH en extraer_submuestra.py

# 2) Generar la submuestra local
python 05_deep_learning_asl/src/extraer_submuestra.py

# 3) Correr los notebooks en orden
jupyter nbconvert --to notebook --execute --inplace \
  05_deep_learning_asl/notebooks/avance_eda_preprocesamiento.ipynb
jupyter nbconvert --to notebook --execute --inplace \
  05_deep_learning_asl/notebooks/entrega_final_modelos.ipynb    # ~30-60 min en CPU
jupyter nbconvert --to notebook --execute --inplace \
  05_deep_learning_asl/notebooks/ejercicio8_fotos_propias.ipynb
```

Para usar el modelo ya entrenado sin re-ejecutar nada, carga directamente
`models/mejor_modelo.keras` con `tf.keras.models.load_model(...)`