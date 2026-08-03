"""
Utilidades para el Laboratorio 2 (Deep Learning — LSTM).

Construye ventanas supervisadas a partir de series mensuales, entrena y
tunea modelos LSTM (Keras/TensorFlow), y compara contra los modelos
clásicos del Laboratorio 1 (ARIMA/SARIMA, Prophet, Holt-Winters, SES,
seasonal naive) usando las mismas métricas (MAE, RMSE) y el mismo
split temporal 70/30.

Reutiliza rutas/estilo de 01_limpieza_eda/src/config.py y las funciones
de partición de 02_series_tiempo/src/series.py.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "01_limpieza_eda" / "src"))
sys.path.append(str(ROOT / "02_series_tiempo" / "src"))
sys.path.append(str(ROOT / "03_modelado_prediccion" / "src"))
import config  # noqa: E402
import series  # noqa: E402

RANDOM_SEED = 42


def fijar_semillas(seed: int = RANDOM_SEED):
    """Fija semillas de numpy/tensorflow/python para reproducibilidad."""
    import os
    import random
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf
    tf.random.set_seed(seed)


# ------------------------------------------------------------------
# Transformación log1p (idéntica a modelado.py del Laboratorio 1)
# ------------------------------------------------------------------

def log1p(s):
    return np.log1p(s)


def expm1(s):
    if isinstance(s, pd.Series):
        return np.expm1(s).clip(lower=0)
    return np.clip(np.expm1(s), 0, None)


# ------------------------------------------------------------------
# Escalado min-max ajustado SOLO sobre entrenamiento (evita fuga de info)
# ------------------------------------------------------------------

class EscaladorMinMax:
    def __init__(self):
        self.min_ = None
        self.max_ = None

    def fit(self, arr):
        arr = np.asarray(arr, dtype=float)
        self.min_ = arr.min()
        self.max_ = arr.max()
        return self

    def transform(self, arr):
        arr = np.asarray(arr, dtype=float)
        rango = (self.max_ - self.min_)
        rango = rango if rango != 0 else 1.0
        return (arr - self.min_) / rango

    def inverse_transform(self, arr):
        rango = (self.max_ - self.min_)
        rango = rango if rango != 0 else 1.0
        return np.asarray(arr, dtype=float) * rango + self.min_

    def fit_transform(self, arr):
        return self.fit(arr).transform(arr)


# ------------------------------------------------------------------
# Ventanas supervisadas para entrenamiento tipo secuencia -> siguiente valor
# ------------------------------------------------------------------

def crear_ventanas(arr_1d: np.ndarray, look_back: int):
    """
    Convierte un arreglo 1D en pares (X, y) para entrenamiento supervisado:
    X[i] = arr[i : i+look_back], y[i] = arr[i+look_back]
    """
    X, y = [], []
    for i in range(len(arr_1d) - look_back):
        X.append(arr_1d[i:i + look_back])
        y.append(arr_1d[i + look_back])
    X = np.array(X).reshape(-1, look_back, 1)
    y = np.array(y)
    return X, y


def preparar_datos_lstm(s_train: pd.Series, s_test: pd.Series, look_back: int,
                         transform_log: bool = True):
    """
    Prepara datos de entrenamiento/prueba para LSTM:
      1. (opcional) log1p para estabilizar varianza (igual que en Laboratorio 1)
      2. escalado min-max ajustado SOLO con entrenamiento
      3. ventanas supervisadas de tamaño `look_back`

    Para poder predecir los n_test meses de prueba de forma autoregresiva
    (walk-forward), se arma la secuencia continua train+test en la escala
    transformada, y las ventanas de prueba se construyen encadenando las
    predicciones del propio modelo (no se usan valores reales de prueba
    como insumo, para que la predicción sea una proyección genuina a futuro).
    """
    y_tr = s_train.values.astype(float)
    y_te = s_test.values.astype(float)

    if transform_log:
        y_tr_t = log1p(y_tr)
        y_te_t = log1p(y_te)
    else:
        y_tr_t = y_tr.copy()
        y_te_t = y_te.copy()

    escalador = EscaladorMinMax().fit(y_tr_t)
    y_tr_s = escalador.transform(y_tr_t)
    y_te_s = escalador.transform(y_te_t)  # solo para poder medir en la misma escala si se requiere

    X_train, y_train = crear_ventanas(y_tr_s, look_back)

    return {
        "X_train": X_train, "y_train": y_train,
        "y_tr_s": y_tr_s, "y_te_s": y_te_s,
        "escalador": escalador, "transform_log": transform_log,
        "look_back": look_back,
    }


def pronostico_recursivo(modelo, y_tr_s: np.ndarray, look_back: int, n_pasos: int) -> np.ndarray:
    """
    Pronóstico multi-paso "walk-forward" (recursivo/autoregresivo): usa la
    última ventana de entrenamiento para predecir el paso siguiente, lo
    agrega a la secuencia, y repite hasta cubrir n_pasos. Así se simula un
    pronóstico real a futuro (igual que SARIMA/Prophet/HW en el Lab 1, que
    tampoco ven el conjunto de prueba).
    """
    secuencia = list(y_tr_s[-look_back:])
    preds = []
    for _ in range(n_pasos):
        x = np.array(secuencia[-look_back:]).reshape(1, look_back, 1)
        yhat = float(modelo.predict(x, verbose=0)[0, 0])
        preds.append(yhat)
        secuencia.append(yhat)
    return np.array(preds)


def predicciones_a_escala_original(preds_s: np.ndarray, escalador: EscaladorMinMax,
                                    transform_log: bool, index) -> pd.Series:
    preds_t = escalador.inverse_transform(preds_s)
    if transform_log:
        preds = expm1(preds_t)
    else:
        preds = np.clip(preds_t, 0, None)
    return pd.Series(preds, index=index)


# ------------------------------------------------------------------
# Arquitecturas LSTM
# ------------------------------------------------------------------

def construir_modelo_lstm(look_back: int, unidades=(32,), dropout: float = 0.0,
                           bidireccional: bool = False, lr: float = 1e-3):
    """
    Construye un modelo LSTM configurable:
      - unidades: tupla con el número de neuronas por capa LSTM apilada
        (ej. (32,) = 1 capa; (64, 32) = 2 capas apiladas)
      - dropout: dropout aplicado después de cada capa LSTM
      - bidireccional: si True, envuelve cada capa en Bidirectional()
      - lr: tasa de aprendizaje del optimizador Adam
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    modelo = keras.Sequential()
    modelo.add(layers.Input(shape=(look_back, 1)))
    for i, u in enumerate(unidades):
        return_seq = i < len(unidades) - 1
        capa = layers.LSTM(u, return_sequences=return_seq)
        if bidireccional:
            capa = layers.Bidirectional(capa)
        modelo.add(capa)
        if dropout > 0:
            modelo.add(layers.Dropout(dropout))
    modelo.add(layers.Dense(1))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return modelo


def entrenar_modelo(modelo, X_train, y_train, epochs=200, batch_size=8,
                     val_split=0.15, paciencia=20, verbose=0):
    from tensorflow import keras
    callback = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=paciencia, restore_best_weights=True
    )
    hist = modelo.fit(
        X_train, y_train, epochs=epochs, batch_size=batch_size,
        validation_split=val_split, callbacks=[callback], verbose=verbose,
        shuffle=False,
    )
    return hist


# ------------------------------------------------------------------
# LSTM + características catch22 (ejercicio 2.14): arquitectura de dos
# ramas -- la secuencia entra a la(s) capa(s) LSTM, y el vector de 22
# características catch22 de la serie (estandarizado) entra por una rama
# densa aparte; ambas ramas se concatenan antes de la salida.
# ------------------------------------------------------------------

def construir_modelo_lstm_catch22(look_back: int, n_feats: int, unidades=(32,),
                                   dropout: float = 0.0, lr: float = 1e-3,
                                   unidades_densa: int = 8):
    from tensorflow import keras
    from tensorflow.keras import layers

    entrada_seq = keras.Input(shape=(look_back, 1), name="secuencia")
    x = entrada_seq
    for i, u in enumerate(unidades):
        return_seq = i < len(unidades) - 1
        x = layers.LSTM(u, return_sequences=return_seq)(x)
        if dropout > 0:
            x = layers.Dropout(dropout)(x)

    entrada_c22 = keras.Input(shape=(n_feats,), name="catch22")
    c = layers.Dense(unidades_densa, activation="relu")(entrada_c22)

    combinado = layers.Concatenate()([x, c])
    salida = layers.Dense(1)(combinado)

    modelo = keras.Model(inputs=[entrada_seq, entrada_c22], outputs=salida)
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return modelo


def entrenar_modelo_catch22(modelo, X_train, feats_train, y_train, epochs=200,
                             batch_size=8, val_split=0.15, paciencia=20, verbose=0):
    from tensorflow import keras
    callback = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=paciencia, restore_best_weights=True
    )
    hist = modelo.fit(
        [X_train, feats_train], y_train, epochs=epochs, batch_size=batch_size,
        validation_split=val_split, callbacks=[callback], verbose=verbose,
        shuffle=False,
    )
    return hist


def pronostico_recursivo_catch22(modelo, y_tr_s: np.ndarray, feats_vec: np.ndarray,
                                  look_back: int, n_pasos: int) -> np.ndarray:
    """Igual que pronostico_recursivo, pero pasando el vector catch22 (fijo,
    repetido en cada paso) como segunda entrada del modelo de dos ramas."""
    secuencia = list(y_tr_s[-look_back:])
    feats_batch = feats_vec.reshape(1, -1)
    preds = []
    for _ in range(n_pasos):
        x = np.array(secuencia[-look_back:]).reshape(1, look_back, 1)
        yhat = float(modelo.predict([x, feats_batch], verbose=0)[0, 0])
        preds.append(yhat)
        secuencia.append(yhat)
    return np.array(preds)


# ------------------------------------------------------------------
# Métricas (idénticas a modelado.py del Laboratorio 1, para comparar)
# ------------------------------------------------------------------

def metricas(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {"MAE": mae, "RMSE": rmse}


# ------------------------------------------------------------------
# Rutina de tuneo: entrena varias configuraciones y devuelve tabla ordenada
# ------------------------------------------------------------------

def tunear_lstm(nombre_serie: str, s_train: pd.Series, s_test: pd.Series,
                 configuraciones: list, transform_log: bool = True,
                 epochs: int = 200, batch_size: int = 8, verbose_entrenamiento: int = 0):
    """
    Entrena una lista de configuraciones LSTM (cada una un dict con las
    llaves: look_back, unidades, dropout, bidireccional, lr, nombre) sobre
    la misma serie, y evalúa cada una en el conjunto de prueba real con
    pronóstico recursivo. Devuelve (tabla_resultados, modelos_entrenados,
    predicciones_dict, datos_dict_por_config).
    """
    filas = []
    modelos = {}
    predicciones = {}
    datos_por_config = {}
    n_test = len(s_test)

    for cfg in configuraciones:
        fijar_semillas()
        look_back = cfg["look_back"]
        datos = preparar_datos_lstm(s_train, s_test, look_back, transform_log)
        modelo = construir_modelo_lstm(
            look_back=look_back, unidades=cfg["unidades"], dropout=cfg.get("dropout", 0.0),
            bidireccional=cfg.get("bidireccional", False), lr=cfg.get("lr", 1e-3),
        )
        hist = entrenar_modelo(
            modelo, datos["X_train"], datos["y_train"], epochs=epochs,
            batch_size=batch_size, verbose=verbose_entrenamiento,
        )
        preds_s = pronostico_recursivo(modelo, datos["y_tr_s"], look_back, n_test)
        preds = predicciones_a_escala_original(preds_s, datos["escalador"], transform_log, s_test.index)
        m = metricas(s_test.values, preds.values)

        val_loss_final = float(np.min(hist.history["val_loss"])) if "val_loss" in hist.history else np.nan
        n_epocas_reales = len(hist.history["loss"])

        filas.append({
            "modelo": cfg["nombre"], "look_back": look_back,
            "unidades": str(cfg["unidades"]), "dropout": cfg.get("dropout", 0.0),
            "bidireccional": cfg.get("bidireccional", False), "lr": cfg.get("lr", 1e-3),
            "epocas_entrenadas": n_epocas_reales, "val_loss": round(val_loss_final, 5),
            "MAE": round(m["MAE"], 1), "RMSE": round(m["RMSE"], 1),
        })
        modelos[cfg["nombre"]] = modelo
        predicciones[cfg["nombre"]] = preds
        datos_por_config[cfg["nombre"]] = datos

    tabla = pd.DataFrame(filas).sort_values("RMSE").reset_index(drop=True)
    print(f"=== {nombre_serie}: resultados de tuneo (ordenado por RMSE en prueba) ===")
    print(tabla.to_string(index=False))
    return tabla, modelos, predicciones, datos_por_config


def graficar_comparacion(nombre_serie: str, s_train: pd.Series, s_test: pd.Series,
                          predicciones: dict, mejor_nombre: str, slug: str,
                          color: str, fig_dir=None):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    fig_dir = fig_dir or config.FIG_DIR
    MILES = FuncFormatter(config.miles)
    P = config.PALETTE

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(s_train.index, s_train.values, color=P["gris"], label="Entrenamiento")
    ax.plot(s_test.index, s_test.values, color=color, label="Prueba (real)")
    estilos = ["--", ":", "-.", "--"]
    for i, (nombre, pred) in enumerate(predicciones.items()):
        estilo = "-" if nombre == mejor_nombre else estilos[i % len(estilos)]
        ancho = 2.4 if nombre == mejor_nombre else 1.3
        ax.plot(s_test.index, pred.values, estilo, linewidth=ancho,
                color=P["rojo"] if nombre == mejor_nombre else None,
                label=f"{nombre}" + (" (mejor)" if nombre == mejor_nombre else ""))
    ax.yaxis.set_major_formatter(MILES)
    ax.set_title(f"{nombre_serie} — modelos LSTM vs. valores reales de prueba")
    ax.set_xlabel("mes"); ax.set_ylabel("viajeros"); ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / f"lstm_{slug}_comparacion.png")
    plt.show()
