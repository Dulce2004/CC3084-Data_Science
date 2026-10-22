"""
Ajuste y comparación de modelos de pronóstico para series de tiempo mensuales:
ARIMA/SARIMA (grid search por AIC), Prophet, Holt-Winters, suavizamiento
exponencial simple y seasonal naive. Incluye métricas de error y diagnóstico
de residuos (Ljung-Box).

Reutiliza rutas/estilo definidos en 01_limpieza_eda/src/config.py.
"""
import sys
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "01_limpieza_eda" / "src"))
import config  # noqa: E402

logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)


# ------------------------------------------------------------------
# Transformación y métricas
# ------------------------------------------------------------------

def log1p(s: pd.Series) -> pd.Series:
    """log(1+x): estabiliza varianza y admite ceros (meses sin viajeros)."""
    return np.log1p(s)


def expm1(s):
    """Inversa de log1p; recorta a 0 (no hay viajeros negativos)."""
    return np.expm1(s).clip(lower=0)


def metricas(y_true: pd.Series, y_pred: pd.Series) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {"MAE": mae, "RMSE": rmse}


# ------------------------------------------------------------------
# ARIMA / SARIMA (ajustados en escala log1p)
# ------------------------------------------------------------------

def ajustar_sarima(s_train_log: pd.Series, order: tuple, seasonal_order: tuple):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelo = SARIMAX(s_train_log, order=order, seasonal_order=seasonal_order,
                          enforce_stationarity=False, enforce_invertibility=False)
        res = modelo.fit(disp=False)
    return res


def buscar_sarima(s_train_log: pd.Series, ordenes, ordenes_estacionales, s: int = 12):
    """
    Grid search por AIC sobre combinaciones (p,d,q) x (P,D,Q,s).
    Funciona como equivalente casero de auto_arima/auto.arima: prueba todas las
    combinaciones y deja que el AIC elija, en vez de proponer una sola por criterio experto.
    Devuelve (tabla ordenada por AIC, diccionario {clave: resultado ajustado}).
    """
    filas = []
    modelos = {}
    for order in ordenes:
        for so in ordenes_estacionales:
            seasonal_order = so + (s,)
            try:
                res = ajustar_sarima(s_train_log, order, seasonal_order)
            except Exception:
                continue
            if not np.isfinite(res.aic):
                continue
            clave = f"SARIMA{order}x{seasonal_order}"
            modelos[clave] = res
            filas.append({"modelo": clave, "order": order, "seasonal_order": seasonal_order,
                           "AIC": res.aic, "BIC": res.bic})
    tabla = pd.DataFrame(filas).sort_values("AIC").reset_index(drop=True)
    return tabla, modelos


def predecir_sarima(res, n: int, index) -> pd.Series:
    """Pronóstico de un SARIMAX ajustado en log1p, devuelto en la escala original."""
    pred_log = res.get_forecast(steps=n).predicted_mean
    pred = expm1(pred_log)
    pred.index = index
    return pred


def ljung_box(res, lags=(6, 12, 24)) -> pd.DataFrame:
    from statsmodels.stats.diagnostic import acorr_ljungbox
    tabla = acorr_ljungbox(res.resid, lags=list(lags), return_df=True)
    tabla.index.name = "lag"
    return tabla


# ------------------------------------------------------------------
# Modelos de comparación (escala original, sin transformar)
# ------------------------------------------------------------------

def modelo_holt_winters(s_train: pd.Series, n: int, index, seasonal_periods: int = 12):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelo = ExponentialSmoothing(s_train, trend="add", seasonal="add",
                                       seasonal_periods=seasonal_periods,
                                       initialization_method="estimated")
        res = modelo.fit()
        pred = res.forecast(n)
    pred.index = index
    return pred.clip(lower=0), res


def modelo_ses(s_train: pd.Series, n: int, index):
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelo = SimpleExpSmoothing(s_train, initialization_method="estimated")
        res = modelo.fit()
        pred = res.forecast(n)
    pred.index = index
    return pred.clip(lower=0), res


def modelo_seasonal_naive(s_train: pd.Series, n: int, index, period: int = 12) -> pd.Series:
    """y_hat(t) = y(t - period), repitiendo el último ciclo estacional completo observado."""
    ultimo_ciclo = s_train.iloc[-period:].values
    reps = int(np.ceil(n / period))
    valores = np.tile(ultimo_ciclo, reps)[:n]
    return pd.Series(valores, index=index)


def modelo_prophet(s_train: pd.Series, n: int, index):
    """Ajusta Prophet silenciando el log de cmdstanpy (no aporta al análisis)."""
    import io
    import contextlib
    from prophet import Prophet
    dfp = pd.DataFrame({"ds": s_train.index, "y": s_train.values})
    buf = io.StringIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
            m.fit(dfp)
            futuro = pd.DataFrame({"ds": index})
            fcst = m.predict(futuro)
    pred = pd.Series(fcst["yhat"].values, index=index).clip(lower=0)
    return pred, m


# ------------------------------------------------------------------
# Tabla comparativa final
# ------------------------------------------------------------------

def tabla_comparacion(predicciones: dict, y_test: pd.Series, info_extra: dict = None) -> pd.DataFrame:
    """
    predicciones : {nombre_modelo: serie_predicha}
    info_extra   : {nombre_modelo: {'AIC': ..., 'BIC': ...}} (solo aplica a ARIMA/SARIMA)
    """
    info_extra = info_extra or {}
    filas = []
    for nombre, pred in predicciones.items():
        m = metricas(y_test, pred.reindex(y_test.index))
        extra = info_extra.get(nombre, {})
        filas.append({"modelo": nombre, "MAE": round(m["MAE"], 1), "RMSE": round(m["RMSE"], 1),
                      "AIC": round(extra["AIC"], 1) if "AIC" in extra else np.nan,
                      "BIC": round(extra["BIC"], 1) if "BIC" in extra else np.nan})
    return pd.DataFrame(filas).sort_values("RMSE").reset_index(drop=True)
