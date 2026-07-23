"""
Construcción de series de tiempo mensuales y utilidades de análisis
(descomposición, ACF/PACF, prueba de Dickey-Fuller aumentada).

Reutiliza rutas/estilo definidos en 01_limpieza_eda/src/config.py.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "01_limpieza_eda" / "src"))
import config  # noqa: E402


def cargar_limpio() -> pd.DataFrame:
    """Carga el CSV limpio con la columna `fecha` como datetime."""
    df = pd.read_csv(config.CLEAN_CSV, parse_dates=["fecha"])
    return df


def serie_mensual(df: pd.DataFrame, columna: str = None, valor=None) -> pd.Series:
    """
    Serie mensual de viajeros (suma de `Viajero`) con frecuencia mensual (MS).

    - Sin columna/valor -> total mensual de todos los viajeros.
    - Con columna y valor -> total mensual filtrando `columna == valor`.
    """
    d = df
    if columna is not None and valor is not None:
        d = d[d[columna] == valor]
    s = (d.groupby("fecha")["Viajero"].sum()
           .asfreq("MS")            # frecuencia mensual explícita, sin huecos
           .fillna(0.0))
    s.name = valor if valor is not None else "Total"
    return s


def split_temporal(serie: pd.Series, frac: float = config.TRAIN_FRAC):
    """Divide una serie temporal en entrenamiento/prueba respetando el orden."""
    n = len(serie)
    k = int(round(n * frac))
    return serie.iloc[:k], serie.iloc[k:]


def resumen_serie(serie: pd.Series) -> dict:
    """Inicio, fin, frecuencia y número de observaciones."""
    return {
        "inicio": serie.index.min().date(),
        "fin": serie.index.max().date(),
        "frecuencia": "Mensual (MS)",
        "n_obs": len(serie),
    }


def adf_test(serie: pd.Series) -> dict:
    """Prueba de Dickey-Fuller aumentada. H0: existe raíz unitaria (no estacionaria)."""
    from statsmodels.tsa.stattools import adfuller
    s = serie.dropna()
    stat, pval, lags, nobs, crit, _ = adfuller(s, autolag="AIC")
    return {
        "adf_stat": stat,
        "p_value": pval,
        "lags": lags,
        "n": nobs,
        "crit_5%": crit["5%"],
        "estacionaria_5%": pval < 0.05,
    }
