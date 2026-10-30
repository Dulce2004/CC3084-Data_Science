"""
Etapa 4 — ajuste del procedimiento y generación de estimaciones.

No se incorporan algoritmos nuevos: se reutilizan las funciones del Laboratorio 1
(`03_modelado_prediccion/src/modelado.py`). El procedimiento y la referencia se
eligen en la configuración; todos son deterministas, así que dos ejecuciones con
los mismos datos y la misma configuración producen el mismo resultado.

Los parámetros se calculan solo con el tramo de ajuste: el procedimiento nunca
ve los meses evaluados.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "01_limpieza_eda" / "src"))
sys.path.append(str(ROOT / "03_modelado_prediccion" / "src"))
import modelado as mod  # noqa: E402


def fijar_semilla(semilla: int) -> None:
    """Fija la semilla global. Los modelos usados son deterministas; queda por si
    se añade un procedimiento con componentes aleatorios."""
    import random
    random.seed(semilla)
    np.random.seed(semilla % (2 ** 32 - 1))


# ------------------------------------------------------------------ procedimientos
def _holt_winters(ajuste: pd.Series, n: int, index, periodo: int):
    pred, res = mod.modelo_holt_winters(ajuste, n, index, seasonal_periods=periodo)
    params = {k: float(v) for k, v in res.params.items()
              if isinstance(v, (int, float, np.floating))}
    detalle = f"tendencia y estacionalidad aditivas, período {periodo}"
    return pred, params, detalle


def _ses(ajuste: pd.Series, n: int, index, periodo: int):
    pred, res = mod.modelo_ses(ajuste, n, index)
    params = {k: float(v) for k, v in res.params.items()
              if isinstance(v, (int, float, np.floating))}
    return pred, params, "suavizamiento exponencial simple"


def _seasonal_naive(ajuste: pd.Series, n: int, index, periodo: int):
    pred = mod.modelo_seasonal_naive(ajuste, n, index, period=periodo)
    return pred, {"period": periodo}, f"repite el último ciclo de {periodo} meses"


PROCEDIMIENTOS = {
    "holt_winters": _holt_winters,
    "ses": _ses,
    "seasonal_naive": _seasonal_naive,
}


# ------------------------------------------------------------------ referencias
def _ultimo_valor(ajuste: pd.Series, n: int, index, periodo: int) -> pd.Series:
    """Referencia ingenua: repite el último valor observado antes del período evaluado."""
    return pd.Series(np.repeat(float(ajuste.iloc[-1]), n), index=index)


def _promedio_historico(ajuste: pd.Series, n: int, index, periodo: int) -> pd.Series:
    """
    Referencia histórica: promedio del mismo mes calendario en el tramo de ajuste.
    Usa solo datos anteriores al período evaluado.
    """
    por_mes = ajuste.groupby(ajuste.index.month).mean()
    valores = [float(por_mes.get(m, ajuste.mean())) for m in index.month]
    return pd.Series(valores, index=index)


REFERENCIAS = {
    "ultimo_valor": _ultimo_valor,
    "promedio_historico": _promedio_historico,
    "seasonal_naive": lambda a, n, i, p: mod.modelo_seasonal_naive(a, n, i, period=p),
}


def ajustar_y_estimar(ajuste: pd.Series, meses: pd.DatetimeIndex, cfg) -> dict:
    """
    Ajusta el procedimiento configurado sobre `ajuste` y estima `meses`.

    Devuelve la estimación del procedimiento, la de la referencia, los parámetros
    calculados y un detalle legible de la configuración usada.
    """
    p = cfg["procedimiento"]
    periodo = int(p.get("periodo_estacional", 12))
    n = len(meses)

    if ajuste.isna().any():
        faltan = [str(d.date()) for d in ajuste.index[ajuste.isna()]]
        raise ValueError("el tramo de ajuste tiene meses sin valor observado y no se "
                         "rellenan con cero: " + ", ".join(faltan[:12]))
    if n == 0:
        raise ValueError("el período de evaluación está vacío")

    estimar = PROCEDIMIENTOS[p["modelo"]]
    pred, params, detalle = estimar(ajuste, n, meses, periodo)
    referencia = REFERENCIAS[p["referencia"]](ajuste, n, meses, periodo)

    return {
        "modelo": p["modelo"],
        "detalle": detalle,
        "parametros": params,
        "estimacion": pred.clip(lower=0).rename("estimacion"),
        "referencia_nombre": p["referencia"],
        "referencia": pd.Series(referencia, index=meses).clip(lower=0).rename("referencia"),
        "n_ajuste": int(len(ajuste)),
        "ajuste_inicio": str(ajuste.index.min().date()),
        "ajuste_fin": str(ajuste.index.max().date()),
    }
