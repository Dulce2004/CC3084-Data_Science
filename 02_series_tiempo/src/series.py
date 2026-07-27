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


def kpss_test(serie: pd.Series) -> dict:
    """Prueba KPSS. H0: la serie ES estacionaria (contraria a la de ADF)."""
    from statsmodels.tsa.stattools import kpss
    import warnings
    s = serie.dropna()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, pval, lags, crit = kpss(s, regression="c", nlags="auto")
    return {"kpss_stat": stat, "p_value": pval, "lags": lags,
             "estacionaria_5%": pval > 0.05}


def fuerza_estacional_tendencia(s_tr: pd.Series) -> dict:
    """
    Fuerza de tendencia y estacionalidad (Hyndman & Athanasopoulos, fpp2 6.7),
    a partir de la descomposición STL: F = max(0, 1 - Var(resid)/Var(resid+componente)).
    """
    from statsmodels.tsa.seasonal import STL
    s = s_tr.copy()
    if (s <= 0).any():
        s = s - s.min() + 1.0  # STL requiere serie sin ceros/negativos para log interno opcional
    stl = STL(s, period=12, robust=True).fit()
    var_resid = np.var(stl.resid)
    f_tend = max(0.0, 1 - var_resid / np.var(stl.trend + stl.resid))
    f_estac = max(0.0, 1 - var_resid / np.var(stl.seasonal + stl.resid))
    return {"fuerza_tendencia": f_tend, "fuerza_estacional": f_estac}


def analiza_serie(nombre: str, s: pd.Series, slug: str, color: str, fig_dir=None):
    """
    Grafica serie, descomposición, ACF/PACF y ADF/KPSS (sobre entrenamiento).
    Devuelve (adf_niveles, adf_diff1, kpss_niveles) para uso posterior.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    fig_dir = fig_dir or config.FIG_DIR
    P = config.PALETTE
    MILES = FuncFormatter(config.miles)

    s_tr, s_te = split_temporal(s)
    print(f'=== {nombre} ===')
    print('Inicio:', s_tr.index.min().date(), '| Fin:', s_tr.index.max().date(),
          '| Frecuencia: Mensual (MS) | n =', len(s_tr))

    adf0 = adf_test(s_tr)
    adf1 = adf_test(s_tr.diff().dropna())
    kpss0 = kpss_test(s_tr)
    print(f"ADF niveles      : stat={adf0['adf_stat']:.2f}  p={adf0['p_value']:.3f}  -> "
          f"{'estacionaria' if adf0['estacionaria_5%'] else 'NO estacionaria'} (5%)")
    print(f"ADF 1a diferencia: stat={adf1['adf_stat']:.2f}  p={adf1['p_value']:.3f}  -> "
          f"{'estacionaria' if adf1['estacionaria_5%'] else 'NO estacionaria'} (5%)")
    print(f"KPSS niveles     : stat={kpss0['kpss_stat']:.3f}  p={kpss0['p_value']:.3f}  -> "
          f"{'estacionaria' if kpss0['estacionaria_5%'] else 'NO estacionaria'} (5%)")

    modelo = 'multiplicative' if (s_tr > 0).all() else 'additive'
    dec = seasonal_decompose(s_tr, model=modelo, period=12)
    fig, axs = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
    axs[0].plot(s_tr.index, s_tr.values, color=color); axs[0].set_ylabel('serie')
    axs[0].set_title(f'{nombre} — descomposición {modelo} (período 12)')
    axs[1].plot(dec.trend.index, dec.trend.values, color=P['azul']); axs[1].set_ylabel('tendencia')
    axs[2].plot(dec.seasonal.index, dec.seasonal.values, color=P['verde']); axs[2].set_ylabel('estacional')
    axs[3].scatter(dec.resid.index, dec.resid.values, s=10, color=P['gris']); axs[3].set_ylabel('residuo')
    axs[0].yaxis.set_major_formatter(MILES); axs[1].yaxis.set_major_formatter(MILES)
    if modelo == 'additive':
        axs[2].yaxis.set_major_formatter(MILES); axs[3].yaxis.set_major_formatter(MILES)
    axs[3].set_xlabel('mes')
    plt.tight_layout(); plt.savefig(fig_dir / f'ser_{slug}_decomp.png'); plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(12, 3.4))
    plot_acf(s_tr, lags=min(36, len(s_tr)//2 - 1), ax=ax[0], color=color)
    plot_pacf(s_tr, lags=min(36, len(s_tr)//2 - 1), ax=ax[1], method='ywm', color=color)
    ax[0].set_title(f'ACF — {nombre}'); ax[1].set_title(f'PACF — {nombre}')
    plt.tight_layout(); plt.savefig(fig_dir / f'ser_{slug}_acf.png'); plt.show()
    return adf0, adf1, kpss0
