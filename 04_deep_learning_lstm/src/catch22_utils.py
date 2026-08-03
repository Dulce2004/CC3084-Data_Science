"""
Reimplementación en Python puro (numpy/scipy) de las 22 características de
*catch22* (Lubba et al., 2019, "catch22: CAnonical Time-series CHaracteristics").

Nota de reproducibilidad: el paquete oficial `pycatch22` solo distribuye código
fuente en PyPI (no hay wheels precompiladas para Windows) y requiere compilar
una extensión en C, lo que exige Microsoft C++ Build Tools. Esta máquina no
tiene un compilador de C/C++ instalado (ni MSVC ni gcc/mingw), así que en vez
de bloquear el laboratorio se reimplementaron las 22 características siguiendo
las definiciones publicadas en el paper original y en el catálogo de hctsa
(Fulcher & Jones, 2017), usando únicamente numpy/scipy. Los valores no son
bit-a-bit idénticos a la librería oficial en C, pero capturan la misma idea
estadística de cada característica (forma de la distribución, autocorrelación,
periodicidad, entropía simbólica, fluctuación/self-affinity, espectro, etc.).

Cada función recibe un arreglo 1D (float) y devuelve un escalar float.
`catch22_all(x)` devuelve un diccionario con las 22 características, en el
mismo orden en que aparecen en el paper.
"""
import numpy as np
from scipy import signal, stats


def _z(x):
    x = np.asarray(x, dtype=float)
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else x - x.mean()


def _acf(x, nlags):
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    nlags = min(nlags, n - 1)
    full = np.correlate(x, x, mode="full")
    mid = len(full) // 2
    ac = full[mid:mid + nlags + 1]
    return ac / ac[0] if ac[0] != 0 else np.zeros_like(ac)


# ------------------------------------------------------------------
# 1-2. Distribución (histograma z-scoreado)
# ------------------------------------------------------------------

def DN_HistogramMode_5(x):
    z = _z(x)
    counts, edges = np.histogram(z, bins=5)
    i = np.argmax(counts)
    return float((edges[i] + edges[i + 1]) / 2)


def DN_HistogramMode_10(x):
    z = _z(x)
    counts, edges = np.histogram(z, bins=10)
    i = np.argmax(counts)
    return float((edges[i] + edges[i + 1]) / 2)


# ------------------------------------------------------------------
# 3-4. Autocorrelación lineal
# ------------------------------------------------------------------

def CO_f1ecac(x):
    """Primer lag en que la ACF cae por debajo de 1/e."""
    ac = _acf(x, min(len(x) - 1, 100))
    umbral = 1 / np.e
    bajo = np.where(ac < umbral)[0]
    return float(bajo[0]) if len(bajo) else float(len(ac) - 1)


def CO_FirstMin_ac(x):
    """Primer mínimo local de la función de autocorrelación."""
    ac = _acf(x, min(len(x) - 1, 100))
    for i in range(1, len(ac) - 1):
        if ac[i] < ac[i - 1] and ac[i] < ac[i + 1]:
            return float(i)
    return float(len(ac) - 1)


# ------------------------------------------------------------------
# 5. Información mutua (histograma) a lag 2
# ------------------------------------------------------------------

def CO_HistogramAMI_even_2_5(x, tau=2, bins=5):
    x = np.asarray(x, dtype=float)
    if len(x) <= tau:
        return 0.0
    a, b = x[:-tau], x[tau:]
    c_xy, xe, ye = np.histogram2d(a, b, bins=bins)
    pxy = c_xy / c_xy.sum()
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = pxy / (px * py)
        terms = pxy * np.log(ratio)
    terms[~np.isfinite(terms)] = 0.0
    return float(terms.sum())


# ------------------------------------------------------------------
# 6. Reversibilidad temporal
# ------------------------------------------------------------------

def CO_trev_1_num(x):
    d = np.diff(np.asarray(x, dtype=float))
    denom = np.mean(d ** 2) ** 1.5
    return float(np.mean(d ** 3) / denom) if denom > 0 else 0.0


# ------------------------------------------------------------------
# 7. pNN40 (proporción de saltos grandes, adaptado de HRV a z-scores)
# ------------------------------------------------------------------

def MD_hrv_classic_pnn40(x):
    z = _z(x)
    d = np.abs(np.diff(z))
    return float(np.mean(d > 0.04))


# ------------------------------------------------------------------
# 8-9. Estadísticos binarios / simbólicos
# ------------------------------------------------------------------

def _racha_mas_larga(binaria, valor):
    mejor = actual = 0
    for b in binaria:
        if b == valor:
            actual += 1
            mejor = max(mejor, actual)
        else:
            actual = 0
    return mejor


def SB_BinaryStats_mean_longstretch1(x):
    x = np.asarray(x, dtype=float)
    binaria = (x > x.mean()).astype(int)
    return float(_racha_mas_larga(binaria, 1))


def SB_TransitionMatrix_3ac_sumdiagcov(x):
    x = np.asarray(x, dtype=float)
    try:
        simbolos = np.array(list(map(int, _qcut_labels(x, 3))))
    except Exception:
        return 0.0
    P = np.zeros((3, 3))
    for i in range(len(simbolos) - 1):
        P[simbolos[i], simbolos[i + 1]] += 1
    filas = P.sum(axis=1, keepdims=True)
    filas[filas == 0] = 1
    P = P / filas
    cov = np.cov(P)
    return float(np.trace(cov))


def _qcut_labels(x, q):
    """Discretiza x en q símbolos por cuantiles, sin depender de pandas.qcut."""
    ranks = stats.rankdata(x, method="average")
    return np.minimum((ranks / (len(x) + 1) * q).astype(int), q - 1)


# ------------------------------------------------------------------
# 10. Periodicidad (Wang et al.)
# ------------------------------------------------------------------

def PD_PeriodicityWang_th0_01(x):
    x = np.asarray(x, dtype=float)
    detrended = signal.detrend(x, type="linear")
    ac = _acf(detrended, min(len(x) - 1, 100))
    umbral = 0.01
    for i in range(1, len(ac) - 1):
        if ac[i] > ac[i - 1] and ac[i] > ac[i + 1] and ac[i] > umbral:
            return float(i)
    return 0.0


# ------------------------------------------------------------------
# 11. Embedding 2D, distancias vs. ajuste exponencial
# ------------------------------------------------------------------

def CO_Embed2_Dist_tau_d_expfit_meandiff(x):
    x = np.asarray(x, dtype=float)
    tau = max(1, int(CO_FirstMin_ac(x)))
    if len(x) - tau < 3:
        return 0.0
    emb = np.column_stack([x[:-tau], x[tau:]])
    d = np.sqrt(np.sum(np.diff(emb, axis=0) ** 2, axis=1))
    if d.mean() == 0:
        return 0.0
    lam = 1.0 / d.mean()
    counts, edges = np.histogram(d, bins=10, density=True)
    centros = (edges[:-1] + edges[1:]) / 2
    esperado = lam * np.exp(-lam * centros)
    return float(np.mean(np.abs(counts - esperado)))


# ------------------------------------------------------------------
# 12. Información mutua automática (estimador gaussiano), primer mínimo
# ------------------------------------------------------------------

def IN_AutoMutualInfoStats_40_gaussian_fmmi(x):
    x = np.asarray(x, dtype=float)
    maxlag = min(40, len(x) // 2 - 1)
    if maxlag < 2:
        return 0.0
    amis = []
    for lag in range(1, maxlag + 1):
        r = np.corrcoef(x[:-lag], x[lag:])[0, 1]
        r = np.clip(r, -0.999999, 0.999999)
        amis.append(-0.5 * np.log(1 - r ** 2))
    for i in range(1, len(amis) - 1):
        if amis[i] < amis[i - 1] and amis[i] < amis[i + 1]:
            return float(i + 1)
    return float(maxlag)


# ------------------------------------------------------------------
# 13, 22. Predictor local (media móvil corta) — residuales
# ------------------------------------------------------------------

def _residuales_media_local(x, ventana):
    x = np.asarray(x, dtype=float)
    preds = np.array([x[i - ventana:i].mean() for i in range(ventana, len(x))])
    reales = x[ventana:]
    return reales - preds


def FC_LocalSimple_mean1_tauresrat(x):
    x = np.asarray(x, dtype=float)
    res = _residuales_media_local(x, 1)
    ac_res = _acf(res, min(len(res) - 1, 50))
    ac_orig = _acf(x, min(len(x) - 1, 50))

    def primer_cruce_cero(ac):
        bajo = np.where(ac < 0)[0]
        return float(bajo[0]) if len(bajo) else float(len(ac))

    denom = primer_cruce_cero(ac_orig)
    return float(primer_cruce_cero(ac_res) / denom) if denom > 0 else 0.0


def FC_LocalSimple_mean3_stderr(x):
    res = _residuales_media_local(x, 3)
    return float(res.std())


# ------------------------------------------------------------------
# 14-15. Outliers positivos / negativos: ¿cuándo ocurren en el tiempo?
# ------------------------------------------------------------------

def _outlier_include_mdrmd(x, signo):
    z = _z(x) * signo
    n = len(z)
    maxval = z.max()
    if maxval <= 0:
        return 0.0
    pasos = np.arange(0, maxval, 0.01 * maxval if maxval > 0 else 1)
    medianas = []
    for umbral in pasos:
        idx = np.where(z >= umbral)[0]
        if len(idx) == 0:
            continue
        medianas.append(np.median(idx) / n)
    return float(np.mean(medianas)) if medianas else 0.0


def DN_OutlierInclude_p_001_mdrmd(x):
    return _outlier_include_mdrmd(x, signo=1)


def DN_OutlierInclude_n_001_mdrmd(x):
    return _outlier_include_mdrmd(x, signo=-1)


# ------------------------------------------------------------------
# 16, 21. Resumen espectral (Welch)
# ------------------------------------------------------------------

def _welch(x):
    x = np.asarray(x, dtype=float)
    fs = 1.0
    nperseg = min(len(x), 64)
    f, pxx = signal.welch(x, fs=fs, window="boxcar", nperseg=nperseg)
    return f, pxx


def SP_Summaries_welch_rect_area_5_1(x):
    f, pxx = _welch(x)
    total = pxx.sum()
    if total == 0:
        return 0.0
    corte = len(pxx) // 5
    corte = max(corte, 1)
    return float(pxx[:corte].sum() / total)


def SP_Summaries_welch_rect_centroid(x):
    f, pxx = _welch(x)
    total = pxx.sum()
    return float((f * pxx).sum() / total) if total > 0 else 0.0


# ------------------------------------------------------------------
# 17. Racha más larga sin decrecer (diferencias)
# ------------------------------------------------------------------

def SB_BinaryStats_diff_longstretch0(x):
    d = np.diff(np.asarray(x, dtype=float))
    binaria = (d >= 0).astype(int)
    return float(_racha_mas_larga(binaria, 0))


# ------------------------------------------------------------------
# 18. Entropía de motivos de 3 símbolos (cuantiles)
# ------------------------------------------------------------------

def SB_MotifThree_quantile_hh(x):
    x = np.asarray(x, dtype=float)
    simbolos = _qcut_labels(x, 3)
    motivos = {}
    for i in range(len(simbolos) - 2):
        m = tuple(simbolos[i:i + 3])
        motivos[m] = motivos.get(m, 0) + 1
    total = sum(motivos.values())
    if total == 0:
        return 0.0
    probs = np.array([c / total for c in motivos.values()])
    ent = -np.sum(probs * np.log2(probs))
    return float(ent)


# ------------------------------------------------------------------
# 19-20. Análisis de fluctuación (self-affinity): R/S y DFA
# ------------------------------------------------------------------

def _fluct_slope(x, metodo="rs"):
    x = np.asarray(x, dtype=float)
    n = len(x)
    perfil = np.cumsum(x - x.mean())
    tam_min, tam_max = 8, max(9, n // 2)
    tamanos = np.unique(np.geomspace(tam_min, tam_max, num=8).astype(int))
    logs_n, logs_f = [], []
    for s in tamanos:
        if s < 4 or s >= n:
            continue
        nseg = n // s
        if nseg < 2:
            continue
        vals = []
        for i in range(nseg):
            seg = perfil[i * s:(i + 1) * s]
            t = np.arange(s)
            coef = np.polyfit(t, seg, 1)
            tendencia = np.polyval(coef, t)
            detr = seg - tendencia
            if metodo == "dfa":
                vals.append(np.sqrt(np.mean(detr ** 2)))
            else:  # rango reescalado (R/S) sobre el segmento detrendado
                r = detr.max() - detr.min()
                sd = detr.std()
                vals.append(r / sd if sd > 0 else 0.0)
        vals = [v for v in vals if v > 0]
        if not vals:
            continue
        logs_n.append(np.log(s))
        logs_f.append(np.log(np.mean(vals)))
    if len(logs_n) < 2:
        return 0.5
    pend, _ = np.polyfit(logs_n, logs_f, 1)
    return float(pend)


def SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1(x):
    """Exponente de Hurst estimado por rango reescalado (R/S) log-log."""
    return _fluct_slope(x, metodo="rs")


def SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1(x):
    """Exponente alpha de detrended fluctuation analysis (DFA)."""
    return _fluct_slope(x, metodo="dfa")


# ------------------------------------------------------------------
# Diccionario maestro: las 22 características, en el orden del paper
# ------------------------------------------------------------------

NOMBRES_CATCH22 = [
    "DN_HistogramMode_5", "DN_HistogramMode_10", "CO_f1ecac", "CO_FirstMin_ac",
    "CO_HistogramAMI_even_2_5", "CO_trev_1_num", "MD_hrv_classic_pnn40",
    "SB_BinaryStats_mean_longstretch1", "SB_TransitionMatrix_3ac_sumdiagcov",
    "PD_PeriodicityWang_th0_01", "CO_Embed2_Dist_tau_d_expfit_meandiff",
    "IN_AutoMutualInfoStats_40_gaussian_fmmi", "FC_LocalSimple_mean1_tauresrat",
    "DN_OutlierInclude_p_001_mdrmd", "DN_OutlierInclude_n_001_mdrmd",
    "SP_Summaries_welch_rect_area_5_1", "SB_BinaryStats_diff_longstretch0",
    "SB_MotifThree_quantile_hh", "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1",
    "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1", "SP_Summaries_welch_rect_centroid",
    "FC_LocalSimple_mean3_stderr",
]

_FUNCIONES = {
    "DN_HistogramMode_5": DN_HistogramMode_5,
    "DN_HistogramMode_10": DN_HistogramMode_10,
    "CO_f1ecac": CO_f1ecac,
    "CO_FirstMin_ac": CO_FirstMin_ac,
    "CO_HistogramAMI_even_2_5": CO_HistogramAMI_even_2_5,
    "CO_trev_1_num": CO_trev_1_num,
    "MD_hrv_classic_pnn40": MD_hrv_classic_pnn40,
    "SB_BinaryStats_mean_longstretch1": SB_BinaryStats_mean_longstretch1,
    "SB_TransitionMatrix_3ac_sumdiagcov": SB_TransitionMatrix_3ac_sumdiagcov,
    "PD_PeriodicityWang_th0_01": PD_PeriodicityWang_th0_01,
    "CO_Embed2_Dist_tau_d_expfit_meandiff": CO_Embed2_Dist_tau_d_expfit_meandiff,
    "IN_AutoMutualInfoStats_40_gaussian_fmmi": IN_AutoMutualInfoStats_40_gaussian_fmmi,
    "FC_LocalSimple_mean1_tauresrat": FC_LocalSimple_mean1_tauresrat,
    "DN_OutlierInclude_p_001_mdrmd": DN_OutlierInclude_p_001_mdrmd,
    "DN_OutlierInclude_n_001_mdrmd": DN_OutlierInclude_n_001_mdrmd,
    "SP_Summaries_welch_rect_area_5_1": SP_Summaries_welch_rect_area_5_1,
    "SB_BinaryStats_diff_longstretch0": SB_BinaryStats_diff_longstretch0,
    "SB_MotifThree_quantile_hh": SB_MotifThree_quantile_hh,
    "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1": SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1,
    "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1": SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1,
    "SP_Summaries_welch_rect_centroid": SP_Summaries_welch_rect_centroid,
    "FC_LocalSimple_mean3_stderr": FC_LocalSimple_mean3_stderr,
}


def catch22_all(x) -> dict:
    """Calcula las 22 características de catch22 para una serie 1D."""
    x = np.asarray(x, dtype=float)
    salida = {}
    for nombre in NOMBRES_CATCH22:
        try:
            salida[nombre] = _FUNCIONES[nombre](x)
        except Exception:
            salida[nombre] = np.nan
    return salida
