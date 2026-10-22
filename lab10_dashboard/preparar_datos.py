"""
Lab 10 · Prepara los datos del tablero a partir del código del Laboratorio 1.

No entrena modelos nuevos: reproduce, con las mismas funciones de
03_modelado_prediccion/src/modelado.py y el mismo split 70/30 del Lab 1,
los modelos clásicos ya evaluados (SARIMA manual, Holt-Winters, Prophet,
SES y seasonal naive) y guarda sus predicciones y métricas en CSV para que
el tablero cargue rápido.

Uso (desde la raíz del repo, con data/processed/migracion_limpia.csv generado):
    python 01_limpieza_eda/src/limpieza.py
    python lab10_dashboard/preparar_datos.py
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
for p in ("01_limpieza_eda/src", "02_series_tiempo/src", "03_modelado_prediccion/src"):
    sys.path.insert(0, str(ROOT / p))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import series  # noqa: E402
import modelado as mod  # noqa: E402

OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(exist_ok=True)

# Series modeladas en el Lab 1 que se muestran en el tablero
SERIES = {
    "Total": (None, None),
    "Vía Aérea": ("Vía", "Aérea"),
    "Vía Terrestre": ("Vía", "Terrestre"),
    "Vía Marítima": ("Vía", "Marítima"),
    "Turista": ("Tipo de Viajero", "Turista"),
    "Excursionista": ("Tipo de Viajero", "Excursionista"),
}


def agregado(df: pd.DataFrame) -> pd.DataFrame:
    """Suma mensual por las dimensiones que usa el tablero (reduce 161k filas a ~20k)."""
    dims = ["fecha", "Vía", "Tipo de Viajero", "Región dos", "Frontera"]
    return df.groupby(dims, as_index=False)["Viajero"].sum()


def modelos_serie(nombre: str, s: pd.Series):
    tr, te = series.split_temporal(s)
    idx = te.index
    D = 1 if series.fuerza_estacional_tendencia(tr)["fuerza_estacional"] > 0.3 else 0

    preds, bandas = {}, {}
    res = mod.ajustar_sarima(mod.log1p(tr), (1, 1, 1), (1, D, 1, 12))
    preds[f"SARIMA(1,1,1)(1,{D},1)"] = mod.predecir_sarima(res, len(te), idx)
    preds["Holt-Winters"], _ = mod.modelo_holt_winters(tr, len(te), idx)
    p, m = mod.modelo_prophet(tr, len(te), idx)
    preds["Prophet"] = p
    fc = m.predict(pd.DataFrame({"ds": idx}))
    bandas["Prophet"] = (fc["yhat_lower"].clip(lower=0).values, fc["yhat_upper"].values)
    preds["Suavizamiento simple (SES)"], _ = mod.modelo_ses(tr, len(te), idx)
    preds["Ingenuo estacional"] = mod.modelo_seasonal_naive(tr, len(te), idx)

    filas_pred, filas_met = [], []
    for modelo, pr in preds.items():
        lo, hi = bandas.get(modelo, (np.full(len(te), np.nan), np.full(len(te), np.nan)))
        filas_pred.append(pd.DataFrame({
            "serie": nombre, "modelo": modelo, "fecha": idx,
            "real": te.values, "pred": pr.values, "lo": lo, "hi": hi,
        }))
        met = mod.metricas(te, pr)
        mape = float(np.mean(np.abs(te.values - pr.values) / np.maximum(te.values, 1))) * 100
        filas_met.append({"serie": nombre, "modelo": modelo, "MAE": met["MAE"],
                          "RMSE": met["RMSE"], "MAPE": mape,
                          "estable": bool(np.isfinite(pr.values).all() and pr.max() < 10 * s.max())})
    return pd.concat(filas_pred), filas_met, tr.index.max()


def main():
    df = series.cargar_limpio()
    agregado(df).to_csv(OUT / "agregado_mensual.csv", index=False)

    todas_pred, todas_met = [], []
    for nombre, (col, val) in SERIES.items():
        s = series.serie_mensual(df, col, val)
        print(f"[modelos] {nombre}…")
        p, m, fin_train = modelos_serie(nombre, s)
        todas_pred.append(p)
        todas_met += m
    pd.concat(todas_pred).to_csv(OUT / "predicciones.csv", index=False)
    met = pd.DataFrame(todas_met)
    met["rank"] = met.groupby("serie")["RMSE"].rank(method="first").astype(int)
    met.sort_values(["serie", "rank"]).to_csv(OUT / "metricas.csv", index=False)
    print(f"[ok] entrenamiento hasta {fin_train.date()} · archivos en {OUT.relative_to(ROOT)}")
    print(met.sort_values(["serie", "rank"]).to_string(index=False))


if __name__ == "__main__":
    main()
