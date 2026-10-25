"""
Precálculo del tablero (Laboratorio 10).

No entrena modelos nuevos: vuelve a correr los mismos modelos de los Laboratorios 1 y 2
(mismas funciones, órdenes y configuraciones) para guardar sus predicciones mes a mes y
que el tablero no dependa de Prophet ni de TensorFlow.

Genera en 05_dashboard_interactivo/data/resultados/:
  - mercados.csv     viajeros por mes x región x mercado (País) x vía x tipo de viajero
  - fronteras.csv    viajeros por mes x vía x frontera x tipo de viajero
  - predicciones.csv real y pronóstico de cada modelo en el conjunto de prueba (abr-2021 a jun-2026)
  - metricas.csv     MAE y RMSE de cada modelo por serie

Requiere Python 3.12 (TensorFlow) y requirements-precalculo.txt. Desde la raíz del repo:
    python 05_dashboard_interactivo/src/preparar_datos.py
"""
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for etapa in ["01_limpieza_eda", "02_series_tiempo", "03_modelado_prediccion", "04_deep_learning_lstm"]:
    sys.path.append(str(ROOT / etapa / "src"))
import limpieza  # noqa: E402
import series  # noqa: E402
import modelado as mod  # noqa: E402
import lstm_utils as lu  # noqa: E402

warnings.filterwarnings("ignore")
OUT = ROOT / "05_dashboard_interactivo" / "data" / "resultados"

# (nombre en el tablero, columna, valor, D) -- d=1 en todas; D según la fuerza estacional (Lab 1, 03_modelado.ipynb)
SERIES = [
    ("Total", None, None, 1),
    ("Vía Aérea", "Vía", "Aérea", 1),
    ("Vía Terrestre", "Vía", "Terrestre", 0),
    ("Vía Marítima", "Vía", "Marítima", 1),
    ("Turista", "Tipo de Viajero", "Turista", 1),
    ("Excursionista", "Tipo de Viajero", "Excursionista", 0),
    ("Viajero", "Tipo de Viajero", "Viajero", 0),
    ("Cruceristas", "Tipo de Viajero", "Cruceristas", 1),
]

# Mejor LSTM de cada serie en el Laboratorio 2 (04_lstm_modelos.ipynb, celda de cfgs)
LSTM = {
    "Total": {"nombre": "LSTM", "look_back": 12, "unidades": (32,), "dropout": 0.0, "lr": 1e-3},
    "Vía Aérea": {"nombre": "LSTM", "look_back": 18, "unidades": (32, 16), "dropout": 0.1, "lr": 5e-4},
}

# RMSE publicados en los informes de los Labs 1 y 2, para comprobar que se reprodujeron.
# Los modelos de statsmodels son deterministas (deben coincidir); Prophet y LSTM dependen de la versión
# de cmdstan/TensorFlow, así que solo se informa la diferencia.
RMSE_EXACTO = {
    ("Total", "SES"): 199918.7,
    ("Total", "Holt-Winters"): 278370.6,
    ("Vía Aérea", "SES"): 41349.5,
    ("Vía Terrestre", "SARIMA grid-AIC"): 181595.6,
    ("Cruceristas", "Seasonal naive"): 10090.9,
}
RMSE_REFERENCIA = {
    ("Total", "Prophet"): 107269.2,
    ("Vía Terrestre", "Prophet"): 71292.4,
    ("Cruceristas", "Prophet"): 9063.8,
    ("Total", "LSTM"): 162501.8,
    ("Vía Aérea", "LSTM"): 55424.2,
}


def agregados(df: pd.DataFrame):
    mercados = (df.groupby(["fecha", "Región dos", "País", "Vía", "Tipo de Viajero"], as_index=False)["Viajero"]
                  .sum())
    fronteras = df.groupby(["fecha", "Vía", "Frontera", "Tipo de Viajero"], as_index=False)["Viajero"].sum()
    return mercados, fronteras


def modelos_lab1(nombre: str, s: pd.Series, D: int):
    """Mismo flujo que modelar_serie() del Lab 1: devuelve {modelo: (pred, lo, hi, detalle)}."""
    tr, te = series.split_temporal(s)
    n, idx = len(te), te.index
    tr_log = mod.log1p(tr)

    manual_order, manual_seasonal = (1, 1, 1), (1, D, 1, 12)
    manual_res = mod.ajustar_sarima(tr_log, manual_order, manual_seasonal)
    ordenes = [(p, 1, q) for p in range(3) for q in range(3)]
    ordenes_est = [(P, D, Q) for P in range(2) for Q in range(2)]
    tabla_grid, modelos_grid = mod.buscar_sarima(tr_log, ordenes, ordenes_est, s=12)
    best_key = tabla_grid.iloc[0]["modelo"]

    pred_pr, m = mod.modelo_prophet(tr, n, idx)
    banda = m.predict(pd.DataFrame({"ds": idx}))
    out = {
        "Prophet": (pred_pr, banda["yhat_lower"].clip(lower=0).values, banda["yhat_upper"].values,
                    f"intervalo {m.interval_width:.0%}"),
        "SES": (mod.modelo_ses(tr, n, idx)[0], None, None, ""),
        "Holt-Winters": (mod.modelo_holt_winters(tr, n, idx)[0], None, None, "tendencia y estacionalidad aditivas"),
        "Seasonal naive": (mod.modelo_seasonal_naive(tr, n, idx), None, None, "repite el último año"),
        "SARIMA manual": (mod.predecir_sarima(manual_res, n, idx), None, None, f"{manual_order}x{manual_seasonal}"),
        "SARIMA grid-AIC": (mod.predecir_sarima(modelos_grid[best_key], n, idx), None, None,
                            best_key.replace("SARIMA", "")),
    }
    if nombre in LSTM:
        cfg = LSTM[nombre]
        _, _, preds, _ = lu.tunear_lstm(nombre, tr, te, [cfg], epochs=200, batch_size=8)
        out["LSTM"] = (preds[cfg["nombre"]], None, None,
                       f"capas {cfg['unidades']}, ventana {cfg['look_back']} meses (Lab 2)")
    return tr, te, out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # Como en 04_lstm_modelos.ipynb: fijar semillas (e importar TensorFlow) antes de todo; si no, el LSTM de
    # Total no reproduce el RMSE publicado.
    lu.fijar_semillas()
    df = limpieza.limpiar()
    mercados, fronteras = agregados(df)
    mercados.to_csv(OUT / "mercados.csv", index=False)
    fronteras.to_csv(OUT / "fronteras.csv", index=False)

    filas_pred, filas_met = [], []
    for nombre, col, val, D in SERIES:
        print(f"[modelos] {nombre}")
        s = series.serie_mensual(df, col, val)
        tr, te, preds = modelos_lab1(nombre, s, D)
        for modelo, (pred, lo, hi, detalle) in preds.items():
            pred = pred.reindex(te.index)
            m = mod.metricas(te, pred)
            filas_met.append({"serie": nombre, "modelo": modelo, "detalle": detalle,
                              "MAE": round(m["MAE"], 1), "RMSE": round(m["RMSE"], 1)})
            filas_pred.append(pd.DataFrame({
                "fecha": te.index, "serie": nombre, "modelo": modelo, "real": te.values,
                "pred": pred.values, "lo": lo, "hi": hi,
            }))
    metricas = pd.DataFrame(filas_met)
    predicciones = pd.concat(filas_pred, ignore_index=True)

    # Autoverificación contra los informes de los Labs 1 y 2
    def rmse_de(serie, modelo):
        return metricas.query("serie == @serie and modelo == @modelo")["RMSE"].iloc[0]

    for (serie, modelo), rmse in RMSE_EXACTO.items():
        assert abs(rmse_de(serie, modelo) - rmse) / rmse < 0.001, f"{serie}/{modelo}: {rmse_de(serie, modelo)} vs {rmse}"
    for (serie, modelo), rmse in RMSE_REFERENCIA.items():
        obtenido = rmse_de(serie, modelo)
        print(f"[verificación] {serie}/{modelo}: RMSE {obtenido:,.1f} (publicado {rmse:,.1f}; "
              f"diferencia {obtenido / rmse - 1:+.1%})")

    metricas.to_csv(OUT / "metricas.csv", index=False)
    predicciones.to_csv(OUT / "predicciones.csv", index=False)
    print("[listo]", *(p.name for p in sorted(OUT.glob("*.csv"))))


if __name__ == "__main__":
    main()
