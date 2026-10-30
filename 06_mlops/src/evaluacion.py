"""
Etapa 5 — evaluación y regla de aceptación.

La evaluación oculta los meses de `periodos.evaluacion_*` del ajuste y los estima
como si no estuvieran disponibles. Se calculan MAE, RMSE y error relativo del
procedimiento y de la referencia sobre los mismos meses.

MAPE se informa solo cuando ningún valor observado del período evaluado es cero;
si alguno lo es, queda en blanco y se explica, porque la división no está
definida (se prefiere el error relativo, que divide por el promedio del período).

La regla de aceptación se lee de `aceptacion` y se aplica tal como está escrita:
no se ajusta después de ver los resultados.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def metricas(observado: pd.Series, estimado: pd.Series) -> dict:
    """MAE, RMSE, error relativo y MAPE (este último solo si no hay ceros observados)."""
    y = pd.Series(observado).astype(float)
    yhat = pd.Series(estimado).astype(float).reindex(y.index)
    error = yhat - y
    promedio = float(y.mean())

    mae = float(np.abs(error).mean())
    rmse = float(np.sqrt((error ** 2).mean()))
    salida = {
        "n_meses": int(len(y)),
        "promedio_observado": promedio,
        "MAE": mae,
        "RMSE": rmse,
        "error_relativo": mae / promedio if promedio else np.nan,
        "sesgo": float(error.mean()),
    }
    if (y == 0).any():
        salida["MAPE"] = np.nan
        salida["nota_mape"] = ("no se calcula: hay meses con valor observado igual a cero y "
                               "la división no está definida; se usa el error relativo")
    else:
        salida["MAPE"] = float((np.abs(error) / y).mean())
        salida["nota_mape"] = ""
    return salida


def comparar(observado: pd.Series, estimacion: pd.Series, referencia: pd.Series,
             nombre_modelo: str, nombre_referencia: str) -> pd.DataFrame:
    """Tabla de métricas del procedimiento y de la referencia sobre los mismos meses."""
    filas = []
    for nombre, serie, papel in [(nombre_modelo, estimacion, "procedimiento"),
                                 (nombre_referencia, referencia, "referencia")]:
        fila = {"papel": papel, "procedimiento": nombre}
        fila.update(metricas(observado, serie))
        filas.append(fila)
    return pd.DataFrame(filas)


def aplicar_regla(tabla: pd.DataFrame, reporte_controles, estimaciones: pd.DataFrame,
                  meses_evaluacion: pd.DatetimeIndex, cfg) -> dict:
    """
    Aplica la regla de aceptación de la versión candidata.

    Cada condición se evalúa por separado y se informa; basta que una falle para
    rechazar. El orden es el del enunciado: primero los controles de calidad,
    después la integridad de las salidas y al final el desempeño predictivo.
    """
    a = cfg["aceptacion"]
    proc = tabla.loc[tabla["papel"] == "procedimiento"].iloc[0]
    ref = tabla.loc[tabla["papel"] == "referencia"].iloc[0]

    condiciones = []

    def agregar(nombre, cumple, detalle):
        condiciones.append({"condicion": nombre, "cumple": bool(cumple), "detalle": detalle})

    if a.get("controles_criticos_sin_error", True):
        sin_error = not reporte_controles.detiene
        agregar("controles críticos sin error", sin_error,
                "controles en error: " +
                (", ".join(r.id for r in reporte_controles.errores) or "ninguno"))

    if a.get("salidas_completas", True):
        esperadas = len(meses_evaluacion)
        completas = (len(estimaciones) == esperadas
                     and not estimaciones["valor"].isna().any()
                     and np.isfinite(estimaciones["valor"].astype(float)).all())
        agregar("salidas completas", completas,
                f"{len(estimaciones)} de {esperadas} meses estimados, "
                f"{int(estimaciones['valor'].isna().sum())} valores vacíos")

    mae_max = a.get("mae_max")
    if mae_max is not None:
        agregar(f"MAE ≤ {float(mae_max):,.0f} viajeros/mes", proc["MAE"] <= float(mae_max),
                f"MAE del procedimiento: {proc['MAE']:,.1f}")

    rel_max = a.get("error_relativo_max")
    if rel_max is not None:
        agregar(f"error relativo ≤ {float(rel_max):.0%}",
                proc["error_relativo"] <= float(rel_max),
                f"error relativo del procedimiento: {proc['error_relativo']:.1%}")

    if a.get("mejor_o_igual_que_referencia", False):
        agregar("MAE no peor que la referencia", proc["MAE"] <= ref["MAE"],
                f"procedimiento {proc['MAE']:,.1f} contra referencia {ref['MAE']:,.1f}")

    aprobada = all(c["cumple"] for c in condiciones)
    return {
        "aprobada": aprobada,
        "condiciones": condiciones,
        "no_cumplidas": [c["condicion"] for c in condiciones if not c["cumple"]],
        "supera_referencia": bool(proc["MAE"] <= ref["MAE"]),
        "comparacion_referencia": (
            f"MAE procedimiento {proc['MAE']:,.1f} contra referencia "
            f"({ref['procedimiento']}) {ref['MAE']:,.1f}"),
    }


def limitaciones_validacion(cfg) -> list[str]:
    """
    Limitaciones que impiden reproducir exactamente la situación de enero y
    febrero de 2026, cuando esos meses todavía no tenían valor observado.
    """
    p = cfg.periodos
    return [
        ("La fuente actual ya trae los meses evaluados; al ocultarlos se reproduce la "
         "ausencia de valor observado, pero no el resto del estado de la fuente en esa "
         "fecha (revisiones posteriores de meses anteriores ya están incorporadas)."),
        (f"El tramo de ajuste termina en {p['ajuste_fin'].date()} y la evaluación cubre "
         f"{p['evaluacion_inicio'].date()} a {p['evaluacion_fin'].date()}: la estimación de "
         "cada mes se hace con el mismo ajuste, no reestimando el procedimiento mes a mes."),
        ("El factor de reparto es un supuesto fijo del observatorio; en enero y febrero de "
         "2026 no se dispone de un valor calibrado del período, así que el error medido "
         "mezcla el error del procedimiento con el del supuesto."),
        ("No se calculan errores para meses sin valor observado comparable: si la fuente no "
         "trae el mes, queda sin métrica en lugar de compararse contra cero."),
    ]
