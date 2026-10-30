"""
Etapa 3 — preparación de la serie del destino.

Construye la serie mensual observada del destino asignado a partir de la fuente
mensual, aplicando el estimador del observatorio:

    V(m) = D(m) + f · A(m)

  D(m)  entradas observadas en los puestos de entrada del propio destino
        (`procedimiento.puestos_destino`), para los tipos de viajero del destino.
  A(m)  turistas que ingresan por el puesto base del reparto
        (`procedimiento.puesto_reparto`, vía aérea) y pueden continuar al destino.
  f     factor de reparto: supuesto del observatorio, no un parámetro estimado.

D y A se calculan sobre conjuntos de registros disjuntos, así que ningún viajero
se cuenta dos veces (lo verifica el control 6). Los meses del rango de cobertura
sin información quedan como NaN, nunca como cero.
"""
from __future__ import annotations

import pandas as pd


def _fecha_mensual(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(dict(year=df["Año"], month=df["Mes cod"], day=1))


def _serie(df: pd.DataFrame, puestos, tipos, meses: pd.DatetimeIndex) -> pd.Series:
    """Suma mensual de viajeros en los puestos y tipos indicados, reindexada a `meses`."""
    d = df.loc[df["Frontera"].isin(puestos) & df["Tipo de Viajero"].isin(tipos)]
    s = d.groupby("fecha")["Viajero"].sum()
    # reindex sin fillna: un mes sin información queda NaN y lo reporta el control 2
    return s.reindex(meses)


def construir(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """
    Devuelve un marco mensual con las componentes del estimador y la serie del
    destino, indexado por el rango de cobertura configurado.

    Columnas: `directas`, `base_reparto`, `reparto`, `viajeros_destino`.
    """
    p = cfg["procedimiento"]
    meses = cfg.meses_cobertura()
    df = df.assign(fecha=_fecha_mensual(df))

    directas = _serie(df, p["puestos_destino"], p["tipos_destino"], meses)
    base = _serie(df, [p["puesto_reparto"]], p["tipos_reparto"], meses)
    factor = float(p["factor_reparto"])

    tabla = pd.DataFrame({
        "directas": directas,
        "base_reparto": base,
        "reparto": base * factor,
    }, index=meses)
    tabla["viajeros_destino"] = tabla["directas"] + tabla["reparto"]
    tabla.index.name = "fecha"
    return tabla


def serie_destino(tabla: pd.DataFrame) -> pd.Series:
    """Serie mensual observada del destino, con frecuencia mensual explícita."""
    s = tabla["viajeros_destino"].asfreq("MS")
    s.name = "viajeros_destino"
    return s


def partir(serie: pd.Series, cfg) -> tuple[pd.Series, pd.Series]:
    """
    Separa ajuste y evaluación respetando el orden temporal.

    El tramo de ajuste termina en `periodos.ajuste_fin`; los meses de evaluación
    se ocultan del ajuste aunque tengan valor observado, de modo que los
    parámetros se calculan solo con información anterior al mes evaluado.
    """
    p = cfg.periodos
    ajuste = serie.loc[p["ajuste_inicio"]:p["ajuste_fin"]]
    evaluacion = serie.loc[p["evaluacion_inicio"]:p["evaluacion_fin"]]
    return ajuste, evaluacion


def descripcion(cfg) -> dict:
    """Metadatos del procedimiento, para registrar con cada ejecución."""
    p = cfg["procedimiento"]
    return {
        "variable": "viajeros_destino",
        "definicion": ("viajeros internacionales que ingresan al destino en el mes, "
                       "estimados como entradas directas más el factor de reparto "
                       "aplicado a los turistas del puesto aéreo base"),
        "unidad": "viajeros por mes",
        "agregacion": "mes × destino",
        "formula": "V(m) = D(m) + f * A(m)",
        "puestos_destino": list(p["puestos_destino"]),
        "tipos_destino": list(p["tipos_destino"]),
        "puesto_reparto": p["puesto_reparto"],
        "tipos_reparto": list(p["tipos_reparto"]),
        "factor_reparto": float(p["factor_reparto"]),
        "factor_reparto_rango": list(p.get("factor_reparto_rango") or []),
    }
