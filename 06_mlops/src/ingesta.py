"""
Etapa 1 — lectura de las fuentes.

Lee la fuente institucional mensual tal como viene (sin corregir nada: corregir
antes de validar escondería los problemas que los controles deben detectar) y
devuelve, junto al marco de datos, la identificación de la fuente y su huella
digital, que se registra con cada ejecución.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import config as cfg_mod


class ErrorIngesta(Exception):
    """No se pudo leer una fuente obligatoria."""


def _normalizar_texto(df: pd.DataFrame) -> pd.DataFrame:
    """Quita espacios sobrantes (misma regla que 01_limpieza_eda/src/limpieza.py)."""
    df = df.copy()
    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].astype("string").str.strip()
    return df


def leer_migracion(cfg) -> tuple[pd.DataFrame, dict]:
    """
    Lee la Base de Migración y describe la fuente.

    Acepta .xlsx (hoja configurada) y .csv, para que la prueba controlada pueda
    trabajar con una copia o con datos sintéticos sin tocar el código.
    """
    ruta: Path = cfg.fuente_migracion
    if not ruta.exists():
        raise ErrorIngesta(
            f"no se encontró la fuente institucional en {ruta}. Monte los datos o "
            f"defina {cfg_mod.ENV_DATOS} con su ubicación.")

    hoja = cfg.get("fuentes", "migracion", "hoja", defecto="Datos")
    if ruta.suffix.lower() in {".xlsx", ".xlsm"}:
        df = pd.read_excel(ruta, sheet_name=hoja)
    elif ruta.suffix.lower() == ".csv":
        df = pd.read_csv(ruta)
    else:
        raise ErrorIngesta(f"extensión de fuente no soportada: {ruta.suffix}")

    df = _normalizar_texto(df)
    fuente = {
        "nombre": "Base de Migración (fuente mensual)",
        "ruta": str(ruta),
        "formato": ruta.suffix.lower().lstrip("."),
        "hoja": hoja if ruta.suffix.lower() != ".csv" else None,
        "huella_sha256": cfg_mod.huella(ruta),
        "bytes": ruta.stat().st_size,
        "filas": int(len(df)),
        "columnas": list(df.columns),
        "granularidad": cfg.get("fuentes", "migracion", "granularidad"),
    }
    return df, fuente
