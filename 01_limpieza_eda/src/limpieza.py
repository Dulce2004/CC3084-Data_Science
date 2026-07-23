"""
Limpieza e ingesta de la base de migración.

Etapas:
  1. Ingesta de la hoja `Datos` del Excel crudo.
  2. Normalización de texto (strip de espacios).
  3. Construcción de la columna de fecha mensual (`fecha`).
  4. Depuración de categorías basura en `Región dos`.
  5. Validaciones (rango temporal, nulos, duplicados).
  6. Escritura del CSV limpio en data/processed/.

Se ejecuta como script:  python -m src.limpieza   (desde 01_limpieza_eda/)
o directamente:          python 01_limpieza_eda/src/limpieza.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
import config  # noqa: E402

# Categorías no válidas en 'Región dos' (ruido de captura, muy pocas filas)
REGION_DOS_INVALIDAS = {"0", "Cruceros"}


def ingesta(path=config.RAW_XLSX) -> pd.DataFrame:
    """Lee la hoja de datos del Excel crudo."""
    return pd.read_excel(path, sheet_name="Datos")


def normalizar_texto(df: pd.DataFrame) -> pd.DataFrame:
    """Quita espacios sobrantes en todas las columnas de texto."""
    df = df.copy()
    obj_cols = df.select_dtypes(include=["object", "string"]).columns
    for c in obj_cols:
        df[c] = df[c].astype("string").str.strip()
    return df


def construir_fecha(df: pd.DataFrame) -> pd.DataFrame:
    """Crea la columna `fecha` (primer día de cada mes) a partir de Año y Mes cod."""
    df = df.copy()
    df["fecha"] = pd.to_datetime(
        dict(year=df["Año"], month=df["Mes cod"], day=1)
    )
    return df


def depurar_categorias(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina filas con `Región dos` inválida y reporta cuántas se quitaron."""
    df = df.copy()
    mask = df["Región dos"].isin(REGION_DOS_INVALIDAS)
    n = int(mask.sum())
    if n:
        print(f"  - filas con 'Región dos' inválida eliminadas: {n}")
    return df.loc[~mask].reset_index(drop=True)


def validar(df: pd.DataFrame) -> None:
    """Chequeos mínimos de calidad; imprime un resumen."""
    print("  - registros:", len(df))
    print("  - rango temporal:", df["fecha"].min().date(), "→", df["fecha"].max().date())
    print("  - meses únicos:", df["fecha"].nunique())
    print("  - nulos totales:", int(df.isna().sum().sum()))
    print("  - filas duplicadas:", int(df.duplicated().sum()))
    print("  - viajeros negativos:", int((df["Viajero"] < 0).sum()))


def limpiar() -> pd.DataFrame:
    print("[limpieza] ingesta…")
    df = ingesta()
    print(f"  - crudo: {df.shape[0]} filas, {df.shape[1]} columnas")
    df = normalizar_texto(df)
    df = construir_fecha(df)
    df = depurar_categorias(df)
    validar(df)
    return df


def main():
    df = limpiar()
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.CLEAN_CSV, index=False)
    print(f"[limpieza] escrito: {config.CLEAN_CSV.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
