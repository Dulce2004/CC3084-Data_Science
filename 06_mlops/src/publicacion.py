"""
Etapas 6 y 7 — almacenamiento de resultados y publicación de la versión aprobada.

Separación deliberada:
  - `guardar_ejecucion` escribe siempre lo que produjo la ejecución, aprobada o
    no, en `data/resultados/<run_id>/`. Es el expediente de la corrida.
  - `publicar` copia esa salida a `data/aprobado/<run_id>/` y mueve el puntero
    `version_actual.json` únicamente cuando la versión fue aprobada.

Si una ejecución falla o es rechazada, el puntero no se toca y el tablero sigue
consultando la última versión aprobada. Las versiones anteriores se conservan,
así que restaurar una es reescribir el puntero (`restaurar`).
"""
from __future__ import annotations

import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PUNTERO = "version_actual.json"
ARCHIVOS = ["estimaciones.csv", "metricas.csv", "controles.csv", "metadatos.json"]


def _sin_no_finitos(valor):
    """
    Reemplaza NaN e infinitos por None.

    `json.dumps` los escribiría como `NaN` e `Infinity`, que no son JSON válido y
    rompen a cualquier lector estricto del expediente.
    """
    if isinstance(valor, dict):
        return {k: _sin_no_finitos(v) for k, v in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [_sin_no_finitos(v) for v in valor]
    if isinstance(valor, float) and not math.isfinite(valor):
        return None
    return valor


def _escribir_json(path: Path, datos: dict) -> None:
    path.write_text(json.dumps(_sin_no_finitos(datos), ensure_ascii=False, indent=2,
                               default=str, allow_nan=False),
                    encoding="utf-8")


def tabla_estimaciones(estimacion: pd.Series, observado: pd.Series, cfg,
                       run_id: str) -> pd.DataFrame:
    """
    Archivo de estimaciones con los metadatos que exige el enunciado: destino,
    período estimado, valor, unidad, identificador de ejecución, fecha de
    generación y versión del procedimiento.

    `valor_observado` se incluye para que el tablero distinga lo observado de lo
    estimado; queda vacío en los meses sin valor observado comparable.
    """
    generado = datetime.now(timezone.utc).isoformat(timespec="seconds")
    obs = pd.Series(observado).reindex(estimacion.index)
    return pd.DataFrame({
        "destino": cfg.get("destino", "nombre"),
        "codigo_destino": cfg.get("destino", "codigo"),
        "periodo": estimacion.index.strftime("%Y-%m"),
        "valor": estimacion.astype(float).round(1).to_numpy(),
        "unidad": "viajeros por mes",
        "tipo_valor": "estimado",
        "valor_observado": obs.astype(float).round(1).to_numpy(),
        "run_id": run_id,
        "fecha_generacion": generado,
        "version_procedimiento": cfg.get("version_procedimiento"),
    })


def guardar_ejecucion(cfg, run_id: str, estimaciones: pd.DataFrame | None,
                      metricas: pd.DataFrame | None, controles: pd.DataFrame,
                      metadatos: dict) -> Path:
    """Escribe el expediente completo de la ejecución y devuelve su carpeta."""
    destino = cfg.salida("resultados_dir") / run_id
    destino.mkdir(parents=True, exist_ok=True)
    controles.to_csv(destino / "controles.csv", index=False, encoding="utf-8")
    if estimaciones is not None:
        estimaciones.to_csv(destino / "estimaciones.csv", index=False, encoding="utf-8")
    if metricas is not None:
        metricas.to_csv(destino / "metricas.csv", index=False, encoding="utf-8")
    _escribir_json(destino / "metadatos.json", metadatos)
    # copia de la configuración usada: sin ella el expediente no es reproducible
    shutil.copy2(cfg.ruta_config, destino / cfg.ruta_config.name)
    return destino


def version_aprobada(cfg) -> dict | None:
    """Puntero a la versión aprobada vigente, o None si todavía no existe ninguna."""
    puntero = cfg.salida("aprobado_dir") / PUNTERO
    if not puntero.exists():
        return None
    try:
        return json.loads(puntero.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def versiones_aprobadas(cfg) -> list[str]:
    """Identificadores de las versiones aprobadas conservadas, de la más antigua a la más nueva."""
    base = cfg.salida("aprobado_dir")
    if not base.exists():
        return []
    return sorted(p.name for p in base.iterdir() if p.is_dir())


def publicar(cfg, run_id: str, carpeta_ejecucion: Path, metadatos: dict,
             bitacora=None) -> dict:
    """
    Copia la salida de la ejecución a `aprobado/<run_id>/` y actualiza el puntero.

    Solo debe llamarse cuando la versión fue aprobada; el pipeline es quien lo
    decide. La versión anterior no se borra: queda disponible para restaurarla.
    """
    base = cfg.salida("aprobado_dir")
    base.mkdir(parents=True, exist_ok=True)
    anterior = version_aprobada(cfg)

    destino = base / run_id
    destino.mkdir(parents=True, exist_ok=True)
    for nombre in ARCHIVOS + [cfg.ruta_config.name]:
        origen = carpeta_ejecucion / nombre
        if origen.exists():
            shutil.copy2(origen, destino / nombre)

    puntero = {
        "run_id": run_id,
        "carpeta": destino.relative_to(cfg.raiz).as_posix(),
        "destino": cfg.get("destino", "nombre"),
        "version_procedimiento": cfg.get("version_procedimiento"),
        "fecha_publicacion": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "version_anterior": (anterior or {}).get("run_id"),
        "periodo_estimado": metadatos.get("periodo_evaluacion"),
        "metrica_validacion": metadatos.get("metrica_validacion"),
    }
    _escribir_json(base / PUNTERO, puntero)
    if bitacora is not None:
        bitacora.info(f"[publicación] versión {run_id} aprobada y publicada en "
                      f"{destino.relative_to(cfg.raiz)}"
                      + (f"; versión anterior conservada: {puntero['version_anterior']}"
                         if puntero["version_anterior"] else ""))
    return puntero


def restaurar(cfg, run_id: str, bitacora=None) -> dict:
    """
    Vuelve a dejar vigente una versión aprobada anterior.

    No copia ni regenera nada: solo reescribe el puntero, que es lo que lee el
    tablero. Falla si la versión no está conservada.
    """
    base = cfg.salida("aprobado_dir")
    carpeta = base / run_id
    if not carpeta.is_dir():
        raise FileNotFoundError(
            f"no hay una versión aprobada con identificador {run_id}; "
            f"conservadas: {', '.join(versiones_aprobadas(cfg)) or 'ninguna'}")
    metadatos = {}
    archivo = carpeta / "metadatos.json"
    if archivo.exists():
        metadatos = json.loads(archivo.read_text(encoding="utf-8"))
    actual = version_aprobada(cfg)
    puntero = {
        "run_id": run_id,
        "carpeta": carpeta.relative_to(cfg.raiz).as_posix(),
        "destino": cfg.get("destino", "nombre"),
        "version_procedimiento": metadatos.get("version_procedimiento",
                                               cfg.get("version_procedimiento")),
        "fecha_publicacion": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "version_anterior": (actual or {}).get("run_id"),
        "periodo_estimado": metadatos.get("periodo_evaluacion"),
        "metrica_validacion": metadatos.get("metrica_validacion"),
        "restaurada": True,
    }
    _escribir_json(base / PUNTERO, puntero)
    if bitacora is not None:
        bitacora.info(f"[publicación] restaurada la versión aprobada {run_id}")
    return puntero
