"""
Registro de ejecuciones: identificador, bitácora por corrida e historial.

Cada ejecución del pipeline produce:
  - un identificador `run_id` (fecha-hora UTC, sufijo aleatorio y commit);
  - un archivo de bitácora en logs/ con hora de inicio y fin, etapas completadas,
    advertencias, errores y resultado de la publicación;
  - una línea en el historial JSONL, que es lo que consulta el monitoreo.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

FORMATO = "%(asctime)s | %(levelname)-10s | %(message)s"
FECHA = "%Y-%m-%d %H:%M:%S"


def nuevo_run_id(commit: str) -> str:
    """
    Identificador de ejecución: fecha-hora UTC, sufijo aleatorio corto y commit.

    El sufijo aleatorio evita que dos ejecuciones lanzadas en el mismo segundo
    compartan identificador y se sobreescriban los expedientes.
    """
    sello = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{sello}-{uuid4().hex[:4]}-{commit[:7]}"


@dataclass
class Bitacora:
    """Acumula lo ocurrido en una ejecución y lo escribe en disco."""
    run_id: str
    archivo: Path
    inicio: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    fin: datetime | None = None
    etapas: list = field(default_factory=list)
    advertencias: list = field(default_factory=list)
    errores: list = field(default_factory=list)
    publicacion: str = "no intentada"
    _t0: float = field(default_factory=time.perf_counter)
    _logger: logging.Logger | None = None

    # ------------------------------------------------------------- arranque
    def __post_init__(self):
        self.archivo.parent.mkdir(parents=True, exist_ok=True)
        log = logging.getLogger(f"obs.{self.run_id}")
        log.setLevel(logging.INFO)
        log.handlers.clear()
        log.propagate = False
        archivo = logging.FileHandler(self.archivo, encoding="utf-8")
        archivo.setFormatter(logging.Formatter(FORMATO, FECHA))
        consola = logging.StreamHandler()
        consola.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(archivo)
        log.addHandler(consola)
        self._logger = log
        self.info(f"ejecución {self.run_id} — inicio {self.inicio.isoformat(timespec='seconds')}")

    # ------------------------------------------------------------- mensajes
    def info(self, mensaje: str) -> None:
        self._logger.info(mensaje)

    def advertencia(self, mensaje: str) -> None:
        self.advertencias.append(mensaje)
        self._logger.warning(mensaje)

    def error(self, mensaje: str) -> None:
        self.errores.append(mensaje)
        self._logger.error(mensaje)

    def etapa(self, nombre: str, detalle: str = "") -> None:
        """Marca una etapa como completada (lo que el enunciado pide registrar)."""
        self.etapas.append(nombre)
        self.info(f"[etapa] {nombre}" + (f" — {detalle}" if detalle else ""))

    # ------------------------------------------------------------- cierre
    def cerrar(self, estado: str, extra: dict | None = None,
               historial: Path | None = None) -> dict:
        self.fin = datetime.now(timezone.utc)
        resumen = {
            "run_id": self.run_id,
            "estado": estado,
            "inicio": self.inicio.isoformat(timespec="seconds"),
            "fin": self.fin.isoformat(timespec="seconds"),
            "duracion_s": round(time.perf_counter() - self._t0, 2),
            "etapas": self.etapas,
            "n_advertencias": len(self.advertencias),
            "advertencias": self.advertencias,
            "n_errores": len(self.errores),
            "errores": self.errores,
            "publicacion": self.publicacion,
        }
        resumen.update(extra or {})
        self.info(f"estado final: {estado} | publicación: {self.publicacion} | "
                  f"{resumen['duracion_s']} s | advertencias: {len(self.advertencias)} | "
                  f"errores: {len(self.errores)}")
        if historial is not None:
            historial.parent.mkdir(parents=True, exist_ok=True)
            with open(historial, "a", encoding="utf-8") as f:
                f.write(json.dumps(resumen, ensure_ascii=False, default=str) + "\n")
        for h in list(self._logger.handlers):
            h.close()
            self._logger.removeHandler(h)
        return resumen


def leer_historial(historial: Path) -> list[dict]:
    """Ejecuciones anteriores, de la más antigua a la más reciente."""
    if not historial.exists():
        return []
    filas = []
    for linea in historial.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if linea:
            try:
                filas.append(json.loads(linea))
            except json.JSONDecodeError:
                continue
    return filas
