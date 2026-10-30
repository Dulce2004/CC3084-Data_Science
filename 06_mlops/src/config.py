"""
Carga y validación del archivo de configuración del flujo.

Única puerta de entrada a los parámetros: ninguna otra parte del pipeline lee
valores fijos. Si falta una clave obligatoria el flujo se detiene aquí, antes de
tocar los datos.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_POR_DEFECTO = ROOT / "06_mlops" / "config" / "pipeline.yaml"

# Variable de entorno que permite montar los datos institucionales fuera del
# repositorio (requisito de Docker: no meter los datos en la imagen).
ENV_DATOS = "OBS_DATOS_CRUDOS"

CLAVES_OBLIGATORIAS = [
    ("version_procedimiento",),
    ("semilla",),
    ("destino", "nombre"),
    ("fuentes", "migracion", "ruta"),
    ("fuentes", "migracion", "columnas_requeridas"),
    ("fuentes", "migracion", "clave"),
    ("procedimiento", "modelo"),
    ("procedimiento", "factor_reparto"),
    ("procedimiento", "puestos_destino"),
    ("periodos", "cobertura_inicio"),
    ("periodos", "ajuste_fin"),
    ("periodos", "evaluacion_inicio"),
    ("aceptacion", "mae_max"),
    ("controles", "umbrales"),
    ("salidas", "resultados_dir"),
]


class ErrorConfiguracion(Exception):
    """La configuración no permite ejecutar el flujo."""


def _buscar(d: dict, ruta: tuple):
    actual = d
    for parte in ruta:
        if not isinstance(actual, dict) or parte not in actual:
            return None
        actual = actual[parte]
    return actual


def mes(texto: str) -> pd.Timestamp:
    """'2026-01' -> Timestamp('2026-01-01'). Formato único de período en la config."""
    return pd.Period(str(texto), freq="M").to_timestamp()


def huella(path: Path, bloque: int = 1 << 20) -> str:
    """SHA-256 del archivo: identifica la versión exacta de una fuente."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for trozo in iter(lambda: f.read(bloque), b""):
            h.update(trozo)
    return h.hexdigest()


def commit_git() -> str:
    """Identificador del commit que produjo la ejecución (trazabilidad del código)."""
    try:
        salida = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT,
            capture_output=True, text=True, timeout=10, check=True)
        sucio = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT,
            capture_output=True, text=True, timeout=10, check=True).stdout.strip()
        return salida.stdout.strip() + ("-sucio" if sucio else "")
    except Exception:
        return "desconocido"


@dataclass
class Config:
    """Configuración resuelta: rutas absolutas y períodos como Timestamp."""
    bruta: dict
    ruta_config: Path
    huella_config: str
    raiz: Path = ROOT
    _periodos: dict = field(default_factory=dict)

    # -------------------------------------------------- accesos convenientes
    def __getitem__(self, clave):
        return self.bruta[clave]

    def get(self, *ruta, defecto=None):
        valor = _buscar(self.bruta, ruta)
        return defecto if valor is None else valor

    @property
    def fuente_migracion(self) -> Path:
        """Ruta de la fuente institucional. OBS_DATOS_CRUDOS tiene prioridad."""
        env = os.environ.get(ENV_DATOS)
        if env:
            return Path(env).expanduser().resolve()
        return (self.raiz / self.get("fuentes", "migracion", "ruta")).resolve()

    def salida(self, nombre: str) -> Path:
        return (self.raiz / self.get("salidas", nombre)).resolve()

    @property
    def periodos(self) -> dict:
        if not self._periodos:
            p = self.bruta["periodos"]
            self._periodos = {k: mes(v) for k, v in p.items()}
        return self._periodos

    def meses_evaluacion(self) -> pd.DatetimeIndex:
        p = self.periodos
        return pd.date_range(p["evaluacion_inicio"], p["evaluacion_fin"], freq="MS")

    def meses_cobertura(self) -> pd.DatetimeIndex:
        p = self.periodos
        return pd.date_range(p["cobertura_inicio"], p["cobertura_fin"], freq="MS")

    def nivel_control(self, nombre: str) -> str:
        return str(self.get("controles", nombre, defecto="advertencia"))

    def umbral(self, nombre: str, defecto=None):
        return self.get("controles", "umbrales", nombre, defecto=defecto)


def cargar(ruta: str | Path | None = None) -> Config:
    """Lee el YAML, comprueba las claves obligatorias y resuelve los períodos."""
    ruta = Path(ruta) if ruta else CONFIG_POR_DEFECTO
    if not ruta.exists():
        raise ErrorConfiguracion(f"no existe el archivo de configuración: {ruta}")
    with open(ruta, encoding="utf-8") as f:
        bruta = yaml.safe_load(f)
    if not isinstance(bruta, dict):
        raise ErrorConfiguracion(f"{ruta} no contiene un mapa YAML")

    faltantes = [".".join(c) for c in CLAVES_OBLIGATORIAS if _buscar(bruta, c) is None]
    if faltantes:
        raise ErrorConfiguracion("faltan claves en la configuración: " + ", ".join(faltantes))

    cfg = Config(bruta=bruta, ruta_config=ruta.resolve(), huella_config=huella(ruta))
    _validar_coherencia(cfg)
    return cfg


def _validar_coherencia(cfg: Config) -> None:
    """Comprobaciones que no dependen de los datos: orden de los períodos y rangos."""
    p = cfg.periodos
    if p["ajuste_inicio"] >= p["ajuste_fin"]:
        raise ErrorConfiguracion("periodos.ajuste_inicio debe ser anterior a ajuste_fin")
    if p["evaluacion_inicio"] <= p["ajuste_fin"]:
        raise ErrorConfiguracion(
            "periodos.evaluacion_inicio debe ser posterior a ajuste_fin: la validación "
            "temporal no puede usar información del período evaluado")
    if p["evaluacion_fin"] > p["cobertura_fin"]:
        raise ErrorConfiguracion("el período de evaluación excede la cobertura configurada")

    f = float(cfg.get("procedimiento", "factor_reparto"))
    if not 0 < f <= 1:
        raise ErrorConfiguracion(f"procedimiento.factor_reparto debe estar en (0, 1]; es {f}")

    modelo = cfg.get("procedimiento", "modelo")
    if modelo not in {"holt_winters", "ses", "seasonal_naive"}:
        raise ErrorConfiguracion(f"procedimiento.modelo no reconocido: {modelo}")
    ref = cfg.get("procedimiento", "referencia")
    if ref not in {"ultimo_valor", "promedio_historico", "seasonal_naive"}:
        raise ErrorConfiguracion(f"procedimiento.referencia no reconocida: {ref}")
