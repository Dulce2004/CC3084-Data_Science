"""Datos sintéticos y configuración de prueba compartidos por las pruebas."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "06_mlops" / "src"))

import config as config_mod  # noqa: E402

CONFIG_REAL = ROOT / "06_mlops" / "config" / "pipeline.yaml"

PUESTOS_DESTINO = ["02 Mundo Maya", "17 El Naranjo", "18 Bethel", "20 Melchor de Mencos"]
PUESTO_REPARTO = "01 La Aurora"
TIPOS = ["Turista", "Excursionista"]


def datos_sinteticos(inicio="2009-01", fin="2026-06", semilla=7) -> pd.DataFrame:
    """
    Copia sintética de la fuente institucional: misma estructura y granularidad,
    valores inventados. No contiene datos institucionales.
    """
    rng = np.random.default_rng(semilla)
    meses = pd.period_range(inicio, fin, freq="M")
    filas = []
    for m in meses:
        estacion = 1 + 0.3 * np.sin(2 * np.pi * (m.month - 1) / 12)
        for puesto in PUESTOS_DESTINO + [PUESTO_REPARTO]:
            base = 30000 if puesto == PUESTO_REPARTO else 2500
            for tipo in TIPOS:
                for pais, grupo in [("Estados Unidos de América", "América del Norte"),
                                    ("México", "América del Centro"),
                                    ("Alemania", "Europa")]:
                    filas.append({
                        "Año": m.year,
                        "Mes cod": m.month,
                        "Mes": m.strftime("%b"),
                        "Vía": "Aérea" if puesto in (PUESTO_REPARTO, "02 Mundo Maya")
                               else "Terrestre",
                        "Frontera": puesto,
                        "País": pais,
                        "Región": grupo.upper(),
                        "Región dos": grupo,
                        "Regiones OMT": grupo.upper(),
                        "MCEO": "01 PRUEBA",
                        "Agrupación Residencia": grupo,
                        "Tipo de Viajero": tipo,
                        "Viajero": float(round(base * estacion * rng.uniform(0.8, 1.2), 1)),
                    })
    return pd.DataFrame(filas)


@pytest.fixture
def datos() -> pd.DataFrame:
    return datos_sinteticos()


@pytest.fixture
def cfg(tmp_path, datos):
    """
    Configuración de prueba: misma estructura que la real, pero apuntando a la
    copia sintética y a carpetas de salida temporales.
    """
    bruta = yaml.safe_load(CONFIG_REAL.read_text(encoding="utf-8"))

    fuente = tmp_path / "fuente_sintetica.csv"
    datos.to_csv(fuente, index=False, encoding="utf-8")

    bruta["fuentes"]["migracion"]["ruta"] = fuente.relative_to(tmp_path).as_posix()
    for clave, sub in [("resultados_dir", "resultados"), ("aprobado_dir", "aprobado"),
                       ("logs_dir", "logs")]:
        bruta["salidas"][clave] = sub
    bruta["salidas"]["historial"] = "historial.jsonl"
    bruta["salidas"]["tablero_dir"] = "tablero"

    ruta = tmp_path / "pipeline.yaml"
    ruta.write_text(yaml.safe_dump(bruta, allow_unicode=True, sort_keys=False),
                    encoding="utf-8")

    c = config_mod.cargar(ruta)
    c.raiz = tmp_path            # todas las rutas relativas caen en tmp_path
    return c
