"""
Monitoreo del Observatorio (ejercicio 6).

Se corre después del pipeline, leyendo lo que ya quedó registrado (historial de
ejecuciones y versión aprobada vigente); no vuelve a tocar la fuente ni a ajustar
nada. Produce un reporte por ejecución, con tres verificaciones mínimas:

  actualización  ¿la fuente y los resultados llegan al último período esperado?
  calidad        ¿cambiaron de forma relevante los faltantes, duplicados o la
                 cobertura, respecto a la ejecución aprobada anterior?
  resultados     ¿hay estimaciones negativas, no finitas, o una variación mensual
                 que excede el umbral configurado?

Una variación alta genera una **alerta para revisión**; por sí sola no concluye
que el procedimiento perdió precisión (eso solo se sabe cuando hay observación
comparable, y el error predictivo solo se actualiza entonces).

Uso:
    python 06_mlops/src/monitoreo.py --config 06_mlops/config/pipeline.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as config_mod   # noqa: E402
import publicacion as pub     # noqa: E402
import registro as reg        # noqa: E402

OK, ALERTA = "ok", "alerta"


@dataclass
class Verificacion:
    id: str
    descripcion: str
    estado: str = OK
    detalle: str = ""

    def como_fila(self) -> dict:
        return {"verificacion": self.id, "descripcion": self.descripcion,
                "estado": self.estado, "detalle": self.detalle}


@dataclass
class ReporteMonitoreo:
    run_id: str
    verificaciones: list = field(default_factory=list)

    def agregar(self, v: Verificacion) -> Verificacion:
        self.verificaciones.append(v)
        return v

    @property
    def hay_alertas(self) -> bool:
        return any(v.estado == ALERTA for v in self.verificaciones)

    def tabla(self) -> pd.DataFrame:
        return pd.DataFrame([v.como_fila() for v in self.verificaciones])

    def resumen(self) -> dict:
        return {"run_id": self.run_id, "n_alertas": sum(v.estado == ALERTA
                for v in self.verificaciones),
                "verificaciones": [v.como_fila() for v in self.verificaciones]}


def _leer_carpeta(cfg, carpeta) -> dict:
    """Lee controles.csv, estimaciones.csv y metadatos.json de una carpeta de ejecución.

    `carpeta` puede ser relativa a cfg.raiz (como se guarda en el historial) o
    una ruta absoluta ya resuelta."""
    carpeta = Path(carpeta)
    base = carpeta.resolve() if carpeta.is_absolute() else (cfg.raiz / carpeta).resolve()
    salida = {"base": base}
    for nombre, lector in [("controles.csv", pd.read_csv), ("estimaciones.csv", pd.read_csv)]:
        ruta = base / nombre
        salida[nombre] = lector(ruta) if ruta.exists() else None
    ruta_meta = base / "metadatos.json"
    salida["metadatos.json"] = (json.loads(ruta_meta.read_text(encoding="utf-8"))
                                if ruta_meta.exists() else {})
    return salida


# ====================================================================
# 1. Actualización
# ====================================================================
def verificar_actualizacion(cfg, actual: dict) -> Verificacion:
    """¿La fuente y los resultados corresponden al último período esperado?"""
    v = Verificacion("actualizacion",
                     "la fuente y las estimaciones cubren el último período esperado")
    esperado = cfg.get("monitoreo", "mes_esperado")
    if not esperado:
        v.detalle = "sin 'monitoreo.mes_esperado' configurado: no se verifica"
        return v
    esperado = config_mod.mes(esperado)

    meta = actual.get("metadatos.json", {})
    fuente_hasta = (meta.get("periodo_ajuste", "") + " / " + meta.get("periodo_evaluacion", ""))
    est = actual.get("estimaciones.csv")
    if est is None or est.empty:
        v.estado = ALERTA
        v.detalle = (f"la ejecución no generó estimaciones; se esperaba cobertura hasta "
                     f"{esperado.strftime('%Y-%m')}. Períodos registrados: {fuente_hasta}")
        return v

    ultimo_estimado = pd.to_datetime(est["periodo"], format="%Y-%m").max()
    if ultimo_estimado < esperado:
        v.estado = ALERTA
        v.detalle = (f"la última estimación es de {ultimo_estimado.strftime('%Y-%m')}; "
                     f"se esperaba llegar a {esperado.strftime('%Y-%m')}")
    else:
        v.detalle = f"última estimación {ultimo_estimado.strftime('%Y-%m')} (esperado " \
                    f"{esperado.strftime('%Y-%m')} o posterior)"
    return v


# ====================================================================
# 2. Calidad
# ====================================================================
def verificar_calidad(cfg, actual: dict, anterior: dict | None) -> Verificacion:
    """¿Cambiaron de forma relevante los faltantes, duplicados o la cobertura?"""
    v = Verificacion("calidad", "sin cambios relevantes en faltantes, duplicados o cobertura "
                     "respecto a la última versión aprobada")
    ctrl_actual = actual.get("controles.csv")
    if ctrl_actual is None:
        v.estado = ALERTA
        v.detalle = "la ejecución no dejó controles.csv: no se puede verificar la calidad"
        return v

    errores_actuales = ctrl_actual.loc[ctrl_actual["estado"] == "error", "control"].tolist()
    adv_actuales = ctrl_actual.loc[ctrl_actual["estado"] == "advertencia", "control"].tolist()

    if anterior is None:
        v.detalle = (f"primera versión con la que comparar; {len(errores_actuales)} errores y "
                     f"{len(adv_actuales)} advertencias en esta ejecución")
        if errores_actuales:
            v.estado = ALERTA
        return v

    ctrl_anterior = anterior.get("controles.csv")
    errores_antes = (ctrl_anterior.loc[ctrl_anterior["estado"] == "error", "control"].tolist()
                    if ctrl_anterior is not None else [])
    nuevos = sorted(set(errores_actuales) - set(errores_antes))

    if nuevos:
        v.estado = ALERTA
        v.detalle = "controles que antes pasaban y ahora están en error: " + ", ".join(nuevos)
    elif errores_actuales:
        v.estado = ALERTA
        v.detalle = "controles en error (ya lo estaban en la versión anterior): " + \
                    ", ".join(errores_actuales)
    else:
        v.detalle = (f"sin errores; {len(adv_actuales)} advertencias "
                     f"({', '.join(adv_actuales) if adv_actuales else 'ninguna'})")
    return v


# ====================================================================
# 3. Resultados
# ====================================================================
def verificar_resultados(cfg, actual: dict, anterior: dict | None) -> Verificacion:
    """¿Hay estimaciones negativas, no finitas, o una variación que excede el umbral?"""
    v = Verificacion("resultados",
                     "sin estimaciones negativas o no finitas, ni variaciones fuera de umbral")
    est = actual.get("estimaciones.csv")
    if est is None or est.empty:
        v.estado = ALERTA
        v.detalle = "la ejecución no generó estimaciones"
        return v

    valores = est["valor"].to_numpy(dtype="float64")
    negativos = int((valores < 0).sum())
    no_finitos = int((~np.isfinite(valores)).sum())
    if negativos or no_finitos:
        v.estado = ALERTA
        v.detalle = f"{negativos} estimaciones negativas y {no_finitos} no finitas"
        return v

    umbral_mensual = float(cfg.get("monitoreo", "variacion_mensual_max", defecto=1.0))
    est = est.sort_values("periodo")
    variacion = est["valor"].pct_change().abs()
    excede = variacion[variacion > umbral_mensual]

    if anterior is not None:
        est_ant = anterior.get("estimaciones.csv")
        if est_ant is not None and not est_ant.empty:
            comunes = set(est["periodo"]) & set(est_ant["periodo"])
            if comunes:
                a = est.set_index("periodo").loc[list(comunes), "valor"]
                b = est_ant.set_index("periodo").loc[list(comunes), "valor"]
                var_version = ((a - b).abs() / b.replace(0, np.nan)).max()
                if pd.notna(var_version) and var_version > umbral_mensual:
                    v.estado = ALERTA
                    v.detalle = (f"la estimación de al menos un mes cambió "
                                 f"{var_version:.0%} respecto a la versión aprobada anterior "
                                 f"(umbral {umbral_mensual:.0%}); requiere revisión, no implica "
                                 "por sí solo pérdida de precisión")
                    return v

    if len(excede):
        v.estado = ALERTA
        v.detalle = (f"{len(excede)} meses con variación mensual sobre {umbral_mensual:.0%}: "
                     + ", ".join(est.loc[excede.index, "periodo"].tolist()))
    else:
        v.detalle = f"sin negativos, no finitos, ni variaciones sobre {umbral_mensual:.0%}"
    return v


VERIFICACIONES = [verificar_actualizacion, verificar_calidad, verificar_resultados]


def monitorear(cfg, run_id: str | None = None) -> ReporteMonitoreo:
    """
    Corre las tres verificaciones sobre una ejecución (la última del historial si
    no se indica `run_id`), comparándola contra la versión aprobada vigente antes
    de esa ejecución.
    """
    historial = reg.leer_historial(cfg.salida("historial"))
    if not historial:
        raise ValueError("no hay ejecuciones en el historial; corra el pipeline primero")

    if run_id is None:
        objetivo = historial[-1]
    else:
        coincide = [h for h in historial if h["run_id"] == run_id]
        if not coincide:
            raise ValueError(f"run_id no encontrado en el historial: {run_id}")
        objetivo = coincide[0]

    carpeta_actual = objetivo.get("carpeta")
    actual = _leer_carpeta(cfg, carpeta_actual) if carpeta_actual else {
        "controles.csv": None, "estimaciones.csv": None, "metadatos.json": {}}

    # La línea base de comparación es la versión aprobada ANTES de `objetivo`, no la
    # vigente ahora mismo: si `objetivo` ya se publicó, version_aprobada() devuelve
    # el propio `objetivo` y la comparación quedaría vacía. El puntero guarda a quién
    # reemplazó en 'version_anterior'; si `objetivo` no llegó a publicarse (se
    # rechazó o se detuvo), la vigente actual ya es la línea base correcta.
    vigente = pub.version_aprobada(cfg)
    anterior = None
    if vigente:
        if vigente.get("run_id") == objetivo["run_id"]:
            base_id = vigente.get("version_anterior")
            if base_id:
                carpeta_base = cfg.salida("aprobado_dir") / base_id
                if carpeta_base.exists():
                    anterior = _leer_carpeta(cfg, carpeta_base)
        else:
            anterior = _leer_carpeta(cfg, vigente["carpeta"])

    reporte = ReporteMonitoreo(run_id=objetivo["run_id"])
    reporte.agregar(verificar_actualizacion(cfg, actual))
    reporte.agregar(verificar_calidad(cfg, actual, anterior))
    reporte.agregar(verificar_resultados(cfg, actual, anterior))
    return reporte


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Monitoreo del Observatorio (Lab. 11).")
    p.add_argument("--config", default=None)
    p.add_argument("--run-id", default=None, help="ejecución a monitorear (por defecto, la última)")
    args = p.parse_args(argv)

    cfg = config_mod.cargar(args.config)
    reporte = monitorear(cfg, args.run_id)

    print(f"Monitoreo de la ejecución {reporte.run_id}")
    for v in reporte.verificaciones:
        marca = "ALERTA" if v.estado == ALERTA else "ok"
        print(f"  [{marca:6s}] {v.id}: {v.detalle}")

    salida = cfg.salida("logs_dir") / f"monitoreo-{reporte.run_id}.json"
    salida.write_text(json.dumps(reporte.resumen(), ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(f"reporte escrito en {salida}")
    return 1 if reporte.hay_alertas else 0


if __name__ == "__main__":
    raise SystemExit(main())
