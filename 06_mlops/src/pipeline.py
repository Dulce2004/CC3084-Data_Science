"""
Flujo reproducible del Observatorio de Inteligencia Turística (Laboratorio 11).

Un solo comando ejecuta todas las etapas:

    python 06_mlops/src/pipeline.py --config 06_mlops/config/pipeline.yaml

Etapas, en orden:
    1. lectura        fuente institucional mensual + huella digital
    2. validacion     seis controles de calidad (se detiene aquí si uno crítico falla)
    3. preparacion    serie mensual del destino con el factor de reparto
    4. ajuste         parámetros del procedimiento, solo con el tramo de ajuste
    5. evaluacion     MAE / RMSE del procedimiento y de la referencia
    6. almacenamiento expediente de la ejecución en data/resultados/<run_id>/
    7. publicacion    copia a data/aprobado/ solo si la regla de aceptación aprueba

Nada se modifica a mano entre etapas y nada depende del estado de una sesión: la
única entrada es el archivo de configuración y las fuentes que este declara.

Opciones útiles:
    --solo-validar     corre la lectura y los controles, y termina
    --sin-publicar     ejecuta todo pero no toca la versión aprobada
    --restaurar RUNID  vuelve a dejar vigente una versión aprobada anterior
"""
from __future__ import annotations

import argparse
import platform
import sys
from importlib import metadata
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as config_mod      # noqa: E402
import evaluacion as eva         # noqa: E402
import ingesta                   # noqa: E402
import modelo as mdl             # noqa: E402
import preparacion as prep       # noqa: E402
import publicacion as pub        # noqa: E402
import registro as reg           # noqa: E402
import validaciones as val       # noqa: E402

PAQUETES = ["pandas", "numpy", "statsmodels", "PyYAML", "openpyxl"]


def versiones_dependencias() -> dict:
    """Versiones instaladas, para que la ejecución quede atada a un entorno concreto."""
    salida = {"python": platform.python_version(), "plataforma": platform.platform()}
    for nombre in PAQUETES:
        try:
            salida[nombre] = metadata.version(nombre)
        except metadata.PackageNotFoundError:
            salida[nombre] = "no instalado"
    return salida


class PipelineDetenido(Exception):
    """Un control crítico falló: no se reemplazan los resultados publicados."""


def ejecutar(cfg, solo_validar: bool = False, publicar: bool = True) -> dict:
    """Corre el flujo completo y devuelve el resumen de la ejecución."""
    commit = config_mod.commit_git()
    run_id = reg.nuevo_run_id(commit)
    bitacora = reg.Bitacora(run_id, cfg.salida("logs_dir") / f"{run_id}.log")
    historial = cfg.salida("historial")

    metadatos = {
        "run_id": run_id,
        "destino": cfg.get("destino", "nombre"),
        "codigo_destino": cfg.get("destino", "codigo"),
        "version_procedimiento": cfg.get("version_procedimiento"),
        "commit_git": commit,
        "configuracion": {"archivo": cfg.ruta_config.relative_to(cfg.raiz).as_posix(),
                          "huella_sha256": cfg.huella_config},
        "semilla": cfg.get("semilla"),
        "dependencias": versiones_dependencias(),
        "procedimiento": prep.descripcion(cfg),
        "periodo_ajuste": f"{cfg.periodos['ajuste_inicio'].date()} a "
                          f"{cfg.periodos['ajuste_fin'].date()}",
        "periodo_evaluacion": f"{cfg.periodos['evaluacion_inicio'].date()} a "
                              f"{cfg.periodos['evaluacion_fin'].date()}",
        "limitaciones_validacion": eva.limitaciones_validacion(cfg),
    }

    estimaciones = metricas = None
    reporte = None
    try:
        mdl.fijar_semilla(int(cfg.get("semilla")))

        # ---------------------------------------------------------- 1. lectura
        df, fuente = ingesta.leer_migracion(cfg)
        metadatos["fuente"] = fuente
        bitacora.etapa("lectura", f"{fuente['filas']:,} filas de {fuente['ruta']} "
                                  f"(sha256 {fuente['huella_sha256'][:12]}…)")

        # ------------------------------------------------------- 2. validación
        reporte = val.ejecutar(df, cfg, bitacora)
        metadatos["controles"] = reporte.resumen()
        bitacora.etapa("validacion",
                       f"{reporte.resumen()['n_ok']} ok, "
                       f"{reporte.resumen()['n_advertencias']} advertencias, "
                       f"{reporte.resumen()['n_errores']} errores")
        if reporte.detiene:
            raise PipelineDetenido(
                "controles críticos en error: "
                + ", ".join(r.id for r in reporte.errores if r.detiene))

        if solo_validar:
            carpeta = pub.guardar_ejecucion(cfg, run_id, None, None, reporte.tabla(), metadatos)
            bitacora.etapa("almacenamiento", carpeta.relative_to(cfg.raiz).as_posix())
            bitacora.publicacion = "no intentada (--solo-validar)"
            return bitacora.cerrar("validado", {"carpeta": str(carpeta)}, historial)

        # ------------------------------------------------------ 3. preparación
        tabla = prep.construir(df, cfg)
        serie = prep.serie_destino(tabla)
        ajuste, observado = prep.partir(serie, cfg)
        metadatos["serie"] = {
            "meses": int(serie.notna().sum()),
            "meses_sin_cobertura": int(serie.isna().sum()),
            "promedio_mensual": float(serie.mean()),
            "aporte_directas": float(tabla["directas"].sum()),
            "aporte_reparto": float(tabla["reparto"].sum()),
        }
        bitacora.etapa("preparacion",
                       f"serie del destino con {metadatos['serie']['meses']} meses; "
                       f"ajuste {len(ajuste)} meses, evaluación {len(observado)} meses")

        # ----------------------------------------------------------- 4. ajuste
        meses = cfg.meses_evaluacion()
        resultado = mdl.ajustar_y_estimar(ajuste, meses, cfg)
        metadatos["ajuste"] = {k: resultado[k] for k in
                               ("modelo", "detalle", "parametros", "referencia_nombre",
                                "n_ajuste", "ajuste_inicio", "ajuste_fin")}
        bitacora.etapa("ajuste", f"{resultado['modelo']} ({resultado['detalle']}) sobre "
                                 f"{resultado['n_ajuste']} meses hasta "
                                 f"{resultado['ajuste_fin']}")

        # ------------------------------------------------------- 5. evaluación
        comparables = observado.dropna()
        if comparables.empty:
            raise PipelineDetenido(
                "ningún mes del período de evaluación tiene valor observado comparable: "
                "no se calculan métricas")
        metricas = eva.comparar(comparables, resultado["estimacion"], resultado["referencia"],
                                resultado["modelo"], resultado["referencia_nombre"])
        proc = metricas.loc[metricas["papel"] == "procedimiento"].iloc[0]
        metadatos["metrica_validacion"] = {
            "MAE": round(float(proc["MAE"]), 1),
            "RMSE": round(float(proc["RMSE"]), 1),
            "error_relativo": round(float(proc["error_relativo"]), 4),
            "MAPE": None if pd.isna(proc["MAPE"]) else round(float(proc["MAPE"]), 4),
            "nota_mape": proc["nota_mape"],
            "meses_evaluados": int(proc["n_meses"]),
        }
        bitacora.etapa("evaluacion",
                       f"MAE {proc['MAE']:,.1f} | RMSE {proc['RMSE']:,.1f} | "
                       f"error relativo {proc['error_relativo']:.1%} sobre "
                       f"{int(proc['n_meses'])} meses")

        # --------------------------------------------------- 6. almacenamiento
        estimaciones = pub.tabla_estimaciones(resultado["estimacion"], observado, cfg, run_id)
        regla = eva.aplicar_regla(metricas, reporte, estimaciones, meses, cfg)
        metadatos["aceptacion"] = regla
        carpeta = pub.guardar_ejecucion(cfg, run_id, estimaciones, metricas,
                                        reporte.tabla(), metadatos)
        bitacora.etapa("almacenamiento", carpeta.relative_to(cfg.raiz).as_posix())

        # ------------------------------------------------------ 7. publicación
        for c in regla["condiciones"]:
            marca = "cumple" if c["cumple"] else "NO cumple"
            bitacora.info(f"[aceptación] {marca}: {c['condicion']} — {c['detalle']}")
        bitacora.info("[aceptación] " + regla["comparacion_referencia"]
                      + ("; el procedimiento supera a la referencia"
                         if regla["supera_referencia"]
                         else "; el procedimiento no supera a la referencia (la regla no "
                              "lo exige y no se modifica para aprobar)"))

        if not regla["aprobada"]:
            bitacora.advertencia("versión candidata RECHAZADA: " +
                                 ", ".join(regla["no_cumplidas"]))
            bitacora.publicacion = "rechazada; se conserva la versión aprobada anterior"
            vigente = pub.version_aprobada(cfg)
            bitacora.info("versión vigente: " +
                          (vigente["run_id"] if vigente
                           else "ninguna; las estimaciones no están disponibles"))
            return bitacora.cerrar("rechazada", {"carpeta": str(carpeta), **metadatos.get(
                "metrica_validacion", {})}, historial)

        if not publicar:
            bitacora.publicacion = "no intentada (--sin-publicar)"
            return bitacora.cerrar("aprobada_sin_publicar", {"carpeta": str(carpeta)},
                                   historial)

        puntero = pub.publicar(cfg, run_id, carpeta, metadatos, bitacora)
        bitacora.etapa("publicacion", f"versión vigente {puntero['run_id']}")
        bitacora.publicacion = f"publicada ({puntero['run_id']})"
        return bitacora.cerrar("aprobada", {"carpeta": str(carpeta),
                                            "version_anterior": puntero["version_anterior"],
                                            **metadatos["metrica_validacion"]}, historial)

    except PipelineDetenido as exc:
        bitacora.error(f"pipeline detenido: {exc}")
        bitacora.publicacion = ("no se reemplazaron los resultados publicados; "
                                "se conserva la versión aprobada anterior")
        tabla_controles = reporte.tabla() if reporte else pd.DataFrame()
        metadatos["detenido_en"] = str(exc)
        carpeta = pub.guardar_ejecucion(cfg, run_id, None, None, tabla_controles, metadatos)
        vigente = pub.version_aprobada(cfg)
        bitacora.info("versión vigente: " +
                      (vigente["run_id"] if vigente
                       else "ninguna; las estimaciones no están disponibles"))
        return bitacora.cerrar("detenida", {"carpeta": str(carpeta)}, historial)

    except Exception as exc:                        # error inesperado: mismo trato
        bitacora.error(f"error inesperado: {exc!r}")
        bitacora.publicacion = "no se reemplazaron los resultados publicados"
        metadatos["error"] = repr(exc)
        try:
            pub.guardar_ejecucion(cfg, run_id, None, None,
                                  reporte.tabla() if reporte else pd.DataFrame(), metadatos)
        except Exception:
            pass
        return bitacora.cerrar("fallida", {}, historial)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Flujo reproducible del Observatorio de Inteligencia Turística.")
    p.add_argument("--config", default=None,
                   help="archivo de configuración (por defecto 06_mlops/config/pipeline.yaml)")
    p.add_argument("--solo-validar", action="store_true",
                   help="corre lectura y controles de calidad, y termina")
    p.add_argument("--sin-publicar", action="store_true",
                   help="ejecuta todo sin tocar la versión aprobada")
    p.add_argument("--restaurar", metavar="RUN_ID", default=None,
                   help="deja vigente una versión aprobada anterior y termina")
    args = p.parse_args(argv)

    cfg = config_mod.cargar(args.config)

    if args.restaurar:
        bitacora = reg.Bitacora(f"restaurar-{args.restaurar}",
                                cfg.salida("logs_dir") / f"restaurar-{args.restaurar}.log")
        try:
            puntero = pub.restaurar(cfg, args.restaurar, bitacora)
            bitacora.publicacion = f"restaurada ({puntero['run_id']})"
            bitacora.cerrar("restaurada", {}, cfg.salida("historial"))
            return 0
        except FileNotFoundError as exc:
            bitacora.error(str(exc))
            bitacora.cerrar("fallida", {}, cfg.salida("historial"))
            return 2

    resumen = ejecutar(cfg, solo_validar=args.solo_validar, publicar=not args.sin_publicar)
    return 0 if resumen["estado"] in {"aprobada", "aprobada_sin_publicar", "validado"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
