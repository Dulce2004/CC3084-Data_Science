"""
Registro de ejecuciones en MLflow Tracking (ejercicio 4).

Un run de MLflow por ejecución del pipeline, con lo que pide el enunciado:
  - identificador y fecha de ejecución; commit de Git;
  - identificación de las fuentes y su huella; períodos de ajuste y evaluación;
  - parámetros del procedimiento, métricas de evaluación y resultado de los
    controles de calidad;
  - archivos de estimaciones y configuración; parámetros del modelo ajustado.

Usa un tracking URI de archivo local por defecto (`06_mlops/mlruns/`), que es
"un medio autorizado" para consultar los registros sin depender de un servidor
siempre encendido; `docker-compose.yml` levanta además un servidor de MLflow
para consultarlos con la interfaz web (ver 06_mlops/README.md).

El registro en MLflow es un efecto adicional de la ejecución: si falla, no debe
tumbar el pipeline ni impedir que se guarden los resultados locales. Por eso
`registrar_ejecucion` atrapa sus propias excepciones y las deja como advertencia
en la bitácora.
"""
from __future__ import annotations

import os
from pathlib import Path

EXPERIMENTO = "observatorio-peten-mundo-maya"


def _tracking_uri(cfg) -> str:
    """
    MLFLOW_TRACKING_URI tiene prioridad (apunta al servidor en Docker); si no
    está definida, se usa una base SQLite local al repositorio. MLflow 3.x puso
    el almacén de archivos (`./mlruns`) en modo mantenimiento; SQLite es el
    backend soportado y además es un único archivo fácil de montar como volumen.
    """
    env = os.environ.get("MLFLOW_TRACKING_URI")
    if env:
        return env
    base = cfg.raiz / "06_mlops" / "mlruns"
    base.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{(base / 'mlflow.db').resolve().as_posix()}"


def registrar_ejecucion(cfg, run_id: str, estado: str, metadatos: dict,
                        carpeta_ejecucion: Path, bitacora=None) -> str | None:
    """
    Crea (o reutiliza) el experimento y registra un run con todo lo producido
    por la ejecución. Devuelve el run_id de MLflow, o None si no se pudo
    registrar (se deja constancia en la bitácora, nunca se detiene el pipeline).
    """
    try:
        import mlflow
    except ImportError:
        if bitacora is not None:
            bitacora.advertencia("[mlflow] paquete no instalado: no se registró esta ejecución")
        return None

    try:
        mlflow.set_tracking_uri(_tracking_uri(cfg))
        mlflow.set_experiment(EXPERIMENTO)

        with mlflow.start_run(run_name=run_id) as run:
            # -------------------------------------------------- identificación
            mlflow.set_tags({
                "run_id_pipeline": run_id,
                "estado": estado,
                "commit_git": metadatos.get("commit_git", ""),
                "destino": metadatos.get("destino", ""),
                "version_procedimiento": metadatos.get("version_procedimiento", ""),
            })

            fuente = metadatos.get("fuente") or {}
            mlflow.log_params({
                "fuente_ruta": fuente.get("ruta", ""),
                "fuente_sha256": fuente.get("huella_sha256", ""),
                "fuente_filas": fuente.get("filas", ""),
                "config_sha256": metadatos.get("configuracion", {}).get("huella_sha256", ""),
                "periodo_ajuste": metadatos.get("periodo_ajuste", ""),
                "periodo_evaluacion": metadatos.get("periodo_evaluacion", ""),
                "semilla": metadatos.get("semilla", ""),
            })

            proc = metadatos.get("procedimiento") or {}
            if proc:
                mlflow.log_params({f"procedimiento.{k}": v for k, v in proc.items()
                                   if not isinstance(v, (list, dict))})

            ajuste = metadatos.get("ajuste") or {}
            if ajuste:
                mlflow.log_param("modelo", ajuste.get("modelo", ""))
                mlflow.log_param("referencia", ajuste.get("referencia_nombre", ""))
                for k, v in (ajuste.get("parametros") or {}).items():
                    if isinstance(v, (int, float)):
                        mlflow.log_param(f"modelo.{k}", v)

            # -------------------------------------------------------- métricas
            metrica = metadatos.get("metrica_validacion") or {}
            for k, v in metrica.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, float(v))

            controles = metadatos.get("controles") or {}
            for k, v in controles.items():
                if isinstance(v, (int, float, bool)):
                    mlflow.log_metric(f"controles.{k}", float(v))

            aceptacion = metadatos.get("aceptacion") or {}
            if aceptacion:
                mlflow.log_metric("aceptacion.aprobada", float(aceptacion.get("aprobada", False)))
                mlflow.log_metric("aceptacion.supera_referencia",
                                  float(aceptacion.get("supera_referencia", False)))

            # -------------------------------------------------------- artefactos
            if carpeta_ejecucion and carpeta_ejecucion.exists():
                mlflow.log_artifacts(str(carpeta_ejecucion), artifact_path="expediente")

            if bitacora is not None:
                bitacora.info(f"[mlflow] run {run.info.run_id} registrado en experimento "
                              f"'{EXPERIMENTO}' ({mlflow.get_tracking_uri()})")
            return run.info.run_id

    except Exception as exc:                        # nunca debe tumbar el pipeline
        if bitacora is not None:
            bitacora.advertencia(f"[mlflow] no se pudo registrar la ejecución: {exc!r}")
        return None


def comparar_runs(cfg, n: int = 2):
    """
    Lee los últimos `n` runs del experimento desde MLflow y devuelve sus métricas,
    para comprobar que ejecuciones con los mismos datos producen resultados
    equivalentes (ejercicio 4). Pensado para usarse desde una notebook o la CLI.
    """
    import mlflow
    mlflow.set_tracking_uri(_tracking_uri(cfg))
    exp = mlflow.get_experiment_by_name(EXPERIMENTO)
    if exp is None:
        return None
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id],
                              order_by=["start_time DESC"], max_results=n)
    return runs
