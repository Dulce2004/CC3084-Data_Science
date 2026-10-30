"""
Pruebas del flujo completo sobre la copia sintética.

Cubren lo que el enunciado pide demostrar del pipeline: que corre de principio a
fin con un solo comando, que dos ejecuciones con los mismos datos y la misma
configuración producen resultados equivalentes, que un control crítico en error
detiene la publicación sin tocar la versión aprobada, y que una versión anterior
se puede restaurar.
"""
import json

import numpy as np
import pandas as pd
import pytest

import config as config_mod
import evaluacion as eva
import pipeline
import preparacion as prep
import publicacion as pub


# ------------------------------------------------------------ preparación
def test_la_serie_del_destino_aplica_la_formula(datos, cfg):
    tabla = prep.construir(datos, cfg)
    f = cfg.get("procedimiento", "factor_reparto")
    esperado = tabla["directas"] + f * tabla["base_reparto"]
    pd.testing.assert_series_equal(tabla["viajeros_destino"], esperado, check_names=False)


def test_un_mes_sin_cobertura_queda_vacio_no_en_cero(datos, cfg):
    falta = datos.loc[~((datos["Año"] == 2024) & (datos["Mes cod"] == 3))]
    serie = prep.serie_destino(prep.construir(falta, cfg))
    assert pd.isna(serie.loc["2024-03-01"])
    assert serie.loc["2024-03-01"] != 0


def test_la_particion_respeta_el_orden_temporal(datos, cfg):
    serie = prep.serie_destino(prep.construir(datos, cfg))
    ajuste, evaluacion = prep.partir(serie, cfg)
    assert ajuste.index.max() < evaluacion.index.min()
    assert len(evaluacion) == len(cfg.meses_evaluacion())


def test_el_ajuste_rechaza_meses_sin_valor_observado(datos, cfg):
    import modelo as mdl
    serie = prep.serie_destino(prep.construir(datos, cfg))
    ajuste, _ = prep.partir(serie, cfg)
    ajuste = ajuste.copy()
    ajuste.iloc[5] = np.nan
    with pytest.raises(ValueError, match="no se rellenan con cero"):
        mdl.ajustar_y_estimar(ajuste, cfg.meses_evaluacion(), cfg)


# ------------------------------------------------------------ flujo completo
def test_el_flujo_corre_de_principio_a_fin_y_publica(cfg):
    resumen = pipeline.ejecutar(cfg)
    assert resumen["estado"] == "aprobada", resumen["errores"]
    assert resumen["etapas"] == ["lectura", "validacion", "preparacion", "ajuste",
                                 "evaluacion", "almacenamiento", "publicacion"]
    assert pub.version_aprobada(cfg)["run_id"] == resumen["run_id"]


def test_el_archivo_de_estimaciones_trae_los_metadatos_exigidos(cfg):
    resumen = pipeline.ejecutar(cfg)
    carpeta = cfg.salida("aprobado_dir") / resumen["run_id"]
    est = pd.read_csv(carpeta / "estimaciones.csv")
    for columna in ["destino", "periodo", "valor", "unidad", "run_id",
                    "fecha_generacion", "version_procedimiento"]:
        assert columna in est.columns
    assert len(est) == len(cfg.meses_evaluacion())
    assert est["valor"].notna().all()


def test_la_bitacora_y_el_historial_registran_la_ejecucion(cfg):
    resumen = pipeline.ejecutar(cfg)
    bitacora = (cfg.salida("logs_dir") / f"{resumen['run_id']}.log").read_text(encoding="utf-8")
    assert "[etapa] lectura" in bitacora and "estado final: aprobada" in bitacora
    historial = [json.loads(l) for l in
                 cfg.salida("historial").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert historial[-1]["run_id"] == resumen["run_id"]
    assert historial[-1]["inicio"] and historial[-1]["fin"]


def test_los_metadatos_identifican_datos_codigo_y_parametros(cfg):
    resumen = pipeline.ejecutar(cfg)
    meta = json.loads((cfg.salida("resultados_dir") / resumen["run_id"] /
                       "metadatos.json").read_text(encoding="utf-8"))
    assert len(meta["fuente"]["huella_sha256"]) == 64
    assert meta["commit_git"]
    assert meta["configuracion"]["huella_sha256"]
    assert meta["procedimiento"]["factor_reparto"] == cfg.get("procedimiento", "factor_reparto")
    assert meta["dependencias"]["pandas"]
    assert meta["periodo_ajuste"] and meta["periodo_evaluacion"]


def test_dos_ejecuciones_iguales_dan_resultados_equivalentes(cfg):
    """Mismos datos y misma configuración: las estimaciones deben coincidir."""
    a = pipeline.ejecutar(cfg)
    b = pipeline.ejecutar(cfg)
    assert a["run_id"] != b["run_id"]
    base = cfg.salida("resultados_dir")
    va = pd.read_csv(base / a["run_id"] / "estimaciones.csv")["valor"]
    vb = pd.read_csv(base / b["run_id"] / "estimaciones.csv")["valor"]
    tolerancia = float(cfg.get("aceptacion", "tolerancia_reproducibilidad"))
    assert np.abs(va - vb).max() <= tolerancia


# ------------------------------------------- fallo controlado y recuperación
@pytest.mark.parametrize("problema", ["columna_ausente", "clave_duplicada", "denominador_cero"])
def test_un_problema_introducido_detiene_el_flujo_sin_tocar_lo_publicado(cfg, datos, problema):
    """
    Prueba controlada: primero se publica una versión buena, después se introduce
    un problema en la copia de los datos y se comprueba que el flujo se detiene y
    que la versión aprobada sigue disponible.
    """
    buena = pipeline.ejecutar(cfg)
    assert buena["estado"] == "aprobada"

    roto = datos.copy()
    if problema == "columna_ausente":
        roto = roto.drop(columns=["Viajero"])
    elif problema == "clave_duplicada":
        roto = pd.concat([roto, roto.iloc[[0]]], ignore_index=True)
    else:
        base = (roto["Año"].eq(2026) & roto["Mes cod"].eq(2)
                & roto["Frontera"].eq("01 La Aurora") & roto["Tipo de Viajero"].eq("Turista"))
        roto.loc[base, "Viajero"] = 0.0
    roto.to_csv(cfg.fuente_migracion, index=False, encoding="utf-8")

    fallida = pipeline.ejecutar(cfg)
    assert fallida["estado"] == "detenida"
    assert "no se reemplazaron los resultados publicados" in fallida["publicacion"]
    assert "publicacion" not in fallida["etapas"]
    assert pub.version_aprobada(cfg)["run_id"] == buena["run_id"]
    controles = pd.read_csv(cfg.salida("resultados_dir") / fallida["run_id"] / "controles.csv")
    assert (controles["estado"] == "error").any()


def test_tras_corregir_el_problema_el_flujo_vuelve_a_publicar(cfg, datos):
    buena = pipeline.ejecutar(cfg)
    pd.concat([datos, datos.iloc[[0]]], ignore_index=True).to_csv(
        cfg.fuente_migracion, index=False, encoding="utf-8")
    assert pipeline.ejecutar(cfg)["estado"] == "detenida"
    datos.to_csv(cfg.fuente_migracion, index=False, encoding="utf-8")
    nueva = pipeline.ejecutar(cfg)
    assert nueva["estado"] == "aprobada"
    assert pub.version_aprobada(cfg)["run_id"] == nueva["run_id"] != buena["run_id"]


def test_se_conserva_y_se_restaura_una_version_anterior(cfg):
    primera = pipeline.ejecutar(cfg)
    segunda = pipeline.ejecutar(cfg)
    assert pub.versiones_aprobadas(cfg) == sorted([primera["run_id"], segunda["run_id"]])
    assert pub.version_aprobada(cfg)["run_id"] == segunda["run_id"]

    pub.restaurar(cfg, primera["run_id"])
    vigente = pub.version_aprobada(cfg)
    assert vigente["run_id"] == primera["run_id"] and vigente["restaurada"]
    assert (cfg.salida("aprobado_dir") / primera["run_id"] / "estimaciones.csv").exists()


def test_restaurar_una_version_inexistente_falla_con_mensaje_claro(cfg):
    pipeline.ejecutar(cfg)
    with pytest.raises(FileNotFoundError, match="no hay una versión aprobada"):
        pub.restaurar(cfg, "20200101T000000Z-aaaaaaa")


def test_sin_version_aprobada_se_informa_que_no_hay_estimaciones(cfg, datos):
    """Si nunca se aprobó una versión, el flujo lo dice en lugar de inventar una."""
    datos.drop(columns=["Viajero"]).to_csv(cfg.fuente_migracion, index=False, encoding="utf-8")
    resumen = pipeline.ejecutar(cfg)
    assert resumen["estado"] == "detenida"
    assert pub.version_aprobada(cfg) is None
    bitacora = (cfg.salida("logs_dir") / f"{resumen['run_id']}.log").read_text(encoding="utf-8")
    assert "las estimaciones no están disponibles" in bitacora


# ------------------------------------------------- regla de aceptación
def test_la_regla_rechaza_una_version_que_no_cumple_el_umbral(cfg):
    cfg.bruta["aceptacion"]["mae_max"] = 1.0        # imposible de cumplir
    resumen = pipeline.ejecutar(cfg)
    assert resumen["estado"] == "rechazada"
    assert resumen["publicacion"].startswith("rechazada")
    assert pub.version_aprobada(cfg) is None


def test_el_flujo_no_publica_con_la_opcion_sin_publicar(cfg):
    resumen = pipeline.ejecutar(cfg, publicar=False)
    assert resumen["estado"] == "aprobada_sin_publicar"
    assert pub.version_aprobada(cfg) is None


def test_solo_validar_se_detiene_despues_de_los_controles(cfg):
    resumen = pipeline.ejecutar(cfg, solo_validar=True)
    assert resumen["etapas"] == ["lectura", "validacion", "almacenamiento"]
    assert pub.version_aprobada(cfg) is None


def test_mape_no_se_calcula_con_observados_en_cero():
    y = pd.Series([0.0, 100.0], index=pd.date_range("2026-01", periods=2, freq="MS"))
    m = eva.metricas(y, pd.Series([10.0, 90.0], index=y.index))
    assert pd.isna(m["MAPE"]) and "cero" in m["nota_mape"]


def test_se_documentan_las_limitaciones_de_la_validacion_temporal(cfg):
    limitaciones = eva.limitaciones_validacion(cfg)
    assert len(limitaciones) >= 3
    assert any("factor de reparto" in t for t in limitaciones)


# ------------------------------------------------------------ configuración
def test_la_configuracion_rechaza_evaluar_antes_de_terminar_el_ajuste(tmp_path, cfg):
    import yaml
    bruta = dict(cfg.bruta)
    bruta["periodos"] = dict(bruta["periodos"])
    bruta["periodos"]["evaluacion_inicio"] = bruta["periodos"]["ajuste_fin"]
    ruta = tmp_path / "mala.yaml"
    ruta.write_text(yaml.safe_dump(bruta, allow_unicode=True), encoding="utf-8")
    with pytest.raises(config_mod.ErrorConfiguracion, match="validación temporal"):
        config_mod.cargar(ruta)


def test_la_configuracion_rechaza_un_factor_de_reparto_fuera_de_rango(tmp_path, cfg):
    import yaml
    bruta = dict(cfg.bruta)
    bruta["procedimiento"] = dict(bruta["procedimiento"])
    bruta["procedimiento"]["factor_reparto"] = 0
    ruta = tmp_path / "mala.yaml"
    ruta.write_text(yaml.safe_dump(bruta, allow_unicode=True), encoding="utf-8")
    with pytest.raises(config_mod.ErrorConfiguracion, match="factor_reparto"):
        config_mod.cargar(ruta)


def test_la_configuracion_exige_las_claves_obligatorias(tmp_path):
    import yaml
    ruta = tmp_path / "vacia.yaml"
    ruta.write_text(yaml.safe_dump({"version_procedimiento": "0.1"}), encoding="utf-8")
    with pytest.raises(config_mod.ErrorConfiguracion, match="faltan claves"):
        config_mod.cargar(ruta)


def test_los_metadatos_son_json_valido_sin_nan(cfg):
    """`NaN` e `Infinity` no son JSON válido: deben quedar como null."""
    resumen = pipeline.ejecutar(cfg)
    crudo = (cfg.salida("resultados_dir") / resumen["run_id"] /
             "metadatos.json").read_text(encoding="utf-8")
    assert "NaN" not in crudo and "Infinity" not in crudo
    json.loads(crudo)
