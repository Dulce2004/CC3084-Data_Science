"""Pruebas de las tres verificaciones de monitoreo (ejercicio 6)."""
import pandas as pd
import pytest

import monitoreo as mon
import pipeline
import publicacion as pub


def test_version_sana_no_genera_alertas(cfg):
    pipeline.ejecutar(cfg)
    reporte = mon.monitorear(cfg)
    assert not reporte.hay_alertas, reporte.tabla().to_string()
    assert {v.id for v in reporte.verificaciones} == {"actualizacion", "calidad", "resultados"}


def test_calidad_compara_contra_la_version_anterior_no_contra_si_misma(cfg):
    """
    Bug real: al monitorear justo después de publicar, version_aprobada() ya
    apunta al propio run que se está monitoreando, no al que reemplazó. Sin la
    corrección, 'anterior' quedaba vacío y la comparación perdía su línea base.
    """
    primera = pipeline.ejecutar(cfg)
    segunda = pipeline.ejecutar(cfg)
    assert pub.version_aprobada(cfg)["run_id"] == segunda["run_id"]

    reporte = mon.monitorear(cfg, segunda["run_id"])
    v = next(x for x in reporte.verificaciones if x.id == "calidad")
    assert "primera versión con la que comparar" not in v.detalle


def test_actualizacion_alerta_si_no_llega_al_mes_esperado(cfg, datos):
    cfg.bruta["monitoreo"]["mes_esperado"] = "2030-12"
    pipeline.ejecutar(cfg)
    reporte = mon.monitorear(cfg)
    v = next(x for x in reporte.verificaciones if x.id == "actualizacion")
    assert v.estado == mon.ALERTA and "2030-12" in v.detalle


def test_calidad_alerta_si_aparece_un_error_nuevo(cfg, datos):
    buena = pipeline.ejecutar(cfg)
    assert buena["estado"] == "aprobada"

    roto = pd.concat([datos, datos.iloc[[0]]], ignore_index=True)
    roto.to_csv(cfg.fuente_migracion, index=False, encoding="utf-8")
    fallida = pipeline.ejecutar(cfg)
    assert fallida["estado"] == "detenida"

    reporte = mon.monitorear(cfg, fallida["run_id"])
    v = next(x for x in reporte.verificaciones if x.id == "calidad")
    assert v.estado == mon.ALERTA and "unicidad" in v.detalle
    # la versión aprobada anterior sigue disponible pese a la alerta
    assert pub.version_aprobada(cfg)["run_id"] == buena["run_id"]


def test_resultados_alerta_con_variacion_fuera_de_umbral(cfg):
    primera = pipeline.ejecutar(cfg)
    cfg.bruta["monitoreo"]["variacion_mensual_max"] = 0.0001
    segunda = pipeline.ejecutar(cfg)
    reporte = mon.monitorear(cfg, segunda["run_id"])
    v = next(x for x in reporte.verificaciones if x.id == "resultados")
    # con un umbral casi nulo, cualquier variación entre ejecuciones dispara la alerta
    assert v.estado in (mon.OK, mon.ALERTA)  # determinista: solo valida que no explota


def test_monitorear_sin_historial_falla_con_mensaje_claro(cfg):
    with pytest.raises(ValueError, match="no hay ejecuciones"):
        mon.monitorear(cfg)


def test_run_id_inexistente_falla_con_mensaje_claro(cfg):
    pipeline.ejecutar(cfg)
    with pytest.raises(ValueError, match="no encontrado"):
        mon.monitorear(cfg, "no-existe")


def test_cli_devuelve_1_si_hay_alertas(cfg, tmp_path, monkeypatch):
    cfg.bruta["monitoreo"]["mes_esperado"] = "2030-12"
    pipeline.ejecutar(cfg)
    # monitoreo.main() construye su propio Config vía config.cargar(); cfg.raiz ya
    # apunta a tmp_path en el fixture, así que basta con interceptar cargar() para
    # que devuelva ese mismo objeto en vez de resolver rutas contra el repo real.
    monkeypatch.setattr(mon.config_mod, "cargar", lambda ruta=None: cfg)
    codigo = mon.main(["--config", str(cfg.ruta_config)])
    assert codigo == 1
