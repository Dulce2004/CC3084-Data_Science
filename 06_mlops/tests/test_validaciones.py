"""
Pruebas de los seis controles de calidad.

Cada control se prueba dos veces: sobre la copia sintética sana (debe pasar) y
sobre la misma copia con un problema introducido a propósito (debe detectarlo y
detener la ejecución). Es la prueba controlada que pide el enunciado, hecha con
datos sintéticos y sin tocar la fuente institucional.
"""
import pandas as pd
import pytest

import validaciones as val


# ------------------------------------------------------------------ caso sano
def test_fuente_sana_pasa_los_seis_controles(datos, cfg):
    reporte = val.ejecutar(datos, cfg)
    assert len(reporte.resultados) == 6
    assert not reporte.detiene, [r.como_fila() for r in reporte.errores]
    assert reporte.resumen()["n_errores"] == 0


def test_cada_control_declara_nivel_y_estado(datos, cfg):
    tabla = val.ejecutar(datos, cfg).tabla()
    assert list(tabla.columns) == ["control", "verifica", "nivel", "estado",
                                   "detiene_ejecucion", "filas_afectadas", "detalle"]
    assert set(tabla["nivel"]) <= {"critico", "advertencia"}
    assert set(tabla["estado"]) <= {"ok", "advertencia", "error"}
    assert tabla["verifica"].str.len().gt(10).all()


# ------------------------------------------------- control 1: columnas y tipos
def test_control1_detiene_si_falta_una_columna_indispensable(datos, cfg):
    r = val.control_columnas_tipos(datos.drop(columns=["Viajero"]), cfg)
    assert r.estado == val.ERROR and r.detiene
    assert "Viajero" in r.detalle


def test_control1_detiene_si_el_tipo_es_incompatible(datos, cfg):
    malo = datos.copy()
    malo["Viajero"] = malo["Viajero"].astype(object)
    malo.loc[malo.index[:5], "Viajero"] = "sin dato"
    r = val.control_columnas_tipos(malo, cfg)
    assert r.estado == val.ERROR and r.detiene


def test_control1_advierte_por_columna_nueva(datos, cfg):
    r = val.control_columnas_tipos(datos.assign(ColumnaNueva=1), cfg)
    assert r.estado == val.ADV and not r.detiene


# ------------------------------------------------ control 2: fechas y cobertura
def test_control2_detiene_con_fecha_invalida(datos, cfg):
    malo = datos.copy()
    malo.loc[malo.index[0], "Mes cod"] = 13
    r = val.control_fechas_cobertura(malo, cfg)
    assert r.estado == val.ERROR and r.detiene


def test_control2_detiene_si_falta_un_mes_de_cobertura(datos, cfg):
    falta = datos.loc[~((datos["Año"] == 2024) & (datos["Mes cod"] == 3))]
    r = val.control_fechas_cobertura(falta, cfg)
    assert r.estado == val.ERROR and r.detiene
    assert "2024-03" in r.detalle


def test_control2_no_convierte_meses_sin_cobertura_en_cero(datos, cfg):
    """El mes ausente se reporta; nunca aparece como un cero observado."""
    falta = datos.loc[~((datos["Año"] == 2024) & (datos["Mes cod"] == 3))]
    r = val.control_fechas_cobertura(falta, cfg)
    assert "No se rellenan con cero" in r.detalle


# ------------------------------------------------------- control 3: unicidad
def test_control3_detiene_con_clave_duplicada(datos, cfg):
    dup = pd.concat([datos, datos.iloc[[0]]], ignore_index=True)
    r = val.control_unicidad(dup, cfg)
    assert r.estado == val.ERROR and r.detiene
    assert r.filas_afectadas >= 2


def test_control3_usa_la_clave_de_la_configuracion(datos, cfg):
    r = val.control_unicidad(datos, cfg)
    assert "Agrupación Residencia" in r.detalle


# ------------------------------------------------ control 4: valores faltantes
def test_control4_detiene_si_falta_un_valor_indispensable(datos, cfg):
    malo = datos.copy()
    malo.loc[malo.index[3], "Viajero"] = None
    r = val.control_valores_faltantes(malo, cfg)
    assert r.estado == val.ERROR and r.detiene


def test_control4_permite_las_ausencias_declaradas(datos, cfg):
    """'País' está en nulos_permitidos: una ausencia no debe detener el flujo."""
    malo = datos.copy()
    malo.loc[malo.index[:3], "País"] = None
    r = val.control_valores_faltantes(malo, cfg)
    assert not r.detiene


# ------------------------------------- control 5: valores y denominadores
def test_control5_detiene_con_conteos_negativos(datos, cfg):
    malo = datos.copy()
    malo.loc[malo.index[10], "Viajero"] = -5
    r = val.control_valores_denominadores(malo, cfg)
    assert r.estado == val.ERROR and r.detiene
    assert "negativos" in r.detalle


def test_control5_detiene_con_denominador_cero_en_la_evaluacion(datos, cfg):
    """Sin turistas en el puesto base, el factor de reparto no tiene denominador."""
    malo = datos.copy()
    evaluado = (malo["Año"] == 2026) & (malo["Mes cod"] == 2)
    base = evaluado & malo["Frontera"].eq("01 La Aurora") & malo["Tipo de Viajero"].eq("Turista")
    malo.loc[base, "Viajero"] = 0.0
    r = val.control_valores_denominadores(malo, cfg)
    assert r.estado == val.ERROR and r.detiene
    assert "2026-02" in r.detalle


def test_control5_solo_advierte_si_el_denominador_cero_esta_fuera_de_la_evaluacion(datos, cfg):
    malo = datos.copy()
    fuera = (malo["Año"] == 2015) & (malo["Mes cod"] == 4)
    base = fuera & malo["Frontera"].eq("01 La Aurora") & malo["Tipo de Viajero"].eq("Turista")
    malo.loc[base, "Viajero"] = 0.0
    r = val.control_valores_denominadores(malo, cfg)
    assert r.estado == val.ADV and not r.detiene


# ------------------------------------------- control 6: regla metodológica
def test_control6_detiene_si_la_fuente_trae_granularidad_diaria(datos, cfg):
    r = val.control_regla_metodologica(datos.assign(**{"Día": 1}), cfg)
    assert r.estado == val.ERROR and r.detiene
    assert "presencias diarias" in r.detalle


def test_control6_detiene_si_un_anio_tiene_mas_de_doce_meses(datos, cfg):
    malo = datos.copy()
    malo.loc[malo.index[:4], "Mes cod"] = 13
    r = val.control_regla_metodologica(malo, cfg)
    assert r.estado == val.ERROR and r.detiene


def test_control6_detiene_si_el_reparto_duplicaria_visitantes(datos, cfg):
    """El puesto base del reparto no puede estar también entre los del destino."""
    cfg.bruta["procedimiento"]["puestos_destino"] = ["02 Mundo Maya", "01 La Aurora"]
    r = val.control_regla_metodologica(datos, cfg)
    assert r.estado == val.ERROR and r.detiene
    assert "dos veces" in r.detalle


def test_control6_detiene_si_falta_un_puesto_del_destino(datos, cfg):
    sin_melchor = datos.loc[datos["Frontera"] != "20 Melchor de Mencos"]
    r = val.control_regla_metodologica(sin_melchor, cfg)
    assert r.estado == val.ERROR and r.detiene


def test_control6_detiene_con_factor_de_reparto_invalido(datos, cfg):
    cfg.bruta["procedimiento"]["factor_reparto"] = 1.4
    r = val.control_regla_metodologica(datos, cfg)
    assert r.estado == val.ERROR and r.detiene


def test_control6_advierte_fuera_del_rango_de_sensibilidad(datos, cfg):
    cfg.bruta["procedimiento"]["factor_reparto"] = 0.5
    r = val.control_regla_metodologica(datos, cfg)
    assert r.estado == val.ADV and not r.detiene


# ----------------------------------------------------------- robustez
def test_un_control_roto_queda_en_error_y_no_aprueba(datos, cfg, monkeypatch):
    def explota(df, c):
        raise RuntimeError("control roto")
    monkeypatch.setattr(val, "CONTROLES", [explota])
    monkeypatch.setattr(explota, "__name__", "unicidad", raising=False)
    reporte = val.ejecutar(datos, cfg)
    assert reporte.resumen()["n_errores"] == 1


@pytest.mark.parametrize("control", [c.__name__ for c in val.CONTROLES])
def test_el_nivel_de_cada_control_viene_de_la_configuracion(control, cfg):
    assert cfg.nivel_control(control.replace("control_", "")) == "critico"
