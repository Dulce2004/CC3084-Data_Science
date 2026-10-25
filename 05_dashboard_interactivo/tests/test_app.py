"""Prueba de humo: cada página corre sin excepciones, también con filtros que dejan la selección vacía.

Desde 05_dashboard_interactivo/:  python -m pytest tests -q
"""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "src" / "app.py")
PAGINAS = ["inicio", "panorama", "mercados", "entradas", "pronosticos", "glosario"]


def correr(pagina, **estado):
    at = AppTest.from_file(APP, default_timeout=120)
    for k, v in estado.items():
        at.session_state[k] = v
    at.run()
    at.switch_page(f"app_pages/{pagina}.py").run()
    assert not at.exception, at.exception
    return at


@pytest.mark.parametrize("pagina", PAGINAS)
def test_pagina_corre(pagina):
    correr(pagina)


@pytest.mark.parametrize("pagina", ["panorama", "mercados", "entradas"])
def test_sin_datos_muestra_aviso(pagina):
    at = correr(pagina, tipos=[])
    assert any("No hay datos" in i.value for i in at.info)


@pytest.mark.parametrize("serie", ["Total", "Vía Aérea", "Cruceristas"])
def test_pronosticos_por_serie(serie):
    correr("pronosticos", serie_pron=serie)
