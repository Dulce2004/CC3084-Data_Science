"""Datos, paleta y ayudas compartidas por todas las páginas del tablero."""
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA = Path(__file__).resolve().parents[1] / "data" / "resultados"

# Paleta con significado fijo en toda la aplicación (ver documento de diseño, sección 4).
# Tonos fríos = datos observados; tonos cálidos y línea punteada = estimaciones de modelos.
OBSERVADO = "#2889E2"        # azul: lo que realmente pasó
CONTEXTO = "#8A94A6"         # gris: referencia (entrenamiento, año anterior)
SELECCION = "#1F3A5F"        # azul marino: el elemento que el usuario eligió
ESTIMADO = "#C4571A"         # naranja: pronóstico / estimación
VIAS = {"Aérea": "#2889E2", "Terrestre": "#2A9D8F", "Marítima": "#5B5FC7"}
MODELOS = {
    "Prophet": "#C4571A",
    "SES": "#B7791F",
    "Holt-Winters": "#A23B72",
    "Seasonal naive": "#8C6D46",
    "SARIMA manual": "#D1495B",
    "SARIMA grid-AIC": "#7A2E3A",
    "LSTM": "#6D4C9F",
}
ESCALA_VOLUMEN = "Blues"   # cuánto: más oscuro = más viajeros
ESCALA_ERROR = "Oranges"   # cuánto se equivoca un modelo: más oscuro = peor
MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
TIPOS = ["Turista", "Excursionista", "Viajero", "Cruceristas"]
ULTIMO_MES = pd.Timestamp("2026-06-01")


@st.cache_data
def cargar(nombre: str) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"{nombre}.csv")
    if "fecha" in df:
        df["fecha"] = pd.to_datetime(df["fecha"])
    return df


def filtrar(df: pd.DataFrame, usar_tipo: bool = True) -> pd.DataFrame:
    """Aplica los filtros globales de la barra lateral (período y tipo de viajero)."""
    a0, a1 = st.session_state.get("periodo", (2009, 2026))
    df = df[df["fecha"].dt.year.between(a0, a1)]
    if usar_tipo and "Tipo de Viajero" in df:
        df = df[df["Tipo de Viajero"].isin(st.session_state.get("tipos", TIPOS))]
    return df


def sin_datos(df: pd.DataFrame, que: str) -> bool:
    """Muestra un aviso y devuelve True cuando la selección no tiene información."""
    if df.empty or ("Viajero" in df and df["Viajero"].sum() == 0):
        st.info(f"No hay datos de {que} con los filtros actuales. Amplíe el período o agregue tipos de "
                "viajero en la barra lateral.", icon=":material/filter_alt_off:")
        return True
    return False


def seleccion(key: str, campo: str = "x"):
    """Valor del primer punto que el usuario seleccionó en la gráfica `key` (o None)."""
    try:
        puntos = st.session_state[key]["selection"]["points"]
    except (KeyError, TypeError):
        return None
    return puntos[0].get(campo) if puntos else None


def visibles(opciones: list[str], key: str) -> list[str]:
    """Deja que el usuario elija qué gráficas de la página quiere ver."""
    return st.pills("Gráficas visibles", opciones, selection_mode="multi", default=opciones, key=key) or []


def estilo(fig: go.Figure, alto: int = 420, titulo_y: str = "Viajeros") -> go.Figure:
    fig.update_layout(height=alto, margin=dict(l=10, r=10, t=40, b=10), font=dict(size=14),
                      hoverlabel=dict(font_size=14), legend=dict(orientation="h", y=1.08, x=0, title=None),
                      separators=".,")
    fig.update_yaxes(title=titulo_y, tickformat=",.0f")
    return fig


def miles(x: float) -> str:
    return f"{x:,.0f}"


def nota(observado: str, interpretacion: str | None = None):
    """Separa lo que muestran los datos de la lectura que hacemos de ellos."""
    st.markdown(f":material/visibility: **Dato observado.** {observado}")
    if interpretacion:
        st.markdown(f":material/lightbulb: **Interpretación.** {interpretacion}")
