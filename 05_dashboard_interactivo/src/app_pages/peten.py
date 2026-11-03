"""Petén / Mundo Maya — consulta la versión aprobada que produce el flujo MLOps
(Laboratorio 11). No ajusta nada aquí: solo lee lo que el pipeline ya publicó."""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from comun import CONTEXTO, ESTIMADO, OBSERVADO, estilo, miles, nota

REPO_ROOT = Path(__file__).resolve().parents[3]
APROBADO_DIR = REPO_ROOT / "06_mlops" / "data" / "aprobado"
PUNTERO = APROBADO_DIR / "version_actual.json"


@st.cache_data(ttl=60)
def version_vigente():
    """Lee el puntero a la versión aprobada. None si todavía no existe ninguna."""
    if not PUNTERO.exists():
        return None
    try:
        return json.loads(PUNTERO.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


@st.cache_data(ttl=60)
def cargar_aprobado(carpeta: str):
    base = REPO_ROOT / carpeta
    est = pd.read_csv(base / "estimaciones.csv", parse_dates=False)
    est["fecha"] = pd.to_datetime(est["periodo"], format="%Y-%m")
    return est


st.title("Petén / Mundo Maya")
st.caption("¿Cuántos viajeros internacionales llegan al destino y qué tan confiable es la estimación?")

vigente = version_vigente()

if vigente is None:
    st.warning("**Las estimaciones no están disponibles.** Todavía no existe una versión aprobada del "
               "procedimiento para este destino. Esta página se actualizará automáticamente cuando el "
               "flujo de datos (Laboratorio 11) publique la primera versión aprobada.",
               icon=":material/report:")
    st.stop()

est = cargar_aprobado(vigente["carpeta"])
metrica = vigente.get("metrica_validacion") or {}
fecha_pub = pd.to_datetime(vigente["fecha_publicacion"]).strftime("%d de %B de %Y, %H:%M UTC")

# ------------------------------------------------------------ tarjetas de estado
c1, c2, c3 = st.columns(3)
c1.metric("Última actualización", fecha_pub)
c2.metric("Error típico de la estimación (MAE)",
         f"±{miles(metrica.get('MAE', float('nan')))} viajeros/mes")
c3.metric("Versión del procedimiento", vigente.get("version_procedimiento", "—"))

st.info(
    f":material/verified: Estas estimaciones provienen de una **versión aprobada automáticamente** "
    f"(identificador `{vigente['run_id']}`): pasó los seis controles de calidad y la regla de "
    f"aceptación del flujo de datos antes de publicarse aquí. El error típico (MAE) se calculó "
    f"comparando la estimación contra meses ya observados que se ocultaron del ajuste "
    f"({metrica.get('meses_evaluados', '—')} meses de validación) — es una medida de qué tan "
    f"cerca estuvo el procedimiento cuando se le pidió adivinar meses que ya conocíamos.",
    icon=":material/info:")

# ------------------------------------------------------------ observado vs. estimado
with st.container(border=True):
    st.subheader("Viajeros observados y estimados por mes")
    st.caption("Línea azul sólida = lo que ya se observó. Línea naranja punteada = estimación del "
               "procedimiento para ese mismo mes (se calcula igual, se observe o no, para poder "
               "comparar). Los meses sin información **no se muestran como cero**.")

    fig = go.Figure()
    con_obs = est.dropna(subset=["valor_observado"])
    if len(con_obs):
        fig.add_trace(go.Scatter(x=con_obs["fecha"], y=con_obs["valor_observado"], name="Observado",
                                 line=dict(color=OBSERVADO, width=3),
                                 hovertemplate="Observado: %{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=est["fecha"], y=est["valor"], name="Estimado",
                             line=dict(color=ESTIMADO, width=2.5, dash="dash"),
                             hovertemplate="Estimado: %{y:,.0f}<extra></extra>"))
    fig.update_xaxes(dtick="M1", tickformat="%b %Y")
    st.plotly_chart(estilo(fig, 400), alt="Viajeros observados y estimados en Petén / Mundo Maya")

    nota(
        f"En el período evaluado ({metrica.get('meses_evaluados', '—')} meses) la estimación se "
        f"desvió en promedio {miles(metrica.get('MAE', float('nan')))} viajeros por mes "
        f"({metrica.get('error_relativo', 0):.0%} del volumen típico del destino).",
        "Un error bajo en estos meses no garantiza el mismo desempeño en el futuro: el procedimiento "
        "se reevalúa en cada actualización del flujo de datos."
    )

# ------------------------------------------------------------ tabla
with st.expander("Ver la tabla de estimaciones", icon=":material/table:"):
    tabla = est[["periodo", "valor_observado", "valor"]].rename(
        columns={"periodo": "Mes", "valor_observado": "Observado", "valor": "Estimado"})
    st.dataframe(tabla, hide_index=True,
                column_config={
                    "Observado": st.column_config.NumberColumn(format="localized"),
                    "Estimado": st.column_config.NumberColumn(format="localized"),
                }, alt="Tabla de estimaciones aprobadas para Petén / Mundo Maya")

# ------------------------------------------------------------ limitaciones
st.warning(
    "**Limitaciones.** La estimación combina las entradas observadas en los puestos del destino con "
    "una proporción fija (*factor de reparto*) de los turistas que llegan por vía aérea a La Aurora, "
    "un supuesto del observatorio pendiente de calibrar con el registro de visitantes del Parque "
    "Nacional Tikal — no es un conteo directo de llegadas al destino. Los meses más recientes pueden "
    "no tener valor observado todavía. Los detalles técnicos de cada ejecución (parámetros, huella de "
    "los datos, commit de código) se documentan en MLflow y en `06_mlops/`, no en esta página.",
    icon=":material/warning:")
