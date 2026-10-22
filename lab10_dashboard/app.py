"""
Observatorio de Turismo Sostenible · Guatemala
Lab 10 · Visualización interactiva (CC3084 Data Science, UVG 2026-II)

Ejecutar desde la raíz del repo:
    python lab10_dashboard/preparar_datos.py     # una sola vez
    streamlit run lab10_dashboard/app.py
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DATA = Path(__file__).resolve().parent / "data"

# ------------------------------------------------------------------
# Paleta (mismo significado en toda la app)
# ------------------------------------------------------------------
OBSERVADO = "#1F3A5F"   # azul marino: lo que realmente pasó
ESTIMADO = "#C4571A"    # naranja: estimaciones de modelos (nunca se usa para otra cosa)
NEUTRO = "#A7B1BE"      # gris azulado: contexto / elementos no seleccionados
SECUENCIAL = ["#EAF0F7", "#B7CFEA", "#6FA6DE", "#2889E2", "#1F3A5F"]  # un solo tono, claro→oscuro
CATEGORICO = ["#2889E2", "#1D9E75", "#8A5FBF", "#D4A017", "#C2185B"]  # orden fijo, validado CVD

COLOR_VIA = {"Aérea": CATEGORICO[0], "Terrestre": CATEGORICO[1], "Marítima": CATEGORICO[2]}
COLOR_TIPO = {"Turista": CATEGORICO[0], "Excursionista": CATEGORICO[1],
              "Viajero": CATEGORICO[2], "Cruceristas": CATEGORICO[3]}
MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
FUENTE = "Inter, 'Segoe UI', Roboto, sans-serif"

st.set_page_config(page_title="Observatorio de Turismo · Guatemala", page_icon="🧭", layout="wide")

st.markdown(f"""
<style>
  html, body, [class*="css"] {{ font-family: {FUENTE}; }}
  .block-container {{ padding-top: 1.6rem; max-width: 1350px; }}
  h1, h2, h3 {{ color: {OBSERVADO}; letter-spacing: -0.01em; }}
  .nota {{ color: #52514E; font-size: 0.92rem; margin-top: -0.4rem; }}
  .etiqueta-obs {{ background:#EAF0F7; color:{OBSERVADO}; padding:2px 8px; border-radius:6px;
                   font-size:0.8rem; font-weight:600; }}
  .etiqueta-est {{ background:#F8E6DC; color:#8A3A10; padding:2px 8px; border-radius:6px;
                   font-size:0.8rem; font-weight:600; }}
  div[data-testid="stMetric"] {{ background:#F4F7FB; border-radius:10px; padding:10px 14px; }}
</style>
""", unsafe_allow_html=True)


def estilo(fig, alto=360, leyenda=True):
    fig.update_layout(
        height=alto, margin=dict(l=10, r=10, t=30, b=10), font=dict(family=FUENTE, size=13),
        plot_bgcolor="white", paper_bgcolor="white", showlegend=leyenda,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=None),
        hoverlabel=dict(font_size=13), separators=".,",
    )
    fig.update_xaxes(showgrid=False, linecolor="#C9CED6")
    fig.update_yaxes(gridcolor="#EEF1F5", zeroline=False, tickformat="~s")
    return fig


def seleccion(evento, campo="x"):
    """Devuelve los valores seleccionados (clic/lazo) de un gráfico de plotly en Streamlit."""
    try:
        puntos = evento.selection.points
    except AttributeError:
        return []
    return [p.get(campo) for p in puntos if p.get(campo) is not None]


def sin_datos(msg="No hay datos para esta selección. Pruebe ampliando el rango de años o quitando filtros."):
    st.info(msg, icon="ℹ️")


# ------------------------------------------------------------------
# Datos
# ------------------------------------------------------------------
@st.cache_data
def cargar():
    agg = pd.read_csv(DATA / "agregado_mensual.csv", parse_dates=["fecha"])
    agg["Año"] = agg["fecha"].dt.year
    agg["Mes"] = agg["fecha"].dt.month
    pred = pd.read_csv(DATA / "predicciones.csv", parse_dates=["fecha"])
    met = pd.read_csv(DATA / "metricas.csv")
    return agg, pred, met


if not (DATA / "agregado_mensual.csv").exists():
    st.error("Faltan los datos. Ejecute primero: `python lab10_dashboard/preparar_datos.py`")
    st.stop()

agg, pred, met = cargar()

# ------------------------------------------------------------------
# Barra lateral: filtros globales
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Filtros")
    st.caption("Aplican a las pestañas de exploración.")
    a_min, a_max = int(agg["Año"].min()), int(agg["Año"].max())
    rango = st.slider("Años", a_min, a_max, (2015, a_max))
    vias = st.multiselect("Vía de entrada", list(COLOR_VIA), default=list(COLOR_VIA))
    tipos = st.multiselect("Tipo de viajero", list(COLOR_TIPO), default=["Turista", "Excursionista"],
                           help="Por defecto se excluye 'Viajero': desde 2023 dejó de contar tránsito "
                                "fronterizo, lo que produce una caída artificial.")
    st.divider()
    with st.expander("Glosario"):
        st.markdown("""
- **Turista:** visitante que pernocta al menos una noche.
- **Excursionista:** visitante que entra y sale el mismo día.
- **Crucerista:** pasajero de crucero que desembarca.
- **Estimación / pronóstico:** valor calculado por un modelo, no observado.
- **Error típico:** cuánto se equivocó un modelo, en promedio, al estimar meses ya conocidos.
""")
    st.caption("Fuente: Base de Migración 2009–jun 2026 (INGUAT/IGM). Uso académico; no son cifras oficiales.")

f = agg[agg["Año"].between(*rango) & agg["Vía"].isin(vias) & agg["Tipo de Viajero"].isin(tipos)]

# ------------------------------------------------------------------
# Encabezado
# ------------------------------------------------------------------
st.title("Observatorio de Turismo Sostenible · Guatemala")
st.markdown(
    f"<p class='nota'>Llegadas internacionales al país, {rango[0]}–{rango[1]}. "
    f"<span class='etiqueta-obs'>Observado</span> = dato registrado · "
    f"<span class='etiqueta-est'>Estimado</span> = cálculo de un modelo.</p>", unsafe_allow_html=True)

if f.empty:
    sin_datos("Ningún dato cumple los filtros. Seleccione al menos una vía y un tipo de viajero.")
    st.stop()

anual = f.groupby("Año")["Viajero"].sum()
ult = anual.index.max()
completo = agg.loc[agg["Año"] == ult, "Mes"].max() == 12
k1, k2, k3, k4 = st.columns(4)
k1.metric("Llegadas en el período", f"{anual.sum()/1e6:,.1f} M")
if len(anual) > 1:
    prev = anual.iloc[-2]
    k2.metric(f"Llegadas {ult}" + ("" if completo else " (ene–jun)"), f"{anual.iloc[-1]/1e6:,.2f} M",
              None if not completo else f"{(anual.iloc[-1]/prev-1)*100:+.1f}% vs {ult-1}")
aer = f.loc[f["Vía"] == "Aérea", "Viajero"].sum() / f["Viajero"].sum() * 100
k3.metric("Entran por vía aérea", f"{aer:.0f}%")
mes_pico = f.groupby("Mes")["Viajero"].mean().idxmax()
k4.metric("Mes con más llegadas (promedio)", MESES[mes_pico - 1])

tab1, tab2, tab3, tab4 = st.tabs(["📈 Panorama", "🌎 Mercados y fronteras", "🔮 Predicciones", "ℹ️ Acerca del tablero"])

# ==================================================================
# TAB 1 · Panorama  (V1 → V2 enlazadas, V3)
# ==================================================================
with tab1:
    st.subheader("¿Cómo ha evolucionado la llegada de visitantes?")
    st.markdown("<p class='nota'>Haga clic en una barra para ver el detalle mensual de ese año "
                "en la gráfica de la derecha. Doble clic en el fondo para quitar la selección.</p>",
                unsafe_allow_html=True)
    c1, c2 = st.columns([1.05, 1])

    with c1:
        st.markdown("**V1 · Llegadas por año y vía de entrada**")
        da = f.groupby(["Año", "Vía"], as_index=False)["Viajero"].sum()
        fig = px.bar(da, x="Año", y="Viajero", color="Vía", color_discrete_map=COLOR_VIA,
                     category_orders={"Vía": list(COLOR_VIA)},
                     labels={"Viajero": "Llegadas", "Año": ""})
        fig.update_traces(hovertemplate="%{x} · %{fullData.name}<br><b>%{y:,.0f}</b> llegadas<extra></extra>",
                          marker_line_width=0)
        fig.update_layout(bargap=0.18, clickmode="event+select")
        ev1 = st.plotly_chart(estilo(fig), on_select="rerun", selection_mode="points", key="v1",
                              width="stretch")
        anios_sel = sorted({int(x) for x in seleccion(ev1)})
        if not completo and ult in da["Año"].values:
            st.caption(f"⚠️ {ult} solo tiene datos hasta junio.")

    with c2:
        anio = anios_sel[-1] if anios_sel else int(ult if completo else ult - 1)
        st.markdown(f"**V2 · Detalle mensual de {anio}** {'(seleccionado)' if anios_sel else '(por defecto: último año completo)'}")
        dm = f[f["Año"] == anio].groupby(["Mes", "Vía"], as_index=False)["Viajero"].sum()
        prom = f[f["Año"] != anio].groupby(["Año", "Mes"])["Viajero"].sum().groupby("Mes").mean()
        if dm.empty:
            sin_datos()
        else:
            dm["MesN"] = dm["Mes"].map(lambda m: MESES[m - 1])
            fig2 = px.line(dm, x="MesN", y="Viajero", color="Vía", color_discrete_map=COLOR_VIA,
                           markers=True, category_orders={"MesN": MESES, "Vía": list(COLOR_VIA)},
                           labels={"Viajero": "Llegadas", "MesN": ""})
            fig2.update_traces(line_width=2, marker_size=8,
                               hovertemplate="%{x} · %{fullData.name}<br><b>%{y:,.0f}</b><extra></extra>")
            if len(prom):
                tot_anio = dm.groupby("Mes")["Viajero"].sum()
                fig2.add_scatter(x=[MESES[m - 1] for m in tot_anio.index], y=tot_anio.values,
                                 name=f"Total {anio}", line=dict(color=OBSERVADO, width=2.5),
                                 hovertemplate="%{x} · total<br><b>%{y:,.0f}</b><extra></extra>")
                fig2.add_scatter(x=[MESES[m - 1] for m in prom.index], y=prom.values,
                                 name="Promedio de otros años", line=dict(color=NEUTRO, width=2, dash="dot"),
                                 hovertemplate="%{x} · promedio otros años<br>%{y:,.0f}<extra></extra>")
            st.plotly_chart(estilo(fig2), key="v2", width="stretch")

    st.markdown("**V3 · Calendario de llegadas: ¿en qué meses se concentra la demanda?**")
    st.markdown("<p class='nota'>Cada celda es un mes; más oscuro = más llegadas. "
                "Útil para planificar personal, promoción y capacidad en temporada alta.</p>",
                unsafe_allow_html=True)
    hm = f.groupby(["Año", "Mes"])["Viajero"].sum().unstack("Mes").reindex(columns=range(1, 13))
    fig3 = go.Figure(go.Heatmap(z=hm.values, x=MESES, y=hm.index.astype(str), colorscale=SECUENCIAL,
                                xgap=2, ygap=2, colorbar=dict(title="Llegadas", tickformat="~s"),
                                hovertemplate="%{x} %{y}<br><b>%{z:,.0f}</b> llegadas<extra></extra>"))
    fig3.update_yaxes(autorange="reversed", tickformat="d", gridcolor="white")
    st.plotly_chart(estilo(fig3, alto=max(300, 26 * len(hm)), leyenda=False), key="v3", width="stretch")
    st.caption("Celdas en blanco = meses sin datos (2026 llega a junio). La caída de 2020 coincide con "
               "el cierre de fronteras por la pandemia.")

# ==================================================================
# TAB 2 · Mercados y fronteras  (V4 → V5 enlazadas, V6)
# ==================================================================
with tab2:
    st.subheader("¿De dónde vienen y por dónde entran?")
    st.markdown("<p class='nota'>Haga clic en una región de origen para ver por qué fronteras "
                "entran sus visitantes.</p>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**V4 · Llegadas por región de residencia**")
        dr = (f[f["Región dos"] != "Cruceristas"].groupby("Región dos", as_index=False)["Viajero"].sum()
              .sort_values("Viajero"))
        fig4 = px.bar(dr, x="Viajero", y="Región dos", orientation="h",
                      labels={"Viajero": "Llegadas", "Región dos": ""})
        fig4.update_traces(marker_color=CATEGORICO[0], marker_line_width=0,
                           hovertemplate="%{y}<br><b>%{x:,.0f}</b> llegadas<extra></extra>")
        fig4.update_xaxes(tickformat="~s")
        ev4 = st.plotly_chart(estilo(fig4, leyenda=False), on_select="rerun", selection_mode="points",
                              key="v4", width="stretch")
        regiones_sel = seleccion(ev4, "y")
    with c2:
        reg = regiones_sel if regiones_sel else None
        st.markdown(f"**V5 · Principales fronteras de entrada — {', '.join(reg) if reg else 'todas las regiones'}**")
        dfr = f[f["Región dos"].isin(reg)] if reg else f
        top = (dfr.groupby(["Frontera", "Vía"], as_index=False)["Viajero"].sum()
               .sort_values("Viajero", ascending=False).head(10))
        if top.empty:
            sin_datos()
        else:
            top["Frontera"] = top["Frontera"].str.replace(r"^\d+\s", "", regex=True)
            fig5 = px.bar(top.sort_values("Viajero"), x="Viajero", y="Frontera", color="Vía",
                          orientation="h", color_discrete_map=COLOR_VIA,
                          labels={"Viajero": "Llegadas", "Frontera": ""})
            fig5.update_traces(marker_line_width=0,
                               hovertemplate="%{y} (%{fullData.name})<br><b>%{x:,.0f}</b><extra></extra>")
            fig5.update_xaxes(tickformat="~s")
            st.plotly_chart(estilo(fig5), key="v5", width="stretch")

    st.info("🚧 Próximamente en esta pestaña: V6 · composición por tipo de viajero en el tiempo.", icon="🛠️")

# ==================================================================
# TAB 3 · Predicciones  (V7 ↔ V8 enlazadas)
# ==================================================================
with tab3:
    st.subheader("¿Qué tan bien anticipan los modelos la llegada de visitantes?")
    st.markdown(
        "<p class='nota'>Los modelos se entrenaron con datos hasta <b>marzo 2021</b> y se compararon con lo "
        "que realmente ocurrió después (abr 2021–jun 2026). Son los mismos modelos del Laboratorio 1. "
        "Haga clic en un modelo de la gráfica de errores para resaltarlo.</p>", unsafe_allow_html=True)
    orden = ["Total", "Vía Aérea", "Vía Terrestre", "Vía Marítima", "Turista", "Excursionista"]
    serie = st.selectbox("Serie a analizar", [x for x in orden if x in set(met["serie"])], index=0,
                         on_change=lambda: st.session_state.pop("modelo_sel", None))
    m = met[met["serie"] == serie].copy()
    real_media = pred.loc[(pred["serie"] == serie), "real"].mean()
    m["Error típico (%)"] = m["MAE"] / real_media * 100
    m = m.sort_values("MAE")
    estables = m[m["estable"]]

    c1, c2 = st.columns([1, 1.6])
    with c1:
        st.markdown("**V8 · Error típico por modelo** (menor es mejor)")
        fig8 = px.bar(estables.sort_values("MAE", ascending=False), x="Error típico (%)", y="modelo",
                      orientation="h", labels={"modelo": ""}, custom_data=["MAE"])
        sel8 = st.session_state.get("modelo_sel")
        fig8.update_traces(
            marker_color=[ESTIMADO if mm == (sel8 or estables.iloc[0]["modelo"]) else NEUTRO
                          for mm in estables.sort_values("MAE", ascending=False)["modelo"]],
            hovertemplate="%{y}<br>Se equivoca en promedio <b>%{customdata[0]:,.0f}</b> llegadas/mes "
                          "(%{x:.0f}%)<extra></extra>", marker_line_width=0)
        fig8.update_xaxes(ticksuffix="%", tickformat=".0f")
        ev8 = st.plotly_chart(estilo(fig8, alto=330, leyenda=False), on_select="rerun",
                              selection_mode="points", key="v8", width="stretch")
        s8 = seleccion(ev8, "y")
        if s8 and s8[0] != sel8:
            st.session_state["modelo_sel"] = s8[0]
            st.rerun()
        if len(estables) < len(m):
            st.caption("⚠️ Se ocultó " + ", ".join(m.loc[~m["estable"], "modelo"]) +
                       ": su pronóstico se vuelve inestable a largo plazo (hallazgo del Lab 1).")

    with c2:
        modelo = st.session_state.get("modelo_sel") or estables.iloc[0]["modelo"]
        if modelo not in set(estables["modelo"]):
            modelo = estables.iloc[0]["modelo"]
        st.markdown(f"**V7 · Observado vs. estimado — {serie} · modelo: {modelo}**")
        p = pred[(pred["serie"] == serie) & (pred["modelo"] == modelo)]
        fig7 = go.Figure()
        if p["lo"].notna().any():
            fig7.add_scatter(x=pd.concat([p["fecha"], p["fecha"][::-1]]),
                             y=pd.concat([p["hi"], p["lo"][::-1]]), fill="toself",
                             fillcolor="rgba(196,87,26,0.12)", line=dict(width=0), hoverinfo="skip",
                             name="Rango probable (80%)")
        fig7.add_scatter(x=p["fecha"], y=p["real"], name="Observado", mode="lines",
                         line=dict(color=OBSERVADO, width=2.5),
                         hovertemplate="%{x|%b %Y}<br>Observado: <b>%{y:,.0f}</b><extra></extra>")
        fig7.add_scatter(x=p["fecha"], y=p["pred"], name="Estimado", mode="lines",
                         line=dict(color=ESTIMADO, width=2.5, dash="dash"),
                         hovertemplate="%{x|%b %Y}<br>Estimado: <b>%{y:,.0f}</b><extra></extra>")
        fig7.update_layout(hovermode="x unified")
        fig7.update_xaxes(rangeslider=dict(visible=True, thickness=0.06))
        st.plotly_chart(estilo(fig7, alto=380), key="v7", width="stretch")
        st.caption("Use la barra inferior para acercarse a un período. Interpretación: la brecha entre "
                   "ambas líneas es el error del modelo; no implica causas.")

    with st.expander("Tabla comparativa de modelos (complementaria)"):
        st.dataframe(m[["modelo", "MAE", "RMSE", "Error típico (%)"]].rename(
            columns={"MAE": "Error medio (llegadas)", "RMSE": "Error cuadrático (RMSE)"}),
            hide_index=True, width="stretch",
            column_config={"Error medio (llegadas)": st.column_config.NumberColumn(format="%,.0f"),
                           "Error cuadrático (RMSE)": st.column_config.NumberColumn(format="%,.0f"),
                           "Error típico (%)": st.column_config.NumberColumn(format="%.0f%%")})

# ==================================================================
# TAB 4 · Acerca
# ==================================================================
with tab4:
    st.subheader("Alcance y limitaciones")
    st.markdown("""
- **Qué muestra:** llegadas internacionales mensuales a Guatemala (ene 2009–jun 2026) por vía,
  frontera, región de residencia y tipo de viajero; y el desempeño de 5 modelos de pronóstico
  del Laboratorio 1.
- **Qué no muestra:** gasto, ocupación hotelera ni impacto ambiental; tampoco permite afirmar
  *por qué* cambian las llegadas, solo *cuándo* y *dónde*.
- **Cambio metodológico 2022–2023:** la categoría *Viajero* dejó de incluir tránsito fronterizo
  frecuente, por eso se excluye por defecto.
- **Pronósticos:** se evaluaron sobre un período que empieza justo después de la pandemia, lo que
  los hace parecer peores que en condiciones normales. Úselos como referencia, no como meta.
- **Datos:** uso académico; no son cifras oficiales del INGUAT ni del IGM.
""")
    st.caption("Equipo: Dulce Ambrosio · Daniel Chet · Javier Linares — CC3084 Data Science, UVG 2026.")
