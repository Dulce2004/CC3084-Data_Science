import plotly.graph_objects as go
import streamlit as st

from comun import (CONTEXTO, ESCALA_VOLUMEN, MESES, OBSERVADO, SELECCION, VIAS, cargar, estilo, filtrar, miles,
                   nota, seleccion, sin_datos, visibles)

st.title("Panorama en el tiempo")
st.caption("¿Cuántos visitantes llegan cada año, qué meses pesan más y cuál es la temporada alta?")

df = filtrar(cargar("mercados"))
if sin_datos(df, "viajeros"):
    st.stop()
ver = visibles(["Llegadas por año", "Detalle mensual", "Temporada alta"], key="ver_panorama")

anual = df.groupby(df["fecha"].dt.year)["Viajero"].sum()
anio = int(seleccion("v1") or 0) or (2025 if 2025 in anual.index else int(anual.index.max()))
if anio not in anual.index:
    anio = int(anual.index.max())

if "Llegadas por año" in ver:
    with st.container(border=True):
        st.subheader(":material/link: Llegadas por año")
        st.caption("Total de viajeros que ingresaron cada año. **Haga clic en una barra** para ver sus meses abajo. "
                   "Sirve para ubicar la recuperación frente a 2019 y fijar metas anuales.")
        fig = go.Figure(go.Bar(
            x=anual.index.astype(str), y=anual.values,
            marker_color=[SELECCION if a == anio else OBSERVADO for a in anual.index],
            hovertemplate="%{x}: %{y:,.0f} viajeros<extra></extra>"))
        if 2026 in anual.index:
            fig.add_annotation(x="2026", y=anual[2026], text="ene–jun", showarrow=False, yshift=12, font=dict(size=12))
        fig.update_xaxes(title=None)
        st.plotly_chart(estilo(fig, 380), key="v1", on_select="rerun", selection_mode="points",
                        alt="Viajeros por año; caída en 2020 y recuperación posterior")
        if 2019 in anual.index and 2025 in anual.index:
            nota(f"En 2025 llegaron {miles(anual[2025])} viajeros, {anual[2025] / anual[2019] - 1:+.0%} frente a "
                 f"2019 ({miles(anual[2019])}).",
                 "La caída de 2020 coincide con el cierre de fronteras por la pandemia; los datos muestran la "
                 "coincidencia en el tiempo, no miden su efecto.")

if "Detalle mensual" in ver:
    with st.container(border=True):
        st.subheader(f":material/link: Detalle mensual de {anio}, por vía de entrada")
        st.caption("Muestra los meses del año elegido arriba, separados por vía. Sirve para ver qué meses y qué "
                   "vías explican el total del año.")
        comparar = st.toggle("Comparar con el año anterior", value=False)
        del_anio = df[df["fecha"].dt.year == anio]
        mens = del_anio.groupby([del_anio["fecha"].dt.month, "Vía"])["Viajero"].sum().unstack(fill_value=0)
        fig = go.Figure()
        for via, color in VIAS.items():
            if via in mens:
                fig.add_trace(go.Scatter(x=[MESES[m - 1] for m in mens.index], y=mens[via], name=via,
                                         mode="lines+markers", line=dict(color=color, width=3),
                                         hovertemplate=f"{via}, %{{x}}: %{{y:,.0f}}<extra></extra>"))
        previo = df[df["fecha"].dt.year == anio - 1]
        if comparar and previo.empty:
            st.info(f"No hay datos de {anio - 1} en el período elegido.", icon=":material/info:")
        elif comparar:
            p = previo.groupby(previo["fecha"].dt.month)["Viajero"].sum()
            fig.add_trace(go.Scatter(x=[MESES[m - 1] for m in p.index], y=p.values, name=f"Total {anio - 1}",
                                     line=dict(color=CONTEXTO, dash="dot", width=2),
                                     hovertemplate=f"Total {anio - 1}, %{{x}}: %{{y:,.0f}}<extra></extra>"))
            fig.add_trace(go.Scatter(x=[MESES[m - 1] for m in mens.index], y=mens.sum(axis=1), name=f"Total {anio}",
                                     line=dict(color=SELECCION, width=2),
                                     hovertemplate=f"Total {anio}, %{{x}}: %{{y:,.0f}}<extra></extra>"))
        st.plotly_chart(estilo(fig, 380), alt=f"Viajeros por mes y vía en {anio}")
        top = mens.sum(axis=1)
        nota(f"En {anio} el mes con más llegadas fue {MESES[top.idxmax() - 1]} ({miles(top.max())}) y la vía "
             f"principal fue {mens.sum().idxmax()} ({mens.sum().max() / mens.values.sum():.0%} del año).")

if "Temporada alta" in ver:
    with st.container(border=True):
        st.subheader("Temporada alta: mes contra año")
        st.caption("Cada celda es un mes; más oscuro = más viajeros. Las columnas oscuras que se repiten año tras "
                   "año son la temporada alta. Sirve para planificar personal, promoción y capacidad.")
        tabla = (df.groupby([df["fecha"].dt.year, df["fecha"].dt.month])["Viajero"].sum()
                   .unstack().reindex(columns=range(1, 13)))
        fig = go.Figure(go.Heatmap(
            z=tabla.values, x=MESES, y=tabla.index, colorscale=ESCALA_VOLUMEN, colorbar=dict(title="Viajeros"),
            hovertemplate="%{x} %{y}: %{z:,.0f} viajeros<extra></extra>", hoverongaps=False))
        fig.update_yaxes(dtick=1, autorange="reversed")
        st.plotly_chart(estilo(fig, 480, titulo_y="Año").update_yaxes(tickformat="d"), alt="Mapa de calor de viajeros por mes y año")
        prom = tabla.mean()
        nota(f"En promedio, {MESES[prom.idxmax() - 1]} es el mes con más llegadas y {MESES[prom.idxmin() - 1]} "
             "el de menos. Las celdas vacías son meses sin datos (2026 termina en junio).")
