import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from comun import (ESCALA_VOLUMEN, OBSERVADO, SELECCION, VIAS, cargar, estilo, filtrar, miles, nota,
                   seleccion, sin_datos, visibles)

st.title("Mercados de origen")
st.caption("¿De dónde vienen los visitantes, cuáles son los mercados prioritarios y cómo evolucionan?")
st.info("Hasta 2022, *País* es el país de procedencia; desde 2023 algunas filas son agrupaciones de mercado. "
        "Compare mercados dentro de un mismo tramo de años.", icon=":material/info:")

df = filtrar(cargar("mercados"))
if sin_datos(df, "mercados"):
    st.stop()
ver = visibles(["Mapa de regiones", "Mercados principales", "Evolución del mercado"], key="ver_mercados")

if "Mapa de regiones" in ver:
    with st.container(border=True):
        st.subheader("Mapa de regiones y mercados")
        st.caption("El tamaño de cada bloque es el número de viajeros. **Haga clic en una región** para entrar a "
                   "sus mercados y en el nombre de arriba para volver. Sirve para ver de un vistazo qué regiones "
                   "concentran la demanda.")
        reg = df.groupby(["Región dos", "País"], as_index=False)["Viajero"].sum()
        reg = reg[reg["Viajero"] > 0]
        fig = px.treemap(reg, path=[px.Constant("Todas las regiones"), "Región dos", "País"], values="Viajero",
                         color="Viajero", color_continuous_scale=ESCALA_VOLUMEN, maxdepth=2)
        fig.update_traces(hovertemplate="%{label}: %{value:,.0f} viajeros (%{percentRoot:.1%} del total)"
                                        "<extra></extra>", texttemplate="%{label}<br>%{percentRoot:.1%}")
        fig.update_layout(coloraxis_colorbar=dict(title="Viajeros"))
        st.plotly_chart(estilo(fig, 460), alt="Viajeros por región y mercado de origen")
        top = reg.groupby("Región dos")["Viajero"].sum().sort_values(ascending=False)
        nota(f"{top.index[0]} aporta el {top.iloc[0] / top.sum():.0%} de los viajeros del período elegido.")

if "Mercados principales" in ver or "Evolución del mercado" in ver:
    paises = df.groupby("País")["Viajero"].sum().sort_values(ascending=False)
    paises = paises[paises > 0]
    mercado = seleccion("v5", "y") or paises.index[0]
    if mercado not in paises.index:
        mercado = paises.index[0]

if "Mercados principales" in ver:
    with st.container(border=True):
        st.subheader(":material/link: Mercados principales")
        st.caption("Ranking de mercados por número de viajeros. **Haga clic en una barra** para ver su evolución "
                   "abajo. Sirve para decidir dónde enfocar la promoción.")
        n = st.slider("Cuántos mercados mostrar", 5, 25, 10)
        top = paises.head(n).iloc[::-1]
        fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h",
                               marker_color=[SELECCION if p == mercado else OBSERVADO for p in top.index],
                               hovertemplate="%{y}: %{x:,.0f} viajeros<extra></extra>"))
        fig = estilo(fig, max(320, 28 * n), titulo_y=None)
        fig.update_xaxes(title="Viajeros", tickformat=",.0f")
        st.plotly_chart(fig, key="v5", on_select="rerun", selection_mode="points",
                        alt=f"Los {n} mercados con más viajeros")
        nota(f"Los {n} primeros mercados suman el {paises.head(n).sum() / paises.sum():.0%} de los viajeros.",
             "El Salvador y Guatemala encabezan por el tránsito terrestre fronterizo; para turismo de larga "
             "distancia conviene mirar la vía aérea en la página de vías.")

if "Evolución del mercado" in ver:
    with st.container(border=True):
        st.subheader(f":material/link: Evolución de {mercado}, por vía de entrada")
        st.caption("Viajeros del mercado elegido a lo largo del tiempo, apilados por vía. Sirve para saber si un "
                   "mercado crece y cómo llega.")
        detalle = st.segmented_control("Nivel de detalle", ["Año", "Trimestre", "Mes"], default="Año",
                                       key="detalle_mercado") or "Año"
        d = df[df["País"] == mercado]
        if sin_datos(d, mercado):
            st.stop()
        periodo = {"Año": "Y", "Trimestre": "Q", "Mes": "M"}[detalle]
        serie = (d.groupby([d["fecha"].dt.to_period(periodo).dt.start_time, "Vía"])["Viajero"].sum()
                   .unstack(fill_value=0))
        fig = go.Figure()
        for via, color in VIAS.items():
            if via in serie and serie[via].sum() > 0:
                fig.add_trace(go.Scatter(x=serie.index, y=serie[via], name=via, stackgroup="v",
                                         line=dict(color=color, width=1), fillcolor=color,
                                         hovertemplate=f"{via}: %{{y:,.0f}}<extra></extra>"))
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(estilo(fig, 380), alt=f"Viajeros de {mercado} en el tiempo por vía")
        anual = d.groupby(d["fecha"].dt.year)["Viajero"].sum()
        texto = f"{mercado} sumó {miles(anual.sum())} viajeros en el período elegido."
        if {2019, 2025} <= set(anual.index) and anual[2019] > 0:
            texto += f" En 2025 llegaron {anual[2025] / anual[2019] - 1:+.0%} frente a 2019."
        nota(texto)
