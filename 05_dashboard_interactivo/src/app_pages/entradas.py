import plotly.graph_objects as go
import streamlit as st

from comun import VIAS, cargar, estilo, filtrar, miles, nota, seleccion, sin_datos, visibles

st.title("Vías y puestos de entrada")
st.caption("¿Por dónde entran los visitantes, cómo ha cambiado y qué puestos fronterizos atienden más?")

df = filtrar(cargar("mercados"))
if sin_datos(df, "viajeros"):
    st.stop()
ver = visibles(["Mezcla por vía", "Puestos de entrada"], key="ver_entradas")

por_via = df.groupby([df["fecha"].dt.year, "Vía"])["Viajero"].sum().unstack(fill_value=0)
anios = por_via.index.astype(str)
defecto = "2025" if "2025" in anios else anios[-1]
anio = str(seleccion("v7") or defecto)
via = seleccion("v7", "customdata")
via = (via[0] if isinstance(via, list) else via) or "Aérea"
if anio not in anios or via not in por_via:
    anio, via = defecto, "Aérea"

if "Mezcla por vía" in ver:
    with st.container(border=True):
        st.subheader(":material/link: Mezcla por vía de entrada")
        st.caption("Cada barra es un año dividido por vía. **Haga clic en un segmento** para ver los puestos de esa "
                   "vía en ese año. Sirve para dimensionar aeropuertos, fronteras terrestres y puertos.")
        modo = st.segmented_control("Mostrar", ["Porcentaje", "Viajeros"], default="Porcentaje",
                                    key="modo_vias") or "Porcentaje"
        fig = go.Figure()
        for v, color in VIAS.items():
            if v in por_via:
                fig.add_trace(go.Bar(x=anios, y=por_via[v], name=v, marker_color=color, customdata=[v] * len(anios),
                                     marker_line=dict(width=[3 if (a == anio and v == via) else 0 for a in anios],
                                                      color="#1B2430"),
                                     hovertemplate=f"{v}, %{{x}}: %{{y:,.0f}}<extra></extra>"))
        fig.update_layout(barmode="stack", barnorm="percent" if modo == "Porcentaje" else None)
        fig = estilo(fig, 400, titulo_y="% de viajeros" if modo == "Porcentaje" else "Viajeros")
        if modo == "Porcentaje":
            fig.update_yaxes(tickformat=".0f", ticksuffix=" %")
            fig.update_traces(hovertemplate="%{customdata}, %{x}: %{y:.1f} %<extra></extra>")
        st.plotly_chart(fig, key="v7", on_select="rerun", selection_mode="points",
                        alt="Participación de cada vía de entrada por año")
        p = por_via.div(por_via.sum(axis=1), axis=0)
        nota(f"En {anios[-1]}, la vía terrestre concentra el {p.iloc[-1].get('Terrestre', 0):.0%} de las entradas y la "
             f"aérea el {p.iloc[-1].get('Aérea', 0):.0%}.")
        st.caption(":material/info: Quien llega en crucero se registra como *Crucerista*: agregue ese tipo en la "
                   "barra lateral para ver la vía marítima.")

if "Puestos de entrada" in ver:
    with st.container(border=True):
        st.subheader(f":material/link: Puestos de entrada · vía {via.lower()}, {anio}")
        st.caption("Viajeros por puesto de entrada para la vía y el año elegidos arriba. Sirve para saber qué "
                   "puestos necesitan más personal e infraestructura.")
        fr = filtrar(cargar("fronteras"))
        fr = fr[(fr["fecha"].dt.year == int(anio)) & (fr["Vía"] == via)]
        if not sin_datos(fr, f"puestos de la vía {via.lower()} en {anio}"):
            tabla = fr.groupby("Frontera")["Viajero"].sum().sort_values()
            tabla = tabla[tabla > 0]
            fig = go.Figure(go.Bar(x=tabla.values, y=tabla.index, orientation="h", marker_color=VIAS[via],
                                   hovertemplate="%{y}: %{x:,.0f} viajeros<extra></extra>"))
            fig = estilo(fig, max(260, 34 * len(tabla)), titulo_y=None)
            fig.update_xaxes(title="Viajeros", tickformat=",.0f")
            st.plotly_chart(fig, alt=f"Viajeros por puesto de entrada, vía {via}, {anio}")
            nota(f"{tabla.index[-1]} recibió {miles(tabla.iloc[-1])} viajeros, el {tabla.iloc[-1] / tabla.sum():.0%} "
                 f"de la vía {via.lower()} en {anio}.")
