import streamlit as st

from comun import MESES, cargar, miles

st.title("Observatorio de turismo · Guatemala")
st.markdown("Ingreso mensual de viajeros internacionales al país, de enero 2009 a junio 2026. Este tablero "
            "responde preguntas de gestión: **cuántos llegan, cuándo, de dónde, por dónde entran y "
            "cuántos se esperan**, y qué tan confiables son esos pronósticos.")

df = cargar("mercados")
te = df[df["Tipo de Viajero"].isin(["Turista", "Excursionista"])]
anual = te.groupby(te["fecha"].dt.year)["Viajero"].sum()
pico = anual.loc[2009:2025].idxmax()
mensual = te.groupby(te["fecha"].dt.month)["Viajero"].mean()
met = cargar("metricas")
mejor = met[met["serie"] == "Total"].sort_values("RMSE").iloc[0]

with st.container(horizontal=True):
    st.metric("Visitantes 2025 (turistas + excursionistas)", miles(anual[2025]),
              f"{anual[2025] / anual[2019] - 1:+.0%} frente a 2019", border=True)
    st.metric("Año récord", str(pico), f"{miles(anual[pico])} visitantes", delta_color="off", delta_arrow="off", border=True)
    st.metric("Mes con más llegadas (promedio)", MESES[mensual.idxmax() - 1], "temporada alta",
              delta_color="off", delta_arrow="off", border=True)
    st.metric("Modelo más preciso (total mensual)", mejor["modelo"], f"error típico ±{miles(mejor['RMSE'])} "
              "viajeros/mes", delta_color="off", delta_arrow="off", border=True)
st.caption("Las tarjetas resumen el período completo y no cambian con los filtros. 2019 es el último año "
           "antes de la pandemia.")

st.subheader("Cómo usar el tablero")
c1, c2 = st.columns(2)
with c1, st.container(border=True):
    st.markdown("**:material/touch_app: Haga clic en las gráficas.** Las gráficas marcadas con "
                ":material/link: están enlazadas: al elegir una barra, la gráfica de abajo muestra el detalle "
                "de esa selección. Haga doble clic en una gráfica para quitar la selección.")
    st.markdown("**:material/zoom_in: Acerque o aleje.** Arrastre sobre el eje de tiempo para hacer zoom, y use "
                "los controles *Nivel de detalle* para pasar de años a meses.")
with c2, st.container(border=True):
    st.markdown("**:material/filter_alt: Filtre.** La barra lateral define el período y los tipos de viajero. "
                "En cada página puede ocultar las gráficas que no necesite.")
    st.markdown("**:material/palette: Lea los colores.** Azul = dato observado. Naranja y línea punteada = "
                "estimación de un modelo. El azul oscuro marca lo que usted seleccionó.")

st.subheader("Antes de interpretar")
st.warning("""
- **Categoría *Viajero*:** desde 2023 dejó de registrar a viajeros no turísticos de alta frecuencia (tránsito y \
comercio fronterizo); su caída es un cambio de registro, no una caída real. Por eso el tablero usa por defecto \
**Turista + Excursionista** para comparar años.
- **Mercados (País):** hasta 2022 es el país de procedencia; desde 2023 algunas filas son agrupaciones de mercado.
- **Fuentes por tramo:** 2009–2020 respaldos históricos, 2021–2022 IGM, 2023–jun 2026 sistema INGUAT.
- **2026 solo tiene enero a junio.**
- Los datos son de uso académico y no son cifras oficiales del INGUAT ni del IGM.
""", title="Advertencias sobre los datos", icon=":material/warning:")
