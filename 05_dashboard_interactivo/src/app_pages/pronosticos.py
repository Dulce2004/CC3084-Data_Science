import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from comun import (ESCALA_ERROR, ESTIMADO, MODELOS, CONTEXTO, OBSERVADO, SELECCION, cargar, estilo, miles,
                   nota, seleccion, visibles)

SERIES = {"Total": None, "Vía Aérea": ("Vía", "Aérea"), "Vía Terrestre": ("Vía", "Terrestre"),
          "Vía Marítima": ("Vía", "Marítima"), "Turista": ("Tipo de Viajero", "Turista"),
          "Excursionista": ("Tipo de Viajero", "Excursionista"), "Viajero": ("Tipo de Viajero", "Viajero"),
          "Cruceristas": ("Tipo de Viajero", "Cruceristas")}
METRICAS = {
    "Error relativo": ("error_rel", "Error promedio como % del volumen real del mes", ".0%"),
    "RMSE": ("RMSE", "Error típico en viajeros; castiga más los errores grandes", ",.0f"),
    "MAE": ("MAE", "Error promedio en viajeros, en valor absoluto", ",.0f"),
}

st.title("Modelos de pronóstico")
st.caption("¿Cuántos viajeros se pueden esperar y qué tan confiable es cada modelo?")
st.info("Los modelos se entrenaron con los datos de enero 2009 a marzo 2021 y se evaluaron con abril 2021 a junio "
        "2026, meses que **no vieron**. Por eso aquí se puede comparar cada pronóstico con lo que pasó de verdad. "
        "Son los modelos de los Laboratorios 1 y 2; no se entrenó ninguno nuevo.", icon=":material/info:")

pred = cargar("predicciones")
met = cargar("metricas")
media_real = pred.groupby(["serie", "modelo"])["real"].mean().rename("media_real")
met = met.join(media_real, on=["serie", "modelo"])
met["error_rel"] = met["MAE"] / met["media_real"]
# Un modelo "diverge" cuando su pronóstico se dispara (SARIMA en Vía Aérea: ver README del Lab 1)
maximos = pred.groupby(["serie", "modelo"]).agg(pmax=("pred", "max"), rmax=("real", "max"))
divergentes = set(maximos.index[maximos["pmax"] > 10 * maximos["rmax"]])
met["diverge"] = [(s, m) in divergentes for s, m in zip(met["serie"], met["modelo"])]

serie = st.selectbox("Serie a pronosticar", list(SERIES), key="serie_pron",
                     help="Total de viajeros, o una vía de entrada o un tipo de viajero.")
met_s = met[(met["serie"] == serie) & ~met["diverge"]].sort_values("RMSE")
disponibles = met_s["modelo"].tolist()
elegidos = st.pills("Modelos a mostrar", disponibles, selection_mode="multi", default=disponibles[:3],
                    key=f"modelos_{serie}") or []
for s, m in sorted(divergentes):
    if s == serie:
        st.warning(f"**{m}** se omite en esta serie: su pronóstico se dispara a valores imposibles (diverge). "
                   "Un buen ajuste en entrenamiento no garantiza un buen pronóstico a largo plazo.",
                   icon=":material/trending_up:")
if "LSTM" not in disponibles:
    st.caption(":material/info: El LSTM (red neuronal del Laboratorio 2) solo se entrenó para *Total* y *Vía Aérea*.")

ver = visibles(["Pronóstico contra realidad", "Precisión por modelo", "Error mes a mes", "Todos los modelos y series"],
               key="ver_pronosticos")
p_s = pred[pred["serie"] == serie]

if "Pronóstico contra realidad" in ver:
    with st.container(border=True):
        st.subheader(f"Pronóstico contra realidad · {serie}")
        st.caption("Línea azul = lo que pasó. Líneas punteadas = lo que pronosticó cada modelo. La banda naranja es "
                   "el rango donde Prophet esperaba el valor con 80 % de probabilidad. Use los botones para acercar "
                   "o alejar el período.")
        mens = cargar("mercados")
        if SERIES[serie]:
            col, val = SERIES[serie]
            mens = mens[mens[col] == val]
        hist = mens.groupby("fecha")["Viajero"].sum()
        inicio = p_s["fecha"].min()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index[hist.index < inicio], y=hist[hist.index < inicio], name="Entrenamiento",
                                 line=dict(color=CONTEXTO, width=2), hovertemplate="%{y:,.0f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=hist.index[hist.index >= inicio], y=hist[hist.index >= inicio], name="Real",
                                 line=dict(color=OBSERVADO, width=3), hovertemplate="Real: %{y:,.0f}<extra></extra>"))
        if "Prophet" in elegidos:
            b = p_s[p_s["modelo"] == "Prophet"]
            fig.add_trace(go.Scatter(x=pd.concat([b["fecha"], b["fecha"][::-1]]), y=pd.concat([b["hi"], b["lo"][::-1]]),
                                     fill="toself", fillcolor="rgba(196,87,26,0.15)", line=dict(width=0),
                                     name="Rango Prophet 80 %", hoverinfo="skip"))
        for m in elegidos:
            d = p_s[p_s["modelo"] == m]
            fig.add_trace(go.Scatter(x=d["fecha"], y=d["pred"], name=m, line=dict(color=MODELOS[m], dash="dash", width=2.5),
                                     hovertemplate=f"{m}: %{{y:,.0f}}<extra></extra>"))
        fig.add_vline(x=inicio, line=dict(color=CONTEXTO, dash="dot"))
        fig.add_annotation(x=inicio, y=1, yref="paper", text="inicio de la prueba", showarrow=False, xanchor="left")
        fig.update_xaxes(range=[pd.Timestamp("2018-01-01"), p_s["fecha"].max()], rangeselector=dict(buttons=[
            dict(count=2, label="2 años", step="year", stepmode="backward"),
            dict(count=5, label="5 años", step="year", stepmode="backward"),
            dict(count=8, label="8 años", step="year", stepmode="backward"),
            dict(label="Todo", step="all")]))
        fig = estilo(fig, 480)
        fig.update_layout(hovermode="x unified", legend=dict(y=-0.12, yanchor="top"))
        st.plotly_chart(fig, alt=f"Valores reales y pronósticos de {serie}")
        if not elegidos:
            st.info("Elija al menos un modelo arriba para ver su pronóstico.", icon=":material/touch_app:")
        nota(f"En el período de prueba llegaron en promedio {miles(p_s['real'].mean())} viajeros por mes ({serie}).",
             "Los modelos aprendieron con datos que terminan en plena pandemia; por eso varios subestiman la "
             "recuperación. Un error bajo aquí no garantiza el mismo desempeño en el futuro.")

modelo = seleccion("v10", "y")
if modelo not in disponibles:
    modelo = disponibles[0] if disponibles else None

if "Precisión por modelo" in ver:
    with st.container(border=True):
        st.subheader(":material/link: Precisión por modelo")
        st.caption("Barra más corta = modelo más preciso. **Haga clic en una barra** para ver abajo en qué meses "
                   "se equivocó ese modelo.")
        cual = st.segmented_control("Medida de error", list(METRICAS), default="Error relativo", key="metrica") \
            or "Error relativo"
        colm, ayuda, fmt = METRICAS[cual]
        st.caption(f":material/help: **{cual}:** {ayuda}.")
        orden = met_s.sort_values(colm, ascending=False)
        fig = go.Figure(go.Bar(
            x=orden[colm], y=orden["modelo"], orientation="h",
            marker_color=[MODELOS[m] for m in orden["modelo"]],
            marker_line=dict(width=[3 if m == modelo else 0 for m in orden["modelo"]], color="#1B2430"),
            texttemplate="%{x:" + fmt + "}", textposition="outside",
            hovertemplate="%{y}: %{x:" + fmt + "}<extra></extra>"))
        fig = estilo(fig, 60 + 46 * len(orden), titulo_y=None)
        fig.update_xaxes(title=cual, tickformat=fmt)
        st.plotly_chart(fig, key="v10", on_select="rerun", selection_mode="points",
                        alt=f"Error de cada modelo para {serie}")
        if len(met_s):
            mejor, peor = met_s.iloc[0], met_s.iloc[-1]
            nota(f"Para {serie}, {mejor['modelo']} se equivoca en promedio ±{miles(mejor['MAE'])} viajeros por mes "
                 f"({mejor['error_rel']:.0%} del volumen real); {peor['modelo']}, ±{miles(peor['MAE'])}.",
                 f"Si hubiera que elegir uno para {serie}, los datos favorecen a {mejor['modelo']}; el orden puede "
                 "cambiar en otra serie (ver la tabla de abajo).")

if "Error mes a mes" in ver and modelo:
    with st.container(border=True):
        st.subheader(f":material/link: Error mes a mes · {modelo}")
        st.caption("Barra hacia arriba (naranja) = el modelo pronosticó más viajeros de los que llegaron. Hacia abajo "
                   "(azul oscuro) = pronosticó menos. Sirve para saber en qué épocas no conviene confiar en él.")
        d = p_s[p_s["modelo"] == modelo]
        err = d["pred"] - d["real"]
        fig = go.Figure(go.Bar(x=d["fecha"], y=err, marker_color=np.where(err >= 0, ESTIMADO, SELECCION),
                               customdata=np.stack([d["real"], d["pred"]], axis=1),
                               hovertemplate="%{x|%b %Y}<br>Real: %{customdata[0]:,.0f}<br>Pronóstico: "
                                             "%{customdata[1]:,.0f}<br>Diferencia: %{y:+,.0f}<extra></extra>"))
        st.plotly_chart(estilo(fig, 360, titulo_y="Pronóstico − real (viajeros)"),
                        alt=f"Error mensual del modelo {modelo} en {serie}")
        nota(f"{modelo} sobreestimó en {(err > 0).sum()} de {len(err)} meses y subestimó en {(err < 0).sum()}.")

if "Todos los modelos y series" in ver:
    with st.container(border=True):
        st.subheader("Todos los modelos en todas las series")
        st.caption("Error relativo de cada modelo (filas) en cada serie (columnas). Más oscuro = peor. Sirve para ver "
                   "si un mismo modelo funciona bien en todas partes. Celda vacía = el modelo no aplica a esa serie "
                   "o diverge.")
        tabla = (met[~met["diverge"]].pivot(index="modelo", columns="serie", values="error_rel")
                   .reindex(index=list(MODELOS), columns=list(SERIES)))
        fig = go.Figure(go.Heatmap(z=tabla.values, x=tabla.columns, y=tabla.index, colorscale=ESCALA_ERROR,
                                   zmin=0, zmax=min(1.0, float(np.nanmax(tabla.values))),
                                   text=[[("" if pd.isna(v) else f"{v:.0%}") for v in fila] for fila in tabla.values],
                                   texttemplate="%{text}", colorbar=dict(title="Error", tickformat=".0%"),
                                   hovertemplate="%{y} en %{x}: %{z:.0%}<extra></extra>", hoverongaps=False))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(estilo(fig, 420, titulo_y=None).update_yaxes(tickformat=None),
                        alt="Error relativo por modelo y serie")
        ganador = met[~met["diverge"]].loc[lambda x: x.groupby("serie")["RMSE"].idxmin()]
        conteo = ganador["modelo"].value_counts()
        nota(f"{conteo.index[0]} es el más preciso en {conteo.iloc[0]} de las 8 series.",
             "Ningún modelo gana en todas: conviene elegir el modelo según la serie que se quiera planificar.")

with st.expander("Tabla comparativa de modelos", icon=":material/table:"):
    st.dataframe(met_s[["modelo", "detalle", "MAE", "RMSE", "error_rel"]], hide_index=True,
                 column_config={"modelo": "Modelo", "detalle": "Configuración",
                                "MAE": st.column_config.NumberColumn("MAE (viajeros)", format="localized"),
                                "RMSE": st.column_config.NumberColumn("RMSE (viajeros)", format="localized"),
                                "error_rel": st.column_config.NumberColumn("Error relativo", format="percent")},
                 alt=f"Métricas de error de los modelos para {serie}")
