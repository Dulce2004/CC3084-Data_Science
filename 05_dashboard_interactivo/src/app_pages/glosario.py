import streamlit as st

st.title("Glosario y limitaciones")
st.caption("Términos usados en el tablero, explicados sin estadística.")

st.subheader("Términos")
st.table({
    "Término": ["Turista", "Excursionista", "Viajero (categoría)", "Crucerista", "Vía de entrada", "Mercado",
                "Pronóstico", "Período de prueba", "MAE", "RMSE", "Error relativo", "Rango de 80 %", "Divergir"],
    "Qué significa": [
        "Visitante que pasa al menos una noche en el país.",
        "Visitante que entra y sale el mismo día.",
        "Registro de entradas que no son turismo (tránsito, comercio fronterizo). Cambió de criterio en 2023.",
        "Visitante que llega en crucero y no pernocta en tierra.",
        "Aérea (aeropuertos), terrestre (fronteras) o marítima (puertos).",
        "País o agrupación de países de donde viene el viajero.",
        "Estimación de un modelo sobre cuántos viajeros llegarán en un mes.",
        "Abril 2021 a junio 2026: meses que los modelos no vieron y que sirven para medir si aciertan.",
        "Error absoluto medio: cuántos viajeros, en promedio, se equivoca el modelo cada mes.",
        "Raíz del error cuadrático medio: como el MAE, pero los errores grandes pesan más.",
        "MAE dividido entre el promedio real del mes. 20 % = se equivoca en una quinta parte del volumen.",
        "Prophet calcula un rango donde espera que caiga el valor real 8 de cada 10 veces.",
        "Cuando un modelo produce pronósticos cada vez más lejos de la realidad hasta valores imposibles.",
    ],
}, alt="Glosario de términos del tablero")

st.subheader("Modelos")
st.table({
    "Modelo": ["Seasonal naive", "SES", "Holt-Winters", "SARIMA", "Prophet", "LSTM"],
    "Cómo pronostica": [
        "Repite lo que pasó el mismo mes del año anterior. Sirve de punto de comparación mínimo.",
        "Suavizamiento exponencial simple: promedio que da más peso a los meses recientes; no ve temporadas.",
        "Suavizamiento con tendencia y temporada: sigue la dirección y repite el patrón anual.",
        "Modelo estadístico clásico de series de tiempo con componente de temporada (versión manual y por búsqueda).",
        "Modelo de Meta que separa tendencia y temporada anual; entrega un rango de incertidumbre.",
        "Red neuronal que aprende patrones de los últimos 12 a 18 meses (Laboratorio 2).",
    ],
}, alt="Descripción de los modelos de pronóstico")

st.subheader("Limitaciones")
st.markdown("""
- **Los datos muestran asociaciones, no causas.** Cuando el tablero dice que una caída *coincide* con la pandemia,
  no está midiendo el efecto de la pandemia.
- **Cambios de registro.** La categoría *Viajero* cambió de criterio en 2023 y el campo *País* pasó a agrupar
  mercados desde 2023. Las comparaciones que crucen ese año deben leerse con cuidado.
- **Tres fuentes distintas** (respaldos históricos 2009–2020, IGM 2021–2022, INGUAT 2023–2026).
- **Los modelos se entrenaron hasta marzo 2021**, en plena pandemia, y se evaluaron en la recuperación. Su
  desempeño aquí no garantiza el desempeño en otros períodos.
- **Viajeros fraccionarios:** algunos conteos de la fuente son estimaciones y tienen decimales.
- Datos de uso académico; no son cifras oficiales del INGUAT ni del IGM.
""")

st.subheader("Fuentes")
st.markdown("Base de Migración 2009–jun 2026 (INGUAT/IGM). Modelos: Laboratorio 1 (series de tiempo) y "
            "Laboratorio 2 (LSTM) del curso CC3084 Data Science, UVG.")
