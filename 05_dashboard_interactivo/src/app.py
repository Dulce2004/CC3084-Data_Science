"""Observatorio de turismo: tablero interactivo (Laboratorio 10, CC3084).

Ejecutar desde 05_dashboard_interactivo/:  streamlit run src/app.py
"""
import streamlit as st

from comun import TIPOS

st.set_page_config(page_title="Observatorio de turismo · Guatemala", page_icon=":material/travel_explore:",
                   layout="wide")

paginas = st.navigation({
    "": [st.Page("app_pages/inicio.py", title="Inicio", icon=":material/home:", default=True)],
    "Explorar los datos": [
        st.Page("app_pages/panorama.py", title="Panorama en el tiempo", icon=":material/timeline:"),
        st.Page("app_pages/mercados.py", title="Mercados de origen", icon=":material/public:"),
        st.Page("app_pages/entradas.py", title="Vías y puestos de entrada", icon=":material/flight_land:"),
    ],
    "Pronósticos": [
        st.Page("app_pages/pronosticos.py", title="Modelos de pronóstico", icon=":material/insights:"),
    ],
    "Ayuda": [st.Page("app_pages/glosario.py", title="Glosario y limitaciones", icon=":material/menu_book:")],
})

with st.sidebar:
    st.subheader("Filtros generales")
    st.caption("Aplican a las páginas de exploración. Los pronósticos usan siempre el período de prueba "
               "(abril 2021 a junio 2026).")
    st.slider("Período (años)", 2009, 2026, (2009, 2026), key="periodo")
    st.multiselect("Tipo de viajero", TIPOS, default=["Turista", "Excursionista"], key="tipos",
                   help="Turista: pernocta al menos una noche. Excursionista: entra y sale el mismo día. "
                        "Viajero: categoría que dejó de registrar el tránsito fronterizo en 2023. "
                        "Cruceristas: llegan en crucero.")
    st.caption("Datos de uso académico (Base de Migración 2009–jun 2026). No son cifras oficiales del INGUAT.")

paginas.run()
