"""Entry point aplikasi multipage.  Jalankan:  streamlit run app/streamlit_app.py"""
import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from common import DISCLAIMER, inject_css  # noqa: E402

st.set_page_config(layout="wide", page_title="Zona Gempa Indonesia", page_icon="🌋")
inject_css()

pages = {
    "Peta": [
        st.Page("views/cek_lokasi.py", title="Cek Lokasi", icon="📍", default=True),
        st.Page("views/gempa_terkini.py", title="Gempa Terkini", icon="🔴"),
        st.Page("views/zona.py", title="Zona Seismik", icon="🧭"),
        st.Page("views/peta_gempa.py", title="Peta Gempa", icon="🗺️"),
    ],
    "Analisis": [
        st.Page("views/kategorikan.py", title="Kategorikan Gempa", icon="⚡"),
        st.Page("views/susulan.py", title="Gempa Susulan", icon="🔁"),
        st.Page("views/swarm.py", title="Klaster & Swarm", icon="🐝"),
        st.Page("views/eksplorasi.py", title="Eksplorasi Data", icon="📊"),
    ],
    "Informasi": [
        st.Page("views/metodologi.py", title="Metodologi", icon="📘"),
        st.Page("views/keselamatan.py", title="Panduan Keselamatan", icon="🛡️"),
    ],
}
page = st.navigation(pages)
with st.sidebar:
    st.caption(DISCLAIMER)
page.run()
