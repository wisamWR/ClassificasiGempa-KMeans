"""Peta seluruh gempa dengan filter waktu, magnitudo, kedalaman, intensitas, dan profil klaster."""
import folium
import pandas as pd
import streamlit as st
from folium.plugins import FastMarkerCluster
from streamlit_folium import st_folium

from common import (LAYER_CODES, LAYER_LABELS, MAP_DEFAULT_CENTER, MMI_CLASSES, MMI_COLORS, PROFILE_ORDER,
                    SIGNIFICANT_MW, fmt_local, load_events, mmi_legend)

MAX_MAP_POINTS = 2000
# Key widget yang dikembalikan ke default saat Reset Filter
FILTER_KEYS = ["peta_start_date", "peta_end_date", "peta_mw", "peta_layer", "peta_mmi", "peta_profile",
               "peta_mainshock", "peta_hide_suspect"]

# Marker lingkaran berwarna untuk FastMarkerCluster: row = [lat, lon, warna, popup]
MARKER_CALLBACK = """
function (row) {
    var marker = L.circleMarker(new L.LatLng(row[0], row[1]), {
        radius: 7, color: row[2], fillColor: row[2], fillOpacity: 0.85, weight: 1
    });
    marker.bindPopup(row[3]);
    return marker;
}
"""


def balanced_sample(data, n, by="mmi_class", seed=42):
    """Water filling: kuota dibagi rata per kelas, sisa kuota kelas kecil dialihkan ke kelas lain."""
    if len(data) <= n:
        return data
    counts = data[by].value_counts().sort_values()
    counts = counts[counts > 0]
    quotas, remaining = {}, n
    for i, (lbl, cnt) in enumerate(counts.items()):
        take = min(cnt, remaining // (len(counts) - i))
        quotas[lbl] = take
        remaining -= take
    return pd.concat([data[data[by] == lbl].sample(q, random_state=seed) for lbl, q in quotas.items()])


df = load_events()

st.title("🗺️ Peta Gempa")
st.caption("Katalog gempa BMKG 2008–2025. Warna = estimasi intensitas di episentrum (skala MMI) per gempa. "
           "Waktu ditampilkan dalam zona waktu setempat (perkiraan dari koordinat).")

# --- Filter ---
with st.sidebar:
    st.subheader("🛠️ Filter")
    if st.button("🔄 Reset Filter"):
        for key in FILTER_KEYS:
            st.session_state.pop(key, None)
        st.rerun()

    # min/max eksplisit: default Streamlit membatasi pilihan hanya ±10 tahun dari nilai awal
    data_start, data_end = df["time"].min().date(), df["time"].max().date()
    start = st.date_input("Tanggal Awal", data_start, min_value=data_start, max_value=data_end, key="peta_start_date")
    end = st.date_input("Tanggal Akhir", data_end, min_value=data_start, max_value=data_end, key="peta_end_date")
    mw_range = st.slider("Rentang Mw", 0.0, 9.0, (0.0, 9.0), 0.1, key="peta_mw")
    layers = st.multiselect("Lapisan kedalaman", LAYER_CODES, default=LAYER_CODES, format_func=LAYER_LABELS.get,
                            key="peta_layer")
    selected_mmi = st.multiselect("Kelas Intensitas (MMI)", MMI_CLASSES, default=MMI_CLASSES, key="peta_mmi")
    profiles = st.multiselect("Profil Klaster (K-Means)", PROFILE_ORDER, default=PROFILE_ORDER, key="peta_profile",
                              help="Label relatif antar-klaster dari magnitudo & kedalaman, bukan risiko absolut.")
    mainshock_only = st.checkbox("Hanya gempa utama (tanpa gempa susulan)", value=False, key="peta_mainshock",
                                 help="Hasil declustering Gardner–Knopoff (1974).")
    hide_suspect = st.checkbox("Sembunyikan event semu", value=True, key="peta_hide_suspect",
                               help="Mw ≥ 5 yang tidak tercatat di katalog USGS (±60 s, ≤150 km), "
                                    "atau dengan < 10 fase dan celah azimut > 180°.")

if start > end:
    st.sidebar.warning("Tanggal Awal setelah Tanggal Akhir, rentang ditukar otomatis.")
    start, end = end, start

dates = df["time"].dt.date
mask = ((dates >= start) & (dates <= end) & df["mw"].between(*mw_range) & df["layer"].isin(layers)
        & df["mmi_class"].isin(selected_mmi) & df["label"].isin(profiles))
if mainshock_only:
    mask &= df["is_mainshock"]
if hide_suspect:
    mask &= ~df["suspect_event"]
filtered = df[mask]

# --- Ringkasan ---
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Kejadian", f"{len(filtered):,}")
m2.metric(f"Gempa Mw ≥ {SIGNIFICANT_MW:.0f}", f"{(filtered['mw'] >= SIGNIFICANT_MW).sum():,}")
m3.metric("Mw Tertinggi", f"{filtered['mw'].max():.1f}" if not filtered.empty else "–")
m4.metric("Rata-rata Mw", f"{filtered['mw'].mean():.2f}" if not filtered.empty else "–")

# --- Peta ---
center = [filtered["latitude"].mean(), filtered["longitude"].mean()] if not filtered.empty else MAP_DEFAULT_CENTER
m = folium.Map(location=center, zoom_start=5, prefer_canvas=True)
# Sampel seimbang per kelas intensitas agar kelas minoritas tetap terlihat dan peta tetap ringan
plot_df = balanced_sample(filtered, MAX_MAP_POINTS)
points = [
    [row.latitude, row.longitude, MMI_COLORS.get(row.mmi_class, "gray"),
     f"<b>Mw {row.mw:.1f}</b> ({row.mag_type} {row.mag:.1f})<br>Kedalaman: {row.depth:.0f} km<br>"
     f"Intensitas: MMI {row.mmi_class} (I₀ ≈ {row.intensity:.1f})<br>"
     f"Zona: {row.zone_id}<br>Profil klaster: {row.label}<br>"
     f"{'Gempa utama' if row.is_mainshock else 'Gempa susulan / bagian sekuens'}<br>"
     f"{fmt_local(row.local_time, row.tz)}"]
    for row in plot_df.itertuples()
]
FastMarkerCluster(points, callback=MARKER_CALLBACK).add_to(m)
st_folium(m, height=520, use_container_width=True, returned_objects=[], key="peta_map")
if len(plot_df) < len(filtered):
    st.caption(f"Peta menampilkan {len(plot_df):,} dari {len(filtered):,} gempa (sampel seimbang per kelas intensitas).")

st.subheader("📋 Kelas Intensitas (MMI) & Rekomendasi")
mmi_legend()
st.page_link("views/cek_lokasi.py", label="Ingin tahu kondisi di sekitar kota Anda? Buka Cek Lokasi", icon="📍")
