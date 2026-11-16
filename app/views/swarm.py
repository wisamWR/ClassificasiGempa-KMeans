"""Klaster & Swarm: klaster spatio-temporal ST-DBSCAN dan jenisnya (swarm/doublet, gempa utama-susulan)."""
import altair as alt
import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from common import MAP_DEFAULT_CENTER, MMI_CLASSES, MMI_COLORS, fmt_local, load_events
from gempa import paths
from gempa.swarm import EPS_DAYS, EPS_KM, KINDS, MIN_SAMPLES, SWARM_MAX_GAP
from gempa.timezones import to_local_time

KIND_COLORS = {"swarm": "#9467bd", "mainshock": "#d62728", "secondary": "#7f7f7f"}


@st.cache_data
def load_clusters():
    clusters = pd.read_parquet(paths.ST_CLUSTERS)
    clusters["local_start"], clusters["tz"] = to_local_time(clusters["start"], clusters["latitude"],
                                                            clusters["longitude"])
    clusters["jenis"] = clusters["kind"].map(KINDS)
    return clusters, pd.read_parquet(paths.ST_CLUSTER_EVENTS)


clusters, cluster_events = load_clusters()
df = load_events()

st.title("🐝 Klaster & Swarm")
st.caption(f"Klaster spatio-temporal ST-DBSCAN (Birant & Kut 2007): dua gempa bertetangga bila jarak 3D ≤ "
           f"{EPS_KM:.0f} km dan selisih waktu ≤ {EPS_DAYS:.0f} hari; klaster minimal {MIN_SAMPLES} gempa. "
           "Data: gempa ≥ Mc tahunan "
           "yang lolos verifikasi. Swarm/doublet = tidak ada gempa yang jauh lebih besar dari lainnya "
           f"(selisih Mw terbesar dan kedua < {SWARM_MAX_GAP}).")

cols = st.columns(4)
cols[0].metric("Klaster", f"{len(clusters):,}")
for col, kind in zip(cols[1:], KINDS):
    col.metric(KINDS[kind], f"{(clusters['kind'] == kind).sum():,}")

kinds = st.multiselect("Jenis klaster", list(KINDS), default=list(KINDS), format_func=KINDS.get, key="swarm_kinds")
shown = clusters[clusters["kind"].isin(kinds)]

m = folium.Map(location=MAP_DEFAULT_CENTER, zoom_start=5)
for row in shown.itertuples():
    color = KIND_COLORS[row.kind]
    folium.CircleMarker(
        [row.latitude, row.longitude], radius=float(3 + 1.5 * np.sqrt(row.n_events)), color=color, fill=True,
        fill_color=color, fill_opacity=0.5, weight=1,
        tooltip=f"{row.jenis}: {row.n_events} gempa, Mw maks {row.mw_max:.1f} · "
                f"{fmt_local(row.local_start, row.tz)} · {row.location}",
    ).add_to(m)
st_folium(m, height=460, use_container_width=True, returned_objects=[], key="swarm_map")
st.caption("Ungu = swarm/doublet, merah = gempa utama–susulan, abu-abu = potongan deret gempa lain (mis. susulan di "
           "sepanjang ruptur panjang). Ukuran lingkaran = jumlah gempa.")

st.dataframe(
    pd.DataFrame({
        "Mulai (setempat)": [fmt_local(t, z) for t, z in zip(shown["local_start"], shown["tz"])],
        "Gempa": shown["n_events"],
        "Durasi (hari)": shown["duration_days"].round(1),
        "Mw maks": shown["mw_max"].map("{:.1f}".format),
        "Selisih Mw 1–2": shown["mw_gap"].map("{:.2f}".format),
        "Lokasi": shown["location"],
        "Jenis": shown["jenis"],
    }),
    hide_index=True, width="stretch",
)

if shown.empty:
    st.stop()

labels = {row.st_cluster: f"{row.local_start:%d %b %Y} · {row.location} · {row.jenis} ({row.n_events} gempa)"
          for row in shown.itertuples()}
cluster = st.selectbox("Detail klaster", list(labels), format_func=labels.get, key="swarm_cluster")
members = df.merge(cluster_events[cluster_events["st_cluster"] == cluster], on="eventID")
start = members["time"].min()
members = members.assign(hours=(members["time"] - start).dt.total_seconds() / 3600)
chart = alt.Chart(members[["hours", "mw", "depth", "mmi_class"]]).mark_circle(opacity=0.7).encode(
    x=alt.X("hours:Q", title="Jam sejak gempa pertama klaster"),
    y=alt.Y("mw:Q", title="Mw", scale=alt.Scale(zero=False)),
    size=alt.Size("mw:Q", legend=None, scale=alt.Scale(range=[20, 300])),
    color=alt.Color("mmi_class:N", title="MMI", scale=alt.Scale(domain=MMI_CLASSES,
                                                                range=[MMI_COLORS[c] for c in MMI_CLASSES])),
    tooltip=[alt.Tooltip("hours:Q", format=".1f"), "mw", "depth"],
).properties(height=300, title="Magnitudo terhadap waktu")
st.altair_chart(chart, width="stretch")
st.caption("Pada deret gempa utama–susulan, gempa terbesar muncul di awal lalu disusul gempa yang lebih kecil; pada "
           "swarm magnitudonya mirip sepanjang waktu. Catatan: gempa-gempa utama Lombok 2018 (29 Juli, 5 dan 19 "
           "Agustus) tidak ada di katalog, sehingga deret susulannya tampil sebagai swarm.")
