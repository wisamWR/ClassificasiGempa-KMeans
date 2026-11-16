"""Zona Seismik: peta zona per lapisan kedalaman, statistik hazard, dan grafik Gutenberg-Richter tiap zona."""
import folium
import streamlit as st
from folium.plugins import HeatMap
from streamlit_folium import st_folium

from common import (LAYER_CODES, LAYER_LABELS, MAP_DEFAULT_CENTER, colormap_for, events_table, fmt_prob, fmt_years,
                    gr_chart, load_events, load_focal, load_zoning, mc_by_year, zone_layer)
from gempa.focal import FAULT_LABELS, FAULT_TYPES

# Warna teks Streamlit yang mendekati FAULT_COLORS
FAULT_BADGE = {"naik": "red", "turun": "blue", "geser": "green", "oblik": "gray"}

# label -> (kolom zone_stats, warna rendah -> tinggi, format legenda)
ZONE_METRICS = {
    "Peluang M ≥ 6 dalam 10 tahun": ("p10_m6", ["#fee5d9", "#fb6a4a", "#a50f15"]),
    "Peluang M ≥ 5 dalam 10 tahun": ("p10_m5", ["#fee5d9", "#fb6a4a", "#a50f15"]),
    "Periode ulang M ≥ 6 (tahun)": ("return_period_m6", ["#a50f15", "#fb6a4a", "#fee5d9"]),
    "b-value": ("b", ["#2166ac", "#f7f7f7", "#b2182b"]),
}

df = load_events()
models, zones, geojson, metadata = load_zoning()

st.title("🧭 Zona Seismik")
st.caption(
    f"{len(zones)} zona sumber gempa hasil K-Means per lapisan kedalaman, dari "
    f"{metadata['n_zoning_events']:,} gempa utama di atas batas kelengkapan katalog yang lolos verifikasi. "
    "Klik zona di peta atau pilih dari daftar untuk melihat detailnya."
)

c1, c2, c3 = st.columns([2, 2, 1])
layer = c1.radio("Lapisan kedalaman", LAYER_CODES, format_func=LAYER_LABELS.get, horizontal=True, key="zona_layer")
metric_label = c2.selectbox("Warna zona", list(ZONE_METRICS), key="zona_metric")
heatmap = c3.toggle("Heatmap gempa", value=False, key="zona_heatmap", help="Kepadatan gempa utama lapisan ini.")

column, colors = ZONE_METRICS[metric_label]
layer_zones = zones[zones["layer"] == layer]

m = folium.Map(location=MAP_DEFAULT_CENTER, zoom_start=5, tiles="OpenStreetMap")
colormap = colormap_for(layer_zones[column], colors, metric_label)
zone_layer(geojson, layer_zones, layer, column, colormap,
           ["zone_id", "name", "b", "return_period_m6", "p10_m6", "p50_m6"],
           ["Zona", "Nama", "b-value", "Periode ulang M6 (tahun)", "P(M≥6, 10 th)", "P(M≥6, 50 th)"],
           LAYER_LABELS[layer]).add_to(m)
colormap.add_to(m)
if heatmap:
    layer_events = df[df["in_zoning_set"] & (df["layer"] == layer)]
    HeatMap(layer_events[["latitude", "longitude"]].to_numpy().tolist(), radius=8, blur=10,
            name="Heatmap gempa").add_to(m)
out = st_folium(m, height=520, use_container_width=True, returned_objects=["last_active_drawing"],
                key=f"zona_map_{layer}")

# Klik zona di peta -> pilih zona tersebut di daftar
clicked = ((out or {}).get("last_active_drawing") or {}).get("properties", {}).get("zone_id")
if clicked and clicked != st.session_state.get("zona_last_click"):
    st.session_state["zona_last_click"] = clicked
    if clicked in set(layer_zones["zone_id"]):
        st.session_state["zona_selected"] = clicked

st.dataframe(
    layer_zones.assign(
        b=[f"{b:.2f} ± {s:.2f}" for b, s in zip(layer_zones["b"], layer_zones["b_sigma"])],
        mw_max=layer_zones["mw_max"].map("{:.1f}".format),
        return_period_m5=layer_zones["return_period_m5"].map(fmt_years),
        return_period_m6=layer_zones["return_period_m6"].map(fmt_years),
        p10_m6=layer_zones["p10_m6"].map(fmt_prob),
        p50_m6=layer_zones["p50_m6"].map(fmt_prob),
    )[["zone_id", "name", "n_events", "mw_max", "b", "return_period_m5", "return_period_m6", "p10_m6", "p50_m6"]]
    .rename(columns={"zone_id": "Zona", "name": "Nama", "n_events": "Gempa utama", "mw_max": "Mw maks",
                     "b": "b-value", "return_period_m5": "Periode ulang M ≥ 5",
                     "return_period_m6": "Periode ulang M ≥ 6",
                     "p10_m6": "P(M ≥ 6, 10 th)", "p50_m6": "P(M ≥ 6, 50 th)"}),
    hide_index=True, width="stretch",
)

# --- Detail zona ---
options = list(layer_zones["zone_id"])
if st.session_state.get("zona_selected") not in options:
    st.session_state["zona_selected"] = options[0]
names = dict(zip(zones["zone_id"], zones["name"]))
zone_id = st.selectbox("Detail zona", options, format_func=lambda z: f"{z} · {names[z]}", key="zona_selected")
z = zones.set_index("zone_id").loc[zone_id]

st.subheader(f"{zone_id} · {z['name']}")
cols = st.columns(4)
cols[0].metric("Gempa utama", f"{z['n_events']:,}")
cols[1].metric("Mw terbesar", f"{z['mw_max']:.1f}")
cols[2].metric("b-value", f"{z['b']:.2f} ± {z['b_sigma']:.2f}")
cols[3].metric("Mc zona", f"{z['mc']:.1f}")
cols = st.columns(4)
for col, (mag, t) in zip(cols, [(5, 10), (5, 50), (6, 10), (6, 50)]):
    col.metric(f"P(M ≥ {mag} dalam {t} th)", fmt_prob(z[f"p{t}_m{mag}"]))
st.caption(
    f"Periode ulang M ≥ 5: {fmt_years(z['return_period_m5'])}; M ≥ 6: {fmt_years(z['return_period_m6'])}. "
    f"Kedalaman median {z['depth_median']:.0f} km. Selama katalog tercatat {z['observed_m5']} gempa utama M ≥ 5 "
    f"(model: {z['expected_m5_catalog']:.0f}) dan {z['observed_m6']} gempa M ≥ 6 "
    f"(model: {z['expected_m6_catalog']:.0f})."
    + (" b-value memakai nilai lapisan karena data zona terlalu sedikit." if z["b_source"] == "lapisan" else "")
)

left, right = st.columns([3, 2])
with left:
    years_ok = [year for year, mc in mc_by_year().items() if mc <= z["mc"] + 1e-9]
    zone_events = df[df["in_zoning_set"] & (df["zone_id"] == zone_id)]
    mags = zone_events.loc[zone_events["time"].dt.year.isin(years_ok), "mw"]
    st.altair_chart(gr_chart(mags, z["exposure_years"], z["mc"], z["b"], z["n_b"] / z["exposure_years"],
                             "Distribusi frekuensi–magnitudo (tahun yang katalognya lengkap)"), width="stretch")
with right:
    st.markdown("**Gempa terbesar di zona ini**")
    zone_all = df[(df["zone_id"] == zone_id) & ~df["suspect_event"]]
    st.dataframe(events_table(zone_all, 5).drop(columns=["MMI"]), hide_index=True, width="stretch")

    zone_fm = load_focal().query("zone_id == @zone_id")
    st.markdown(f"**Jenis sesar** ({len(zone_fm)} mekanisme fokal BMKG)")
    if len(zone_fm) >= 5:
        shares = zone_fm["fault_type"].value_counts(normalize=True).reindex(FAULT_TYPES, fill_value=0)
        st.markdown(" · ".join(f":{FAULT_BADGE[t]}[{FAULT_LABELS[t]} **{share:.0%}**]" for t, share in shares.items()))
    else:
        st.caption("Mekanisme fokal di zona ini terlalu sedikit untuk dirangkum.")
