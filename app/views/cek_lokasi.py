"""Cek Lokasi: cari kota atau klik peta -> zona sumber gempa, peluang gempa, dan riwayat gempa di sekitarnya."""
import altair as alt
import folium
import numpy as np
import streamlit as st
from streamlit_folium import st_folium

from common import (GEOCODE_VIEWBOX, LAYER_LABELS, MAP_DEFAULT_CENTER, MMI_ADVICE, MMI_CLASSES, MMI_COLORS,
                    SIGNIFICANT_MW, events_table, fmt_prob, fmt_years, load_events, load_zoning, result_box,
                    search_location)
from gempa.geo import haversine_km
from gempa.location import radius_hazard, zones_at_point
from gempa.stats import poisson_probability
from gempa.timezones import local_timezone

MAX_MAP_EVENTS = 300
TIMELINE_MIN_MW = 4.0
# Lebih jauh dari ini dari tepi zona: statistik zona kurang mewakili lokasi
FAR_FROM_ZONE_KM = 100

df = load_events()
_, zones, geojson, _ = load_zoning()
real = df[~df["suspect_event"]]
zoning_set = df[df["in_zoning_set"]]
period_years = (df["time"].max() - df["time"].min()).total_seconds() / (365.25 * 86400)

st.title("📍 Cek Lokasi")
st.caption("Cari nama kota atau klik peta untuk melihat zona sumber gempa, peluang gempa, dan riwayat gempa di sekitarnya.")

c1, c2 = st.columns([3, 1])
query = c1.text_input("Nama kota/daerah", key="cek_query",
                      placeholder="Ketik nama kota, misal: Cianjur, Palu, Ambon").strip()
radius = c2.slider("Radius (km)", 25, 300, 100, 25, key="cek_radius",
                   help="Gempa dengan episentrum dalam radius ini dihitung.")

if query and query != st.session_state.get("cek_last_query"):
    place, error = search_location(query)
    st.session_state["cek_last_query"] = query
    st.session_state["cek_error"] = error
    if place:
        st.session_state["cek_point"] = place
    else:
        st.session_state.pop("cek_point", None)
elif not query:
    st.session_state.pop("cek_last_query", None)
    st.session_state.pop("cek_error", None)
if st.session_state.get("cek_error"):
    st.warning(st.session_state["cek_error"])

point = st.session_state.get("cek_point")


def location_map(point, near):
    if point is None:
        m = folium.Map(location=MAP_DEFAULT_CENTER, zoom_start=5)
    else:
        m = folium.Map(location=[point["lat"], point["lon"]], zoom_start=int(round(10 - np.log2(radius / 25))))
    shallow = [f for f in geojson["features"] if f["properties"]["layer"] == "DK"]
    folium.GeoJson({"type": "FeatureCollection", "features": shallow}, name="Zona dangkal",
                   style_function=lambda _: {"color": "#555555", "weight": 1, "fillOpacity": 0.05},
                   tooltip=folium.GeoJsonTooltip(["zone_id", "name"], aliases=["Zona", "Nama"])).add_to(m)
    if point is not None:
        for row in near.nlargest(MAX_MAP_EVENTS, "mw").itertuples():
            color = MMI_COLORS[row.mmi_class]
            folium.CircleMarker([row.latitude, row.longitude], radius=2 + row.mw, color=color, fill=True,
                                fill_color=color, fill_opacity=0.7, weight=1,
                                tooltip=f"Mw {row.mw:.1f}, {row.depth} km, {row.local_time:%d %b %Y}").add_to(m)
        folium.Circle([point["lat"], point["lon"]], radius=radius * 1000, color="#d62728", fill=False,
                      weight=2).add_to(m)
        folium.Marker([point["lat"], point["lon"]], icon=folium.Icon(color="red", icon="star"),
                      tooltip=point["label"]).add_to(m)
    return m


def show_map(point, near):
    out = st_folium(location_map(point, near), height=480, use_container_width=True,
                    returned_objects=["last_clicked"], key="cek_map")
    click = (out or {}).get("last_clicked")
    if click and (click["lat"], click["lng"]) != st.session_state.get("cek_last_click"):
        st.session_state["cek_last_click"] = (click["lat"], click["lng"])
        (lat_min, lon_min), (lat_max, lon_max) = GEOCODE_VIEWBOX
        if lat_min <= click["lat"] <= lat_max and lon_min <= click["lng"] <= lon_max:
            st.session_state["cek_point"] = {"lat": click["lat"], "lon": click["lng"],
                                             "label": f"Titik di peta ({click['lat']:.3f}, {click['lng']:.3f})"}
            st.session_state.pop("cek_error", None)
            st.rerun()
        else:
            st.warning("Titik di luar wilayah katalog (lintang −12° s.d. 7°, bujur 94° s.d. 142°).")
    st.caption("Klik peta untuk memilih titik lain. Garis abu-abu = batas zona dangkal; lingkaran merah = radius.")


if point is None:
    st.info("Ketik nama kota di atas, atau klik sebuah titik di peta.")
    show_map(None, None)
    st.stop()

lat, lon = point["lat"], point["lon"]
distance = haversine_km(lat, lon, real["latitude"].to_numpy(), real["longitude"].to_numpy())
near = real.assign(distance=distance)[distance <= radius]
summary, contributions = radius_hazard(zoning_set, zones, lat, lon, radius)
tz = local_timezone([lat], [lon])[0]

result_box(f"📍 {point['label']}",
           f"<p>Lintang {lat:.3f}, bujur {lon:.3f} · radius {radius} km · zona waktu {tz}</p>", "#4C72B0")

# --- 1. Peluang gempa ---
st.subheader(f"Peluang gempa utama dengan episentrum dalam radius {radius} km")
if summary["n_events"] == 0:
    st.info("Tidak ada gempa utama di atas batas kelengkapan katalog dalam radius ini, sehingga peluang tidak dapat "
            "diperkirakan dari katalog. Coba perbesar radius.")
else:
    cols = st.columns(4)
    for col, (m, t) in zip(cols, [(5, 10), (5, 50), (6, 10), (6, 50)]):
        col.metric(f"M ≥ {m} dalam {t} tahun", fmt_prob(summary[f"p{t}_m{m}"]))
    observed = int(((zoning_set["mw"] >= SIGNIFICANT_MW)
                    & (haversine_km(lat, lon, zoning_set["latitude"].to_numpy(),
                                    zoning_set["longitude"].to_numpy()) <= radius)).sum())
    shallow_m6 = contributions.loc[contributions["layer"] == "DK", "rate_m6"].sum()
    st.caption(
        f"Rata-rata sekali tiap {fmt_years(summary['return_period_m5'])} untuk M ≥ 5 dan "
        f"{fmt_years(summary['return_period_m6'])} untuk M ≥ 6, termasuk gempa menengah dan dalam yang guncangannya "
        f"di permukaan jauh lebih lemah. Dari gempa dangkal (< 70 km) saja, peluang M ≥ 6 dalam 50 tahun "
        f"{fmt_prob(float(poisson_probability(shallow_m6, 50)))}. Selama katalog ({period_years:.0f} tahun) tercatat "
        f"{observed} gempa utama M ≥ 5 di radius ini; model memperkirakan {summary['rate_m5'] * period_years:.0f}."
    )
    with st.expander("Cara menghitung"):
        st.markdown(
            "Laju tahunan tiap zona berasal dari relasi Gutenberg–Richter zona tersebut (lihat **Zona Seismik**). "
            "Laju zona dibagi rata ke gempa utamanya, sehingga kontribusi zona = laju zona × (gempa zona di dalam "
            "radius ÷ seluruh gempa zona). Peluang minimal satu kejadian dalam T tahun = 1 − exp(−laju · T) "
            "(model Poisson). Ini statistik katalog, bukan peta bahaya resmi."
        )
        st.dataframe(
            contributions.assign(share=(contributions["share"] * 100).round(1),
                                 rate_m5=contributions["rate_m5"].round(3), rate_m6=contributions["rate_m6"].round(4),
                                 layer=contributions["layer"].map(LAYER_LABELS))
            .rename(columns={"zone_id": "Zona", "name": "Nama", "layer": "Lapisan", "share": "Bagian zona (%)",
                             "rate_m5": "Laju M ≥ 5 / tahun", "rate_m6": "Laju M ≥ 6 / tahun"}),
            hide_index=True, width="stretch",
        )

show_map(point, near)

# --- 2. Zona di bawah lokasi ---
st.subheader("Zona sumber gempa di bawah lokasi ini")
here = zones_at_point(geojson, lat, lon).merge(zones, on=["zone_id", "layer"])
st.dataframe(
    here.assign(
        layer=here["layer"].map(LAYER_LABELS),
        posisi=[("di dalam zona" if inside else f"{d:.0f} km dari tepi zona")
                for inside, d in zip(here["inside"], here["distance_km"])],
        b=[f"{b:.2f} ± {s:.2f}" for b, s in zip(here["b"], here["b_sigma"])],
        return_period_m6=here["return_period_m6"].map(fmt_years),
        p10_m6=here["p10_m6"].map(fmt_prob),
    )[["layer", "zone_id", "name", "posisi", "b", "return_period_m6", "p10_m6"]]
    .rename(columns={"layer": "Lapisan", "zone_id": "Zona", "name": "Nama", "posisi": "Posisi", "b": "b-value",
                     "return_period_m6": "Periode ulang M ≥ 6 (zona)", "p10_m6": "Peluang M ≥ 6 dalam 10 th (zona)"}),
    hide_index=True, width="stretch",
)
if (here.loc[here["layer"] == "DK", "distance_km"] > FAR_FROM_ZONE_KM).all():
    st.warning(f"Lokasi ini lebih dari {FAR_FROM_ZONE_KM} km dari zona gempa dangkal mana pun: menurut katalog, "
               "kegempaan dangkal di sekitarnya rendah, dan statistik zona terdekat kurang mewakili lokasi ini.")
st.caption("Statistik zona berlaku untuk seluruh wilayah zona yang luasnya ratusan km, bukan khusus titik ini. "
           "Gempa dangkal paling berpengaruh pada guncangan di permukaan; lapisan dalam hanya punya satu zona yang "
           "poligonnya mencakup wilayah luas.")

# --- 3. Riwayat gempa ---
st.subheader(f"Riwayat gempa dalam radius {radius} km")
if near.empty:
    st.info("Tidak ada catatan gempa dalam radius ini di katalog.")
    st.stop()

strongest = near.loc[near["intensity"].idxmax()]
cols = st.columns(4)
cols[0].metric("Total Kejadian", f"{len(near):,}")
cols[1].metric(f"Gempa Mw ≥ {SIGNIFICANT_MW:.0f}", f"{(near['mw'] >= SIGNIFICANT_MW).sum():,}")
cols[2].metric("Mw Tertinggi", f"{near['mw'].max():.1f}")
cols[3].metric("Intensitas Tertinggi", f"MMI {strongest['mmi_class']}")
result_box(f"Intensitas tertinggi tercatat: MMI {strongest['mmi_class']}",
           f"<p>Mw {strongest['mw']:.1f}, kedalaman {strongest['depth']} km, {strongest['local_time']:%d %b %Y} "
           f"({strongest['location']}).</p><p><b>Rekomendasi:</b> {MMI_ADVICE[strongest['mmi_class']]}</p>",
           MMI_COLORS[strongest["mmi_class"]])

st.markdown("#### 📜 5 Gempa Terbesar")
st.dataframe(events_table(near, 5, distance=True), hide_index=True, width="stretch")

timeline = near[near["mw"] >= TIMELINE_MIN_MW]
if not timeline.empty:
    st.markdown(f"#### 🕒 Linimasa Gempa Mw ≥ {TIMELINE_MIN_MW:.0f}")
    chart = alt.Chart(timeline[["local_time", "mw", "depth", "mmi_class", "location"]]).mark_circle(opacity=0.7).encode(
        x=alt.X("local_time:T", title="Waktu"),
        y=alt.Y("mw:Q", title="Mw", scale=alt.Scale(zero=False)),
        size=alt.Size("mw:Q", legend=None, scale=alt.Scale(range=[15, 400])),
        color=alt.Color("mmi_class:N", title="MMI", scale=alt.Scale(domain=MMI_CLASSES,
                                                                    range=[MMI_COLORS[c] for c in MMI_CLASSES])),
        tooltip=[alt.Tooltip("local_time:T", title="Waktu", format="%d %b %Y %H:%M"), "mw", "depth", "mmi_class",
                 "location"],
    ).properties(height=300)
    st.altair_chart(chart, width="stretch")
