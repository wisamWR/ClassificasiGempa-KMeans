"""Gempa Terkini dari feed publik BMKG, dengan zona sumber dan konteks statistik zona."""
import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from common import (BMKG_URL, MAP_DEFAULT_CENTER, MMI_COLORS, fmt_local, fmt_years, load_bmkg, load_zoning,
                    result_box)
from gempa.bmkg import SOURCE_URL, FeedError

# Lebih jauh dari ini dari tepi zona: konteks zona kurang mewakili
FAR_FROM_ZONE_KM = 150

_, zones, geojson, _ = load_zoning()

st.title("🔴 Gempa Terkini")
st.caption(f"Sumber data: **BMKG** — [data.bmkg.go.id]({SOURCE_URL}). Magnitudo BMKG dikonversi ke Mw seperti "
           "katalog; intensitas dan konteks zona adalah perkiraan aplikasi ini, bukan informasi resmi BMKG.")
if st.button("🔄 Muat ulang"):
    load_bmkg.clear()
    st.rerun()


def has_tsunami_potential(text):
    text = (text or "").lower()
    return "berpotensi" in text and "tidak" not in text


def coordinates(lat, lon):
    return f"{abs(lat):.2f}° {'LS' if lat < 0 else 'LU'}, {lon:.2f}° BT"


def context_line(event):
    line = (f"Zona **{event['zone_id']}** · {event['zone_name']}: gempa utama M ≥ {event['mw']:.1f} di zona ini "
            f"rata-rata terjadi sekali tiap **{fmt_years(event['recurrence_years'])}**.")
    if event["distance_km"] > FAR_FROM_ZONE_KM:
        line += f" Episentrum {event['distance_km']:.0f} km dari tepi zona, jadi konteks ini kurang mewakili."
    return line


def events_table(events, extra_column, extra_label):
    return pd.DataFrame({
        "Waktu (setempat)": [fmt_local(t, z) for t, z in zip(events["local_time"], events["tz"])],
        "M": events["mag"].map("{:.1f}".format),
        "Kedalaman (km)": events["depth"].astype(int),
        "Wilayah": events["region"],
        extra_label: events[extra_column].fillna("-"),
        "MMI episentrum": events["mmi_class"],
        "Zona": events["zone_id"],
        "Gempa sebesar ini di zona": events["recurrence_years"].map(lambda y: f"sekali tiap {fmt_years(y)}"),
    })


def events_map(events):
    m = folium.Map(location=MAP_DEFAULT_CENTER, zoom_start=5)
    folium.GeoJson({"type": "FeatureCollection",
                    "features": [f for f in geojson["features"] if f["properties"]["layer"] == "DK"]},
                   style_function=lambda _: {"color": "#777777", "weight": 1, "fillOpacity": 0.03}).add_to(m)
    for row in events.itertuples():
        color = MMI_COLORS[row.mmi_class]
        folium.CircleMarker(
            [row.latitude, row.longitude], radius=3 + 1.5 * row.mag, color=color, fill=True, fill_color=color,
            fill_opacity=0.75, weight=1,
            tooltip=f"M {row.mag:.1f}, {row.depth:.0f} km · {fmt_local(row.local_time, row.tz)} · {row.region}",
        ).add_to(m)
    return m


# --- Gempa terbaru ---
try:
    latest, fetched_at = load_bmkg("terbaru")
except FeedError:
    st.warning(f"Data BMKG tidak bisa diambil saat ini (koneksi terputus atau server sibuk). Coba **Muat ulang**, "
               f"atau buka langsung [BMKG]({BMKG_URL}).")
    st.stop()

fetched_wib = fetched_at.tz_localize(None) + pd.Timedelta(hours=7)
st.caption(f"Diambil {fmt_local(fetched_wib, 'WIB')}; data disimpan sementara 5 menit.")
if latest.empty:
    st.info("Feed gempa terbaru BMKG sedang kosong.")
else:
    event = latest.iloc[0]
    if has_tsunami_potential(event["tsunami"]):
        st.error(f"⚠️ {event['tsunami']}. Ikuti arahan resmi [BMKG]({BMKG_URL}) dan BPBD setempat.")
    left, right = st.columns([3, 2])
    with left:
        result_box(
            f"M {event['mag']:.1f} · {event['region']}",
            f"<p>{fmt_local(event['local_time'], event['tz'])} · kedalaman {event['depth']:.0f} km · "
            f"{coordinates(event['latitude'], event['longitude'])}</p>"
            f"<p><b>Potensi tsunami:</b> {event['tsunami'] or '-'}<br><b>Dirasakan:</b> {event['felt'] or '-'}</p>"
            f"<p><b>Perkiraan intensitas di episentrum:</b> MMI {event['mmi_class']} (I₀ ≈ {event['intensity']:.1f}, "
            f"Mw {event['mw']:.1f})</p>",
            MMI_COLORS[event["mmi_class"]],
        )
        st.markdown(context_line(event))
    with right:
        if event["shakemap_url"]:
            st.image(event["shakemap_url"], caption="Peta guncangan (shakemap) © BMKG", width="stretch")

# --- Daftar gempa ---
st.subheader("Daftar gempa terbaru")
tab_big, tab_felt = st.tabs(["M ≥ 5", "Dirasakan"])
for tab, feed, column, label in [(tab_big, "terkini", "tsunami", "Potensi tsunami"),
                                 (tab_felt, "dirasakan", "felt", "Dirasakan (MMI)")]:
    with tab:
        try:
            events, _ = load_bmkg(feed)
        except FeedError:
            st.warning("Daftar ini tidak bisa diambil dari BMKG saat ini. Coba **Muat ulang** beberapa saat lagi.")
            continue
        if events.empty:
            st.info("Feed ini sedang kosong.")
            continue
        st_folium(events_map(events), height=420, use_container_width=True, returned_objects=[],
                  key=f"terkini_map_{feed}")
        st.dataframe(events_table(events, column, label), hide_index=True, width="stretch")

st.caption("Kolom *gempa sebesar ini di zona* memakai relasi Gutenberg–Richter zona (lihat **Zona Seismik**): "
           "rata-rata selang waktu gempa utama dengan magnitudo sebesar itu atau lebih di zona yang sama.")
