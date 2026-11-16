"""Data, model, dan komponen tampilan yang dipakai semua halaman."""
import json
import sys
from pathlib import Path

import altair as alt
import folium
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from branca.colormap import LinearColormap
from geopy.exc import GeopyError
from geopy.geocoders import Nominatim

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

from gempa import bmkg, paths  # noqa: E402
from gempa.focal import FAULT_LABELS, FAULT_TYPES  # noqa: E402
from gempa.intensity import MMI_CLASSES, MMI_DESCRIPTIONS, epicentral_intensity, mmi_class  # noqa: E402
from gempa.location import zone_context  # noqa: E402
from gempa.magnitude import to_mw  # noqa: E402
from gempa.stats import annual_rate  # noqa: E402
from gempa.timezones import to_local_time  # noqa: E402
from gempa.zoning import DEPTH_LAYERS, depth_layer  # noqa: E402

# --- Konstanta tampilan ---
MMI_COLORS = {"I-II": "#28a745", "III-IV": "#ffc107", "V-VI": "#fd7e14", "VII+": "#dc3545"}
MMI_ADVICE = {
    "I-II": "Tetap tenang, rutinitas normal.",
    "III-IV": "Waspada benda jatuh, lindungi kepala.",
    "V-VI": "Jauhi kaca/jendela, berlindung di bawah meja.",
    "VII+": "Berlindung saat guncangan, lalu evakuasi ke area terbuka. Hindari gedung retak.",
}
LAYER_LABELS = {"DK": "Dangkal (< 70 km)", "MN": "Menengah (70–300 km)", "DL": "Dalam (> 300 km)"}
LAYER_CODES = [code for code, _, _ in DEPTH_LAYERS]
FAULT_COLORS = {"naik": "#d62728", "turun": "#1f77b4", "geser": "#2ca02c", "oblik": "#9e9e9e"}
# Profil klaster K-Means (relatif antar-klaster, bukan risiko absolut)
PROFILE_ORDER = ["Low", "Moderate", "High", "Very High"]
SIGNIFICANT_MW = 5.0
MAP_DEFAULT_CENTER = [-2.5, 118.0]
# Batas wilayah katalog (lat, lon) agar hasil geocoding tidak nyasar ke luar area data
GEOCODE_VIEWBOX = [(-12.0, 94.0), (7.0, 142.0)]
BMKG_URL = "https://www.bmkg.go.id"
INARISK_URL = "https://inarisk.bnpb.go.id"

DISCLAIMER = (
    "Aplikasi ini **tidak memprediksi gempa**. Statistik dihitung dari katalog BMKG 2008–2025 dan bukan peta bahaya "
    f"resmi. Informasi gempa terkini dan peringatan dini: [BMKG]({BMKG_URL}). Kajian risiko bencana: "
    f"[InaRISK BNPB]({INARISK_URL})."
)

# Semua CSS di satu tempat. Warna teks mengikuti tema Streamlit; aksen per kartu lewat variabel --accent.
CSS = """
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 100%; }
    .info-card {
        background-color: rgba(128, 128, 128, 0.08); border-top: 4px solid var(--accent, gray);
        padding: 14px; border-radius: 8px; height: 100%;
    }
    .info-card h4 { margin: 0 0 6px 0; padding: 0; color: var(--accent, inherit); }
    .info-card p { font-size: 0.85rem; margin: 0 0 4px 0; }
    .result-box {
        padding: 14px 20px; margin: 8px 0 16px 0; border: 2px solid var(--accent, gray); border-radius: 10px;
        background-color: rgba(128, 128, 128, 0.05);
    }
    .result-box h3 { margin: 0 0 4px 0; padding: 0; color: var(--accent, inherit); }
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def card(title, body, accent):
    st.markdown(f'<div class="info-card" style="--accent: {accent};"><h4>{title}</h4>{body}</div>',
                unsafe_allow_html=True)


def result_box(title, body, accent):
    st.markdown(f'<div class="result-box" style="--accent: {accent};"><h3>{title}</h3>{body}</div>',
                unsafe_allow_html=True)


def mmi_legend():
    for col, mmi in zip(st.columns(4), MMI_CLASSES):
        with col:
            card(f"MMI {mmi}", f"<p><b>Efek:</b> {MMI_DESCRIPTIONS[mmi]}</p><p><b>Saran:</b> {MMI_ADVICE[mmi]}</p>",
                 MMI_COLORS[mmi])
    st.caption("Intensitas diestimasi dari Mw dan kedalaman dengan relasi Gutenberg–Richter (1956): "
               "I₀ = 1,5·Mw − 3,5·log₁₀(h) + 3. Ini perkiraan guncangan di atas pusat gempa, bukan di lokasi Anda.")


# --- Data & model ---
@st.cache_data(show_spinner="Memuat katalog gempa...")
def load_events():
    """Katalog + profil klaster (gempa.train) + zona tiap event (gempa.zoning) + waktu setempat."""
    df = pd.read_parquet(paths.PROCESSED_EVENTS)
    zones = pd.read_parquet(paths.ZONE_EVENTS)
    df = df.merge(zones, on="eventID", how="left", validate="one_to_one")
    df["mmi_class"] = df["mmi_class"].astype(str)
    df["suspect_event"] = df["suspect_event"].fillna(False).astype(bool)
    df["in_zoning_set"] = df["in_zoning_set"].fillna(False).astype(bool)
    df["layer"] = depth_layer(df["depth"])
    df["local_time"], df["tz"] = to_local_time(df["time"], df["latitude"], df["longitude"])
    return df


@st.cache_data
def load_focal():
    """Mekanisme fokal (gempa.focal) + Mw, zona, dan lapisan event; tanpa event semu."""
    fm = pd.read_parquet(paths.FOCAL_MECHANISMS)
    events = load_events()[["eventID", "mw", "zone_id", "layer", "suspect_event", "location"]]
    fm = fm.merge(events, on="eventID", how="inner", validate="one_to_one")
    fm = fm[~fm["suspect_event"]].reset_index(drop=True)
    fm["jenis"] = fm["fault_type"].astype(str).map(FAULT_LABELS)
    return fm


def fault_type_chart(fm, by=None, sort=None, height=220):
    """Bar proporsi jenis sesar (opsional per kelompok `by`)."""
    labels = [FAULT_LABELS[t] for t in FAULT_TYPES]
    color = alt.Color("jenis:N", title="Jenis sesar", sort=labels,
                      scale=alt.Scale(domain=labels, range=[FAULT_COLORS[t] for t in FAULT_TYPES]))
    x = alt.X("count():Q", stack="normalize", title="Proporsi", axis=alt.Axis(format="%"))
    chart = alt.Chart(fm[["jenis"] + ([by] if by else [])]).mark_bar().encode(
        x=x, color=color, order=alt.Order("jenis_order:Q"), tooltip=["jenis", "count()"],
    ).transform_calculate(jenis_order=f"indexof({labels}, datum.jenis)")
    if by:
        chart = chart.encode(y=alt.Y(f"{by}:N", title=None, sort=sort))
    # Kelompok: tinggi per batang tetap agar label tidak bertumpuk; tanpa kelompok: satu batang setinggi `height`
    return chart.properties(height=alt.Step(36) if by else height)


@st.cache_resource
def load_zoning():
    models = joblib.load(paths.ZONING_MODEL)
    zones = pd.read_csv(paths.ZONE_STATS)
    geojson = json.loads(paths.ZONES_GEOJSON.read_text(encoding="utf-8"))
    metadata = json.loads(paths.ZONING_METADATA.read_text(encoding="utf-8"))
    return models, zones, geojson, metadata


@st.cache_resource
def load_kmeans():
    # Pipeline = MinMaxScaler + KMeans (fitur: mag, depth)
    pipeline = joblib.load(paths.PIPELINE_FILE)
    info = pd.read_csv(paths.CLUSTER_INFO_FILE).dropna(subset=["cluster", "label"])
    metadata = json.loads(paths.METADATA_FILE.read_text(encoding="utf-8"))
    return pipeline, dict(zip(info["cluster"], info["label"])), metadata


@st.cache_data
def load_report():
    return json.loads(paths.CLEAN_REPORT.read_text(encoding="utf-8"))


def mc_by_year():
    return {row["year"]: row["mc"] for row in load_report()["completeness"]["by_year"]}


# Feed BMKG disimpan 5 menit: cukup segar dan jauh di bawah batas 60 permintaan/menit
@st.cache_data(ttl=300, show_spinner="Mengambil data gempa terkini dari BMKG...")
def load_bmkg(feed):
    """(gempa feed BMKG + Mw, intensitas, zona & konteks zona, waktu setempat; waktu pengambilan UTC)."""
    df = bmkg.parse_feed(bmkg.fetch_feed(feed))
    fetched_at = pd.Timestamp.now(tz="UTC")
    if df.empty:
        return df, fetched_at
    models, zones, geojson, _ = load_zoning()
    df["mw"] = to_mw(df["mag"], pd.Series(bmkg.MAG_TYPE, index=df.index))["mw"].to_numpy()
    df["intensity"] = epicentral_intensity(df["mw"].to_numpy(), df["depth"].to_numpy())
    df["mmi_class"] = np.asarray(mmi_class(df["intensity"].to_numpy())).astype(str)
    df["local_time"], df["tz"] = to_local_time(df["time"], df["latitude"], df["longitude"])
    context = zone_context(models, zones, geojson, df["latitude"], df["longitude"], df["depth"], df["mw"])
    return pd.concat([df, context], axis=1), fetched_at


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def geocode(query):
    geolocator = Nominatim(user_agent="klasifikasi-gempa-kmeans-viewer")
    location = geolocator.geocode(query, timeout=10, viewbox=GEOCODE_VIEWBOX, bounded=True)
    if location is None:
        return None
    return {"lat": location.latitude, "lon": location.longitude, "label": location.address}


def search_location(query):
    """(lokasi, pesan galat) untuk nama tempat."""
    try:
        place = geocode(query)
    except GeopyError:
        return None, "Gagal koneksi ke layanan pencarian lokasi. Coba lagi, atau klik titik di peta."
    if place is None:
        return None, f"Lokasi \"{query}\" tidak ditemukan di wilayah Indonesia."
    return place, None


# --- Format ---
def fmt_local(ts, tz):
    return f"{ts:%d %b %Y %H:%M} {tz}"


def fmt_years(years):
    if not np.isfinite(years):
        return "tidak dapat dihitung"
    if years < 1:
        return f"±{max(years * 12, 1):.0f} bulan"
    return f"±{years:.0f} tahun" if years >= 10 else f"±{years:.1f} tahun"


def fmt_prob(p):
    if p >= 0.995:
        return "> 99%"
    if p < 0.005:
        return "< 1%"
    return f"{p:.0%}"


def events_table(events, n=10, distance=False):
    top = events.nlargest(n, "mw")
    table = pd.DataFrame({
        "Waktu (setempat)": [fmt_local(t, z) for t, z in zip(top["local_time"], top["tz"])],
        "Mw": top["mw"].round(1),
        "Kedalaman (km)": top["depth"],
        "MMI": top["mmi_class"],
        "Gempa utama": top["is_mainshock"],
        "Lokasi": top["location"],
    })
    if distance:
        table["Jarak (km)"] = top["distance"].round(0)
    return table


# --- Peta & grafik ---
def zone_layer(geojson, zones, layer, column, colormap, tooltip_fields, tooltip_aliases, name, show=True):
    """GeoJson zona satu lapisan, diwarnai menurut kolom statistik zona."""
    values = zones.set_index("zone_id")[column]
    features = [f for f in geojson["features"] if f["properties"]["layer"] == layer]

    def style(feature):
        value = values.get(feature["properties"]["zone_id"], np.nan)
        color = colormap(value) if np.isfinite(value) else "#999999"
        return {"fillColor": color, "color": "#444444", "weight": 1, "fillOpacity": 0.5}

    return folium.GeoJson(
        {"type": "FeatureCollection", "features": features}, name=name, show=show, style_function=style,
        highlight_function=lambda _: {"weight": 3, "fillOpacity": 0.7},
        tooltip=folium.GeoJsonTooltip(tooltip_fields, aliases=tooltip_aliases, localize=True),
    )


def colormap_for(values, colors, caption):
    values = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna()
    return LinearColormap(colors, vmin=float(values.min()), vmax=float(values.max()), caption=caption)


def gr_chart(mags, exposure_years, mc, b, rate_mc, title=""):
    """Grafik Gutenberg-Richter: laju tahunan kumulatif N(>= M) observasi + garis log N = a - bM."""
    mags = np.asarray(mags, dtype=float)
    grid = np.round(np.arange(np.floor(mags.min() * 10) / 10, mags.max() + 0.05, 0.1), 1)
    observed = pd.DataFrame({"Mw": grid, "Laju tahunan N(≥ Mw)": [(mags >= g).sum() / exposure_years for g in grid],
                             "seri": "observasi"})
    mm = np.linspace(mc, max(mags.max(), 7.0), 30)
    fitted_label = f"G-R, b = {b:.2f}"
    fitted = pd.DataFrame({"Mw": mm, "Laju tahunan N(≥ Mw)": annual_rate(rate_mc, b, mc, mm), "seri": fitted_label})
    x = alt.X("Mw", scale=alt.Scale(zero=False))
    y = alt.Y("Laju tahunan N(≥ Mw)", scale=alt.Scale(type="log"))
    color = alt.Color("seri", title=None, legend=alt.Legend(orient="top"),
                      scale=alt.Scale(domain=["observasi", fitted_label], range=["#7f7f7f", "#d62728"]))
    points = alt.Chart(observed[observed["Laju tahunan N(≥ Mw)"] > 0]).mark_circle(size=40).encode(
        x=x, y=y, color=color, tooltip=["Mw", alt.Tooltip("Laju tahunan N(≥ Mw)", format=".3g")])
    line = alt.Chart(fitted).mark_line(strokeWidth=2).encode(x=x, y=y, color=color)
    mc_rule = alt.Chart(pd.DataFrame({"Mc": [mc]})).mark_rule(strokeDash=[4, 4]).encode(
        x=alt.X("Mc", title="Mw", scale=alt.Scale(zero=False)))
    return (points + line + mc_rule).properties(title=title, height=320)
