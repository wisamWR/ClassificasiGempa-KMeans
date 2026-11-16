import sys
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from pathlib import Path
from folium.plugins import FastMarkerCluster
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.exc import GeopyError

# Path relatif terhadap root project, jadi app bisa dijalankan dari folder mana pun
ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
sys.path.insert(0, str(ROOT_DIR / "src"))

from gempa.geo import haversine_km  # noqa: E402
from gempa.intensity import MMI_CLASSES, MMI_DESCRIPTIONS, epicentral_intensity, mmi_class  # noqa: E402

# --- 1. Page Config ---
st.set_page_config(layout="wide", page_title="Earthquake Risk Viewer")

# --- CSS: Wide Mode tapi Rapi ---
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 2rem;
            padding-left: 1rem;
            padding-right: 1rem;
            max-width: 100%;
        }
        h1 { margin-bottom: 0rem; }
    </style>
""", unsafe_allow_html=True)

# --- CONSTANTS: SAFETY TIPS ---
TIPS_SEBELUM_GEMPA = [
    "🏠 Identifikasi zona aman di rumah: di bawah meja kokoh, jauh dari jendela.",
    "🎒 Siapkan tas siaga darurat berisi air minum, makanan kering, senter, obat pribadi.",
    "📍 Kenali rute evakuasi dan titik kumpul di lingkungan Anda.",
    "🔧 Pastikan lemari dan rak buku terikat ke dinding untuk mencegah roboh.",
    "📱 Aktifkan notifikasi peringatan dini BMKG di smartphone."
]

TIPS_SAAT_GEMPA = [
    "🛑 Jangan panik. Tetap tenang dan bertindak cepat.",
    "🙇 Berlindung di bawah meja atau benda kokoh, lindungi kepala dan leher.",
    "🚪 Jauhi jendela, cermin, lemari, dan benda berat yang bisa jatuh.",
    "🏃 Jika di luar ruangan, jauhi gedung, tiang listrik, dan pohon besar.",
    "🚗 Jika sedang berkendara, hentikan kendaraan di tempat terbuka."
]

TIPS_SETELAH_GEMPA = [
    "🔍 Periksa diri sendiri dan orang sekitar. Berikan pertolongan pertama jika perlu.",
    "⚠️ Waspada gempa susulan. Jangan masuk gedung yang retak.",
    "📻 Dengarkan informasi resmi dari BMKG melalui radio atau aplikasi.",
    "🔥 Periksa instalasi gas dan listrik. Matikan jika ada kebocoran.",
    "📞 Hubungi keluarga untuk memberi kabar, hindari telepon berlebihan."
]

# Kelas intensitas (MMI) estimasi per gempa: dasar warna, legenda, dan filter
MMI_COLORS = {
    'I-II': '#28a745',      # Green
    'III-IV': '#ffc107',    # Yellow
    'V-VI': '#fd7e14',      # Orange
    'VII+': '#dc3545'       # Red
}
MMI_ADVICE = {
    'I-II': 'Tetap tenang, rutinitas normal.',
    'III-IV': 'Waspada benda jatuh, lindungi kepala.',
    'V-VI': 'Jauhi kaca/jendela, berlindung di bawah meja.',
    'VII+': 'Berlindung saat guncangan, lalu evakuasi ke area terbuka. Hindari gedung retak.'
}

# Profil klaster K-Means (relatif antar-klaster, bukan risiko absolut)
RISK_ORDER = ['Low', 'Moderate', 'High', 'Very High']

MAP_DEFAULT_CENTER = [-2.5, 118.0]
MAX_MAP_POINTS = 2000
SIGNIFICANT_MW = 5.0
# Batas wilayah katalog (lat, lon) agar hasil geocoding tidak nyasar ke luar area data
GEOCODE_VIEWBOX = [(-12.0, 94.0), (7.0, 142.0)]
# Key widget yang dikembalikan ke default saat Reset Filter
FILTER_KEYS = ['filter_start_date', 'filter_end_date', 'filter_mmi', 'filter_profile', 'filter_mainshock',
               'filter_hide_suspect', 'search_query_input', 'radius_input']

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

def get_color(mmi): return MMI_COLORS.get(mmi, 'gray')

def balanced_sample(data, n, by='mmi_class', seed=42):
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

@st.cache_data(show_spinner=False, ttl=24 * 3600)
def geocode(query):
    geolocator = Nominatim(user_agent="klasifikasi-gempa-kmeans-viewer")
    location = geolocator.geocode(query, timeout=10, viewbox=GEOCODE_VIEWBOX, bounded=True)
    if location is None:
        return None
    return {'lat': location.latitude, 'lon': location.longitude, 'address': location.address}

# --- 2. Load Data & Models ---
@st.cache_data
def load_data():
    # Dihasilkan oleh: python -m gempa.train
    df = pd.read_parquet(PROCESSED_DIR / "earthquakes_with_cluster.parquet")
    df['mmi_class'] = df['mmi_class'].astype(str)
    return df

@st.cache_resource
def load_model():
    # Pipeline = MinMaxScaler + KMeans (fitur: mag, depth)
    pipeline = joblib.load(MODELS_DIR / "kmeans_pipeline.joblib")
    cluster_info = pd.read_csv(MODELS_DIR / "cluster_info.csv")
    cluster_info = cluster_info.dropna(subset=['cluster', 'label'])
    return pipeline, cluster_info

try:
    df = load_data()
    pipeline, cluster_info = load_model()
    id_to_label = dict(zip(cluster_info['cluster'], cluster_info['label']))
except Exception as e:
    st.error(f"Error loading system: {e}")
    st.stop()


# --- 3. LAYOUT BEGINS ---

# --- A. Header (MAIN PAGE) ---
st.title("🌋 Earthquake Risk Viewer")
st.caption("Katalog gempa BMKG 2008–2025. Warna = estimasi intensitas di episentrum (skala MMI) per gempa. "
           "Bukan prediksi gempa; rujuk informasi resmi BMKG.")

# Search Logic Processing
search_context = None

# --- Logic: Get Inputs from Session State (Defined at bottom) ---
search_query = st.session_state.get('search_query_input', '').strip()
search_radius = st.session_state.get('radius_input', 200)


if search_query:
    try:
        search_context = geocode(search_query)
        if search_context is None:
            st.warning("Lokasi tidak ditemukan.")
    except GeopyError:
        st.warning("Gagal koneksi geocoding.")

# Filter Data Global (Sidebar Filters + Search)
# --- Sidebar Filters ---
st.sidebar.title("🛠️ Tools & Filter")
if st.sidebar.button("🔄 Reset Filter"):
    for key in FILTER_KEYS:
        st.session_state.pop(key, None)
    st.rerun()

# min/max eksplisit: default Streamlit membatasi pilihan hanya ±10 tahun dari nilai awal
data_start, data_end = df['time'].min().date(), df['time'].max().date()
min_date = st.sidebar.date_input("Tanggal Awal", data_start, min_value=data_start, max_value=data_end, key='filter_start_date')
max_date = st.sidebar.date_input("Tanggal Akhir", data_end, min_value=data_start, max_value=data_end, key='filter_end_date')
selected_mmi = st.sidebar.multiselect("Kelas Intensitas (MMI)", MMI_CLASSES, default=MMI_CLASSES, key='filter_mmi')
selected_profiles = st.sidebar.multiselect("Profil Klaster (K-Means)", RISK_ORDER, default=RISK_ORDER, key='filter_profile',
                                           help="Label relatif antar-klaster dari magnitudo & kedalaman, bukan risiko absolut.")
mainshock_only = st.sidebar.checkbox("Hanya gempa utama (tanpa gempa susulan)", value=False, key='filter_mainshock',
                                     help="Hasil declustering Gardner–Knopoff (1974).")
hide_suspect = st.sidebar.checkbox("Sembunyikan event semu", value=True, key='filter_hide_suspect',
                                   help="Mw ≥ 5 yang tidak tercatat di katalog USGS (±60 s, ≤150 km), "
                                        "atau dengan < 10 fase dan celah azimut > 180°.")

if min_date > max_date:
    st.sidebar.warning("Tanggal Awal setelah Tanggal Akhir, rentang ditukar otomatis.")
    min_date, max_date = max_date, min_date

# Base Filter
mask = (
    (df['time'].dt.date >= min_date) &
    (df['time'].dt.date <= max_date) &
    (df['mmi_class'].isin(selected_mmi)) &
    (df['label'].isin(selected_profiles))
)
if mainshock_only:
    mask &= df['is_mainshock']
if hide_suspect:
    mask &= ~df['suspect_event']
filtered = df[mask]

# Apply Search Filter
map_center = [filtered['latitude'].mean(), filtered['longitude'].mean()] if not filtered.empty else MAP_DEFAULT_CENTER
map_zoom = 5

if search_context:
    map_center = [search_context['lat'], search_context['lon']]
    map_zoom = 8

    # Distance Filter (lingkaran radius sebenarnya, bukan kotak lat/lon)
    distance = haversine_km(search_context['lat'], search_context['lon'],
                            filtered['latitude'].to_numpy(), filtered['longitude'].to_numpy())
    filtered = filtered.assign(distance=distance)
    filtered = filtered[filtered['distance'] <= search_radius]


# --- B. Metrics ---
st.divider()
m1, m2, m3, m4 = st.columns(4)
with m1: st.metric("Total Kejadian", f"{len(filtered):,}")
with m2: st.metric(f"Gempa Mw ≥ {SIGNIFICANT_MW:.0f}", f"{(filtered['mw'] >= SIGNIFICANT_MW).sum():,}")
with m3: st.metric("Mw Tertinggi", f"{filtered['mw'].max():.1f}" if not filtered.empty else "0")
with m4: st.metric("Rata-rata Mw", f"{filtered['mw'].mean():.2f}" if not filtered.empty else "0")

# --- C. Intensity Legend (Top) ---
st.subheader("📋 Kelas Intensitas (MMI) & Rekomendasi")
info_cols = st.columns(4)
for col, mmi in zip(info_cols, MMI_CLASSES):
    clr = get_color(mmi)
    with col:
        st.markdown(f"""
        <div style="background-color: rgba(128, 128, 128, 0.08); border-top: 4px solid {clr}; padding: 15px; border-radius: 8px; height: 100%;">
            <h3 style="margin:0 0 5px 0; color: {clr};">MMI {mmi}</h3>
            <p style="font-size: 0.8rem; margin-bottom:5px;"><b>Efek:</b> {MMI_DESCRIPTIONS[mmi]}</p>
            <p style="font-size: 0.8rem;"><b>Saran:</b> {MMI_ADVICE[mmi]}</p>
        </div>
        """, unsafe_allow_html=True)
st.caption("Intensitas diestimasi dari Mw dan kedalaman dengan relasi Gutenberg–Richter (1956): "
           "I₀ = 1.5·Mw − 3.5·log₁₀(h) + 3. Ini perkiraan guncangan di atas pusat gempa, bukan di lokasi Anda.")

# --- D. Search Analysis Result Display (Main Area) ---
if search_context and not filtered.empty:
    strongest = filtered.loc[filtered['intensity'].idxmax()]
    biggest = filtered.loc[filtered['mw'].idxmax()]
    latest = filtered.loc[filtered['time'].idxmax()]
    years = max((pd.Timestamp(max_date) - pd.Timestamp(min_date)).days / 365.25, 1 / 365.25)
    significant_mainshocks = (filtered['is_mainshock'] & (filtered['mw'] >= SIGNIFICANT_MW)).sum()
    s_clr = get_color(strongest['mmi_class'])
    st.markdown(f"""
    <div style="margin: 20px 0; padding: 20px; border: 2px solid {s_clr}; border-radius: 10px; background-color: rgba(128, 128, 128, 0.05);">
        <h3 style="margin:0;">📍 Analisis Wilayah: {search_context['address']}</h3>
        <p>Ditemukan <b>{len(filtered):,}</b> gempa dalam radius {search_radius} km
           ({(filtered['mw'] >= SIGNIFICANT_MW).sum():,} di antaranya Mw ≥ {SIGNIFICANT_MW:.0f}).</p>
        <ul>
            <li><b>Gempa terbesar:</b> Mw {biggest['mw']:.1f}, kedalaman {biggest['depth']:.0f} km,
                {biggest['time']:%d %b %Y} ({biggest['location']})</li>
            <li><b>Gempa terbaru:</b> Mw {latest['mw']:.1f}, {latest['time']:%d %b %Y %H:%M} UTC</li>
            <li><b>Gempa utama Mw ≥ {SIGNIFICANT_MW:.0f}:</b> rata-rata {significant_mainshocks / years:.2f} per tahun
                ({significant_mainshocks} kejadian)</li>
        </ul>
        <h3 style="color: {s_clr}; margin-bottom: 0;">Intensitas tertinggi tercatat: MMI {strongest['mmi_class']}</h3>
        <p><b>Rekomendasi:</b> {MMI_ADVICE[strongest['mmi_class']]}</p>
    </div>
    """, unsafe_allow_html=True)
elif search_context:
    st.info(f"📍 {search_context['address']}: Tidak ada catatan gempa dalam radius {search_radius}km di dataset ini (dengan filter saat ini).")

# --- E. Map ---
m = folium.Map(location=map_center, zoom_start=map_zoom, prefer_canvas=True)

# Sampel seimbang per kelas intensitas agar kelas minoritas tetap terlihat dan peta tetap ringan
plot_df = balanced_sample(filtered, MAX_MAP_POINTS)
points = [
    [row.latitude, row.longitude, get_color(row.mmi_class),
     f"<b>Mw {row.mw:.1f}</b> ({row.mag_type} {row.mag:.1f})<br>Kedalaman: {row.depth:.0f} km<br>"
     f"Intensitas: MMI {row.mmi_class} (I₀ ≈ {row.intensity:.1f})<br>"
     f"Profil klaster: {row.label}<br>{'Gempa utama' if row.is_mainshock else 'Gempa susulan / bagian sekuens'}<br>"
     f"{row.time:%d %b %Y %H:%M} UTC"]
    for row in plot_df.itertuples()
]
FastMarkerCluster(points, callback=MARKER_CALLBACK).add_to(m)

if search_context:
    folium.Marker([search_context['lat'], search_context['lon']], icon=folium.Icon(color='red', icon='star')).add_to(m)
    folium.Circle([search_context['lat'], search_context['lon']], radius=search_radius*1000, color='red', fill=True, fill_opacity=0.1, popup=f"Radius: {search_radius}km").add_to(m)

st_folium(m, height=500, use_container_width=True, returned_objects=[])
if len(plot_df) < len(filtered):
    st.caption(f"Peta menampilkan {len(plot_df):,} dari {len(filtered):,} gempa (sampel seimbang per kelas intensitas).")

# --- F. Search Input (Moved Below Map) ---
st.markdown("### 🔍 Cari Lokasi")
# Use key to bind to session state, used at top of script
st.text_input("Nama Kota/Daerah", placeholder="Ketik nama kota, misal: Cianjur, Ambon...", key="search_query_input")
# Radius Slider below search bar
st.slider("Radius Pencarian (km)", min_value=10, max_value=500, value=200, step=10, key="radius_input", help="Geser untuk memperluas atau mempersempit area pencarian.")

# --- G. Significant Earthquakes in Area ---
if search_context and not filtered.empty:
    st.markdown(f"#### 📜 Gempa Terbesar dalam Radius {search_radius} km")
    top = filtered.nlargest(10, 'mw')[['time', 'mw', 'depth', 'mmi_class', 'is_mainshock', 'location', 'distance']]
    st.dataframe(
        top.assign(time=top['time'].dt.strftime('%Y-%m-%d %H:%M'), mw=top['mw'].round(1), distance=top['distance'].round(0))
           .rename(columns={'time': 'Waktu (UTC)', 'mw': 'Mw', 'depth': 'Kedalaman (km)', 'mmi_class': 'MMI',
                            'is_mainshock': 'Gempa utama', 'location': 'Lokasi', 'distance': 'Jarak (km)'}),
        hide_index=True, width='stretch',
    )

# --- H. Safety Guide (Bottom) ---
st.markdown("---")
st.subheader("🛡️ Panduan Keselamatan Gempa")
tab1, tab2, tab3 = st.tabs(["🛡️ Persiapan (Sebelum)", "🚨 Tindakan (Saat Gempa)", "⛑️ Pemulihan (Sesudah)"])
with tab1:
    for t in TIPS_SEBELUM_GEMPA: st.info(t)
with tab2:
    for t in TIPS_SAAT_GEMPA: st.warning(t)
with tab3:
    for t in TIPS_SETELAH_GEMPA: st.success(t)


# --- Sidebar Utilities ---
with st.sidebar:
    st.markdown("---")
    st.subheader("⚡ Estimasi Manual")

    p3 = st.number_input("Magnitudo (Mw)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
    p4 = st.number_input("Kedalaman (km)", min_value=0.0, max_value=800.0, value=10.0, step=10.0)

    if st.button("Hitung"):
        try:
            i0 = float(epicentral_intensity(p3, p4))
            mmi_res = mmi_class(i0)
            # Profil klaster: model dilatih dengan 2 fitur (mag, depth); pipeline menangani scaling
            profile = id_to_label[pipeline.predict(pd.DataFrame([[p3, p4]], columns=['mag', 'depth']))[0]]

            r_clr = get_color(mmi_res)
            st.markdown(f"""
            <div style="background-color: {r_clr}; padding: 10px; border-radius: 5px; text-align: center; color: white; margin-top: 10px;">
                <h4 style="margin:0;">MMI {mmi_res}</h4>
                <small>I₀ ≈ {i0:.1f}</small>
            </div>
            """, unsafe_allow_html=True)
            st.caption(MMI_ADVICE[mmi_res])
            st.caption(f"Profil klaster K-Means: **{profile}**")
        except Exception as e:
            st.error(f"Error: {e}")
