import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from datetime import datetime
from pathlib import Path
from folium.plugins import FastMarkerCluster
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.exc import GeopyError
from sklearn.pipeline import Pipeline

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
        .risk-card {
            border-radius: 8px;
            padding: 15px;
            color: white;
            margin-bottom: 10px;
            height: 100%;
        }
        .safety-box {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
            border-left: 5px solid #ff4b4b;
            color: #31333F;
        }
        .safety-header {
            font-size: 1.5rem;
            font-weight: bold;
            margin-bottom: 10px;
            color: #0e1117;
        }
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

RISK_ORDER = ['Low', 'Moderate', 'High', 'Very High']
RISK_COLORS = {
    'Low': '#28a745',       # Green
    'Moderate': '#ffc107',  # Yellow
    'High': '#fd7e14',      # Orange
    'Very High': '#dc3545'  # Red
}
RISK_SHORT = {'Low': 'Low', 'Moderate': 'Mod', 'High': 'High', 'Very High': 'Very'}

RISK_DETAILS = {
    'Low': {'impact': 'Guncangan lemah, dirasakan sedikit orang.', 'advice': 'Tetap tenang, rutinitas normal.', 'bg': 'rgba(40, 167, 69, 0.2)'},
    'Moderate': {'impact': 'Barang bergoyang, potensi kerusakan ringan.', 'advice': 'Waspada benda jatuh, lindungi kepala.', 'bg': 'rgba(255, 193, 7, 0.2)'},
    'High': {'impact': 'Kerusakan struktur bangunan kurang kokoh.', 'advice': 'Jauhi kaca/jendela, berlindung di bawah meja.', 'bg': 'rgba(253, 126, 20, 0.2)'},
    'Very High': {'impact': 'Kerusakan masif, potensi rubuh total.', 'advice': 'SEGERA EVAKUASI ke area terbuka. Hindari gedung.', 'bg': 'rgba(220, 53, 69, 0.2)'}
}

MAP_DEFAULT_CENTER = [-2.5, 118.0]
MAX_MAP_POINTS = 2000
# Batas wilayah katalog (lat, lon) agar hasil geocoding tidak nyasar ke luar area data
GEOCODE_VIEWBOX = [(-12.0, 94.0), (7.0, 142.0)]
# Key widget yang dikembalikan ke default saat Reset Filter
FILTER_KEYS = ['filter_start_date', 'filter_end_date', 'filter_risks', 'search_query_input', 'radius_input']

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

def get_color(label): return RISK_COLORS.get(label, 'gray')

def haversine_km(lat1, lon1, lat2, lon2):
    """Jarak great-circle (km) dari satu titik ke array titik."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))

def balanced_sample(data, n, by='label', seed=42):
    """Water filling: kuota dibagi rata per label, sisa kuota label kecil dialihkan ke label lain."""
    if len(data) <= n:
        return data
    counts = data[by].value_counts().sort_values()
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
# Path relatif terhadap root project, jadi app bisa dijalankan dari folder mana pun
ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

@st.cache_data
def load_data():
    df = pd.read_csv(PROCESSED_DIR / "earthquakes_with_cluster.csv", low_memory=False)
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    return df

@st.cache_resource
def load_model():
    try:
        pipeline = joblib.load(MODELS_DIR / "kmeans_pipeline.joblib")
    except FileNotFoundError:
        scaler = joblib.load(MODELS_DIR / "scaler.joblib")
        kmeans = joblib.load(MODELS_DIR / "kmeans_model.joblib")
        pipeline = Pipeline([('scaler', scaler), ('kmeans', kmeans)])

    cluster_info = pd.read_csv(MODELS_DIR / "cluster_info.csv")
    cluster_info = cluster_info.dropna(subset=['cluster', 'label'])
    return pipeline, cluster_info

try:
    df = load_data()
    pipeline, cluster_info = load_model()
    # Init mappings
    cluster_info['label'] = pd.Categorical(cluster_info['label'], categories=RISK_ORDER, ordered=True)
    cluster_info_sorted = cluster_info.sort_values('label')
    id_to_label = dict(zip(cluster_info['cluster'], cluster_info['label']))
    df['label'] = df['cluster'].map(id_to_label)
    mag_col = 'mag' if 'mag' in df.columns else 'magnitude'
except Exception as e:
    st.error(f"Error loading system: {e}")
    st.stop()


# --- 3. LAYOUT BEGINS ---

# --- A. Header (MAIN PAGE) ---
st.title("🌋 Earthquake Risk Viewer")

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
selected_risks = st.sidebar.multiselect("Filter Risiko", RISK_ORDER, default=RISK_ORDER, key='filter_risks')

if min_date > max_date:
    st.sidebar.warning("Tanggal Awal setelah Tanggal Akhir, rentang ditukar otomatis.")
    min_date, max_date = max_date, min_date

# Base Filter
filtered = df[
    (df['time'].dt.date >= min_date) &
    (df['time'].dt.date <= max_date) &
    (df['label'].isin(selected_risks))
]

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

    # Enrich Context
    if not filtered.empty:
        dom_label = filtered['label'].mode()[0]
        search_context['dominant_risk'] = dom_label
        search_context['count'] = len(filtered)
        search_context['advice'] = RISK_DETAILS[dom_label]['advice']


# --- B. Metrics ---
st.divider()
m1, m2, m3 = st.columns(3)
with m1: st.metric("Total Kejadian", f"{len(filtered):,}")
with m2: st.metric("Rata-rata Magnitudo", f"{filtered[mag_col].mean():.2f}" if not filtered.empty else "0")
with m3: st.metric("Magnitudo Tertinggi", f"{filtered[mag_col].max():.2f}" if not filtered.empty else "0")

# --- C. Risk Legend (Top) ---
st.subheader("📋 Informasi Kategori & Rekomendasi")
info_cols = st.columns(4)
for i, (_, row) in enumerate(cluster_info_sorted.iterrows()):
    lbl = row['label']
    clr = get_color(lbl)
    det = RISK_DETAILS.get(lbl, {})
    with info_cols[i]:
        st.markdown(f"""
        <div style="background-color: rgba(128, 128, 128, 0.08); border-top: 4px solid {clr}; padding: 15px; border-radius: 8px; height: 100%;">
            <h3 style="margin:0 0 5px 0; color: {clr};">{lbl}</h3>
            <p style="font-size: 0.8rem; margin-bottom:5px;"><b>Efek:</b> {det.get('impact')}</p>
            <p style="font-size: 0.8rem;"><b>Saran:</b> {det.get('advice')}</p>
        </div>
        """, unsafe_allow_html=True)

# --- D. Search Analysis Result Display (Main Area) ---
if search_context and 'dominant_risk' in search_context:
    r_lbl = search_context['dominant_risk']
    r_clr = get_color(r_lbl)
    st.markdown(f"""
    <div style="margin: 20px 0; padding: 20px; border: 2px solid {r_clr}; border-radius: 10px; background-color: rgba(128, 128, 128, 0.05);">
        <h3 style="margin:0;">📍 Analisis Wilayah: {search_context['address']}</h3>
        <p>Ditemukan <b>{search_context['count']}</b> gempa dalam radius {search_radius}km.</p>
        <h2 style="color: {r_clr};">Status: {r_lbl}</h2>
        <p><b>Rekomendasi Utama:</b> {search_context['advice']}</p>
    </div>
    """, unsafe_allow_html=True)
elif search_context:
    st.info(f"📍 {search_context['address']}: Tidak ada catatan gempa signifikan dalam radius {search_radius}km di dataset ini.")

# --- E. Map ---
m = folium.Map(location=map_center, zoom_start=map_zoom, prefer_canvas=True)

# Sampel seimbang per label agar label minoritas tetap terlihat dan peta tetap ringan
plot_df = balanced_sample(filtered, MAX_MAP_POINTS)
points = [
    [row.latitude, row.longitude, get_color(row.label),
     f"Risk: {row.label}<br>Mag: {getattr(row, mag_col):.1f}<br>Depth: {row.depth:.0f} km"]
    for row in plot_df.itertuples()
]
FastMarkerCluster(points, callback=MARKER_CALLBACK).add_to(m)

if search_context:
    folium.Marker([search_context['lat'], search_context['lon']], icon=folium.Icon(color='red', icon='star')).add_to(m)
    folium.Circle([search_context['lat'], search_context['lon']], radius=search_radius*1000, color='red', fill=True, fill_opacity=0.1, popup=f"Radius: {search_radius}km").add_to(m)

st_folium(m, height=500, use_container_width=True, returned_objects=[])
if len(plot_df) < len(filtered):
    st.caption(f"Peta menampilkan {len(plot_df):,} dari {len(filtered):,} gempa (sampel seimbang per label risiko).")

# --- F. Search Input (Moved Below Map) ---
st.markdown("### 🔍 Cari Lokasi")
c_search, c_date = st.columns([3, 1])

with c_search:
    # Use key to bind to session state, used at top of script
    st.text_input("Nama Kota/Daerah", placeholder="Ketik nama kota, misal: Cianjur, Ambon...", key="search_query_input")
    # Radius Slider below search bar
    st.slider("Radius Pencarian (km)", min_value=10, max_value=500, value=200, step=10, key="radius_input", help="Geser untuk memperluas atau mempersempit area pencarian.")

with c_date:
    now = datetime.now()
    current_month_name = now.strftime("%B")
    season = "Musim Hujan" if now.month in [10, 11, 12, 1, 2, 3] else "Musim Kemarau"
    st.info(f"🗓️ **{current_month_name} {now.year}**\n\n{season}")


# --- F. SAFETY ANALYSIS SECTION (Bottom) ---
st.markdown("---")
st.subheader("🛡️ Analisis Keselamatan: Perspektif Musiman")

if search_context:
    # 1. Historical Analysis for Current Month (data sudah difilter radius)
    df_hist_loc = filtered

    if not df_hist_loc.empty:
        df_month = df_hist_loc[df_hist_loc['time'].dt.month == now.month]

        hist_total = len(df_month)
        hist_high = len(df_month[df_month['label'].isin(['High', 'Very High'])])
        hist_pct = (hist_high / hist_total * 100) if hist_total > 0 else 0

        # Risk Logic with Explained Triggers
        if hist_total == 0:
            s_risk, s_color = "DATA TIDAK CUKUP", "gray"
            s_msg = "Belum ada data historis signifikan di bulan ini."
        elif hist_pct > 50:
            s_risk, s_color = f"RISIKO TINGGI ({current_month_name})", "red"
            s_msg = f"Bulan {current_month_name} memiliki riwayat gempa besar/berbahaya di wilayah ini.\n\n**(Pemicu: >50% gempa tercatat berstatus High/Very High)**"
        elif df_month[mag_col].max() >= 6.0:
            s_risk, s_color = f"RISIKO TINGGI ({current_month_name})", "red"
            s_msg = f"Waspada gempa megathrust atau gempa kuat di musim ini.\n\n**(Pemicu: Pernah terjadi gempa Mag ≥ 6.0 di bulan ini)**"
        elif hist_pct > 25:
            s_risk, s_color = f"WASPADA ({current_month_name})", "orange"
            s_msg = f"Aktivitas seismik cukup aktif di bulan {current_month_name}. Tetap waspada.\n\n**(Pemicu: 25-50% gempa berstatus High Risk)**"
        else:
            s_risk, s_color = f"RELATIF AMAN ({current_month_name})", "green"
            s_msg = f"Secara historis, bulan {current_month_name} cenderung minim gempa besar.\n\n**(Pemicu: Mayoritas gempa Low/Moderate Risk)**"

        # Calculate percentages for all clusters
        cluster_stats = {}
        for r in RISK_ORDER:
            c_count = len(df_month[df_month['label'] == r])
            c_pct = (c_count / hist_total * 100) if hist_total > 0 else 0
            cluster_stats[r] = c_pct

        # Display Layout: 50-50 Split for Status & Stats within Equal Containers
        c1, c2 = st.columns(2)

        # Determine Dominant Cluster for Display
        dom_cluster = df_month['label'].mode()[0] if not df_month.empty else '-'

        # FIXED HEIGHT CONFIG
        CARD_HEIGHT = 350

        with c1:
            st.markdown(f"""
            <div style="background-color:{s_color}; padding:20px; border-radius:10px; color:white; text-align:center; height: {CARD_HEIGHT}px; display: flex; flex-direction: column; justify-content: center;">
                <h4 style="margin:0;">Status Musiman</h4>
                <h2 style="margin:10px 0;">{s_risk}</h2>
                <p style="margin:0;">{s_msg}</p>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            with st.container(height=CARD_HEIGHT, border=True):
                # Header: Centered, No Icon
                st.markdown(f"<h4 style='text-align: center; margin-top: 0;'>Statistik {current_month_name} (Histori)</h4>", unsafe_allow_html=True)

                # Section 1: Key Metrics (Data is already from df_month)
                m1, m2, m3 = st.columns(3)
                with m1: st.metric("Total Kejadian", hist_total)
                with m2: st.metric("Max Magnitude", f"{df_month[mag_col].max():.1f}" if not df_month.empty else "0")
                with m3: st.metric("Dominant Risk", dom_cluster)

                st.divider()

                # Section 2: Risk Percentage Breakdown
                st.caption("Persentase per Label Risiko")
                for col, r in zip(st.columns(4), RISK_ORDER):
                    with col:
                        st.markdown(f"<span style='color:{get_color(r)}; font-weight:bold;'>{RISK_SHORT[r]}</span><br><b>{cluster_stats[r]:.0f}%</b>", unsafe_allow_html=True)

        # Full Width Guide Below
        st.markdown("### 💡 Panduan Keselamatan Lengkap")
        with st.expander("Buka Panduan Langkah demi Langkah", expanded=True):
            tab1, tab2, tab3 = st.tabs(["🛡️ Persiapan (Sebelum)", "🚨 Tindakan (Saat Gempa)", "⛑️ Pemulihan (Sesudah)"])
            with tab1:
                for t in TIPS_SEBELUM_GEMPA: st.info(t)
            with tab2:
                for t in TIPS_SAAT_GEMPA: st.warning(t)
            with tab3:
                for t in TIPS_SETELAH_GEMPA: st.success(t)
    else:
        st.info("Tidak ada data gempa dalam radius ini untuk dianalisis. Perbesar radius atau ubah filter.")
else:
    st.info("ℹ️ **Cari lokasi** di bagian atas untuk melihat Analisis Keselamatan Musiman & Panduan Evakuasi.")


# --- Sidebar Utilities ---
with st.sidebar:
    st.markdown("---")
    st.subheader("⚡ Prediksi Manual")

    p3 = st.number_input("Magnitudo", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
    p4 = st.number_input("Kedalaman (km)", min_value=0.0, max_value=800.0, value=10.0, step=10.0)

    if st.button("Hitung Risiko"):
        try:
            # Model dilatih dengan 2 fitur (mag, depth); pipeline menangani scaling
            arr = pd.DataFrame([[p3, p4]], columns=['mag', 'depth'])
            res = pipeline.predict(arr)[0]
            lbl_res = id_to_label[res]

            # Result Display
            r_clr = get_color(lbl_res)
            st.markdown(f"""
            <div style="background-color: {r_clr}; padding: 10px; border-radius: 5px; text-align: center; color: white; margin-top: 10px;">
                <h4 style="margin:0;">{lbl_res}</h4>
            </div>
            """, unsafe_allow_html=True)
            st.caption(RISK_DETAILS[lbl_res]['advice'])
        except Exception as e:
            st.error(f"Error: {e}")
