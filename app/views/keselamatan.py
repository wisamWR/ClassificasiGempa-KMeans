"""Panduan keselamatan gempa (konten statis) dan legenda intensitas MMI."""
import streamlit as st

from common import BMKG_URL, INARISK_URL, mmi_legend

TIPS_SEBELUM_GEMPA = [
    "🏠 Identifikasi zona aman di rumah: di bawah meja kokoh, jauh dari jendela.",
    "🎒 Siapkan tas siaga darurat berisi air minum, makanan kering, senter, obat pribadi.",
    "📍 Kenali rute evakuasi dan titik kumpul di lingkungan Anda.",
    "🔧 Pastikan lemari dan rak buku terikat ke dinding untuk mencegah roboh.",
    "📱 Aktifkan notifikasi peringatan dini BMKG di smartphone.",
]

TIPS_SAAT_GEMPA = [
    "🛑 Jangan panik. Tetap tenang dan bertindak cepat.",
    "🙇 Berlindung di bawah meja atau benda kokoh, lindungi kepala dan leher.",
    "🚪 Jauhi jendela, cermin, lemari, dan benda berat yang bisa jatuh.",
    "🏃 Jika di luar ruangan, jauhi gedung, tiang listrik, dan pohon besar.",
    "🚗 Jika sedang berkendara, hentikan kendaraan di tempat terbuka.",
    "🌊 Jika di pantai dan guncangan kuat atau lama, segera menjauh ke tempat tinggi tanpa menunggu peringatan tsunami.",
]

TIPS_SETELAH_GEMPA = [
    "🔍 Periksa diri sendiri dan orang sekitar. Berikan pertolongan pertama jika perlu.",
    "⚠️ Waspada gempa susulan. Jangan masuk gedung yang retak.",
    "📻 Dengarkan informasi resmi dari BMKG melalui radio atau aplikasi.",
    "🔥 Periksa instalasi gas dan listrik. Matikan jika ada kebocoran.",
    "📞 Hubungi keluarga untuk memberi kabar, hindari telepon berlebihan.",
]

st.title("🛡️ Panduan Keselamatan Gempa")
st.markdown(f"Informasi gempa terkini dan peringatan dini tsunami: **[BMKG]({BMKG_URL})** (aplikasi InfoBMKG). "
            f"Peta risiko bencana daerah Anda: **[InaRISK BNPB]({INARISK_URL})**.")

tab1, tab2, tab3 = st.tabs(["🛡️ Persiapan (Sebelum)", "🚨 Tindakan (Saat Gempa)", "⛑️ Pemulihan (Sesudah)"])
with tab1:
    for tip in TIPS_SEBELUM_GEMPA:
        st.info(tip)
with tab2:
    for tip in TIPS_SAAT_GEMPA:
        st.warning(tip)
with tab3:
    for tip in TIPS_SETELAH_GEMPA:
        st.success(tip)

st.subheader("📋 Kelas Intensitas (MMI) & Rekomendasi")
mmi_legend()
