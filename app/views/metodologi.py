"""Metodologi: pipeline data, metrik model, keterbatasan, dan sitasi."""
import pandas as pd
import streamlit as st

from common import BMKG_URL, INARISK_URL, LAYER_LABELS, load_kmeans, load_report, load_zoning

report = load_report()
_, zones, _, zoning_meta = load_zoning()
_, _, kmeans_meta = load_kmeans()
dec = report["declustering"]
ver = report["verification"]

st.title("📘 Metodologi")
st.markdown(f"""
Aplikasi ini mengolah **{report['n_raw']:,} gempa** dari katalog BMKG (November 2008 – Oktober 2025).
Kode lengkap ada di paket `src/gempa/` dan notebook di folder `notebooks/`.

#### 1. Pembersihan katalog (`gempa.clean`)
- **Homogenisasi magnitudo ke Mw** dengan relasi Taruna & Pratiwi (2021) untuk tipe M, MLv, ML, dan mb.
- **Magnitude of completeness (Mc)** per tahun dengan metode maximum curvature + 0,2. Mc keseluruhan
  {report['completeness']['mc_overall']:.1f}; {report['completeness']['above_mc']:,} gempa berada di atas Mc tahunannya.
- **Verifikasi silang dengan USGS ComCat:** {ver['checked']:,} gempa Mw ≥ 5 dicocokkan (±{ver['tolerance']['seconds']} detik,
  ≤ {ver['tolerance']['km']} km). Gempa tanpa pasangan, atau yang terekam < 10 fase dengan celah azimut > 180°,
  ditandai **event semu** ({report['suspect_event']:,} event, sebagian besar tahun 2019–2021).
- **Declustering** Gardner & Knopoff (1974): {dec['mainshocks']:,} gempa utama dan {dec['dependent']:,} gempa susulan.
- **Intensitas per gempa:** I₀ = 1,5·Mw − 3,5·log₁₀(h) + 3 (Gutenberg & Richter 1956), dipetakan ke kelas MMI.

#### 2. Profil klaster K-Means (`gempa.train`)
K-Means (k = {kmeans_meta['params']['k']}) pada magnitudo dan kedalaman yang dinormalisasi (MinMaxScaler).
Label Low–Very High diurutkan dari estimasi intensitas centroid. Label ini **relatif antar-klaster** dan hanya
ditampilkan sebagai profil; warna dan rekomendasi di aplikasi memakai kelas MMI tiap gempa.

#### 3. Zonasi seismik (`gempa.zoning`)
- Data: {zoning_meta['n_zoning_events']:,} gempa utama ≥ Mc tahunan yang lolos verifikasi.
- K-Means terpisah untuk tiap lapisan kedalaman pada koordinat Kartesius 3D (km). k = jumlah zona terbanyak yang
  setiap zonanya masih berisi ≥ {zoning_meta['params']['min_zone_events']} gempa, agar b-value stabil.
- HDBSCAN dipakai sebagai pembanding.
- Per zona: Mc, b-value (maximum likelihood Aki 1965, galat Shi & Bolt 1982), laju tahunan dari tahun yang
  katalognya lengkap, periode ulang, dan peluang Poisson P = 1 − exp(−laju · T).

#### 4. Cek Lokasi (`gempa.location`)
Laju zona dibagi rata ke gempa utamanya; peluang di sekitar lokasi menjumlahkan kontribusi zona yang gempanya
berada dalam radius. Zona waktu WIB/WITA/WIT diperkirakan dari koordinat.

#### 5. Gempa Terkini (`gempa.bmkg`)
Feed publik BMKG (`autogempa`, `gempaterkini`, `gempadirasakan` di data.bmkg.go.id) diambil saat halaman dibuka dan
disimpan sementara 5 menit (batas BMKG 60 permintaan per menit). Magnitudo BMKG dikonversi ke Mw seperti tipe M di
katalog, lalu tiap gempa diberi perkiraan intensitas, zona sumber, dan selang waktu rata-rata gempa sebesar itu di
zonanya. Sumber data: BMKG.

#### 6. Mekanisme fokal (`gempa.focal`)
Bidang nodal (strike/dip/rake) dari katalog BMKG diubah menjadi sumbu tekanan (P), tarikan (T), dan netral (B).
Jenis sesar mengikuti Frohlich (1992): plunge T > 50° = sesar naik, plunge P > 60° = sesar turun, plunge B > 60° =
sesar geser, selain itu oblik. Kedua bidang nodal memberi jenis yang sama, karena sumbu P/T/B tidak bergantung pada
pilihan bidang.

#### 7. Gempa susulan (`gempa.stats.omori_mle`)
Deret gempa susulan berasal dari declustering Gardner–Knopoff. Peluruhannya dimodelkan dengan relasi Omori–Utsu
n(t) = K/(t + c)^p yang parameternya ditaksir dengan maximum likelihood (Ogata 1983), memakai gempa susulan ≥ Mc
antara 0,1 dan 100 hari setelah gempa utama.
""")

st.markdown("#### Metrik model")
left, right = st.columns(2)
with left:
    metrics = kmeans_meta["metrics"]
    st.markdown("**K-Means profil klaster** (magnitudo & kedalaman)")
    st.dataframe(pd.DataFrame({"Metrik": ["Silhouette", "Davies–Bouldin", "Calinski–Harabasz"],
                               "Nilai": [round(metrics["silhouette"], 3), round(metrics["davies_bouldin"], 3),
                                         round(metrics["calinski_harabasz"], 0)]}),
                 hide_index=True, width="stretch")
with right:
    comp = pd.DataFrame(zoning_meta["comparison"])
    st.markdown("**Zonasi: K-Means vs HDBSCAN** per lapisan")
    st.dataframe(pd.DataFrame({
        "Lapisan": comp["layer"].map(LAYER_LABELS),
        "Zona K-Means": comp["kmeans_k"],
        "Silhouette K-Means": comp["kmeans_silhouette"].round(3),
        "Klaster HDBSCAN": comp["hdbscan_clusters"],
        "Noise HDBSCAN": (comp["hdbscan_noise"] * 100).round(0).astype(int).astype(str) + "%",
        "ARI": comp["ari_on_hdbscan_core"].round(2),
    }), hide_index=True, width="stretch")
st.caption("HDBSCAN menggabungkan busur Sumatra–Nusa Tenggara menjadi satu klaster dan menyisakan sebagian gempa "
           "tanpa zona, sehingga K-Means (partisi lengkap) dipakai untuk zonasi.")

st.markdown(f"""
#### Keterbatasan
- Katalog hanya {report['completeness']['by_year'][-1]['year'] - report['completeness']['by_year'][0]['year'] + 1}
  tahun, sedangkan gempa besar bisa berulang dalam ratusan tahun. Peluang M ≥ 6 untuk zona yang jarang
  gempa besar sangat tidak pasti.
- Model Poisson menganggap gempa utama saling bebas dan lajunya konstan.
- Peluang dihitung untuk **episentrum**, bukan guncangan di lokasi. Guncangan juga bergantung pada kedalaman, jarak,
  dan kondisi tanah setempat, yang tidak dimodelkan di sini.
- Zona K-Means berbentuk cembung dan tidak mengikuti geometri sesar.
- Event semu kecil (Mw 4,5–5, 2020–2021) mungkin masih tersisa; gempa Lombok 2018 (M6,4 & M6,9) tidak ada di katalog.
- Ini **bukan** pengganti Peta Sumber dan Bahaya Gempa Indonesia (PuSGeN 2017), [BMKG]({BMKG_URL}), atau
  [InaRISK BNPB]({INARISK_URL}).

#### Referensi
- Aki, K. (1965). Maximum likelihood estimate of b in the formula log N = a − bM and its confidence limits. *Bull. Earthq. Res. Inst.*, 43, 237–239.
- Campello, R. J. G. B., Moulavi, D. & Sander, J. (2013). Density-based clustering based on hierarchical density estimates. *PAKDD 2013*, 160–172.
- Frohlich, C. (1992). Triangle diagrams: ternary graphs to display similarity and diversity of earthquake focal mechanisms. *Phys. Earth Planet. Inter.*, 75, 193–198.
- Gardner, J. K. & Knopoff, L. (1974). Is the sequence of earthquakes in Southern California, with aftershocks removed, Poissonian? *BSSA*, 64(5), 1363–1367.
- Gutenberg, B. & Richter, C. F. (1944). Frequency of earthquakes in California. *BSSA*, 34(4), 185–188.
- Gutenberg, B. & Richter, C. F. (1956). Earthquake magnitude, intensity, energy, and acceleration. *BSSA*, 46(2), 105–145.
- Ogata, Y. (1983). Estimation of the parameters in the modified Omori formula for aftershock frequencies by the maximum likelihood procedure. *J. Phys. Earth*, 31, 115–124.
- Pusgen (2017). *Peta Sumber dan Bahaya Gempa Indonesia Tahun 2017*.
- Shi, Y. & Bolt, B. A. (1982). The standard error of the magnitude-frequency b value. *BSSA*, 72(5), 1677–1687.
- Taruna, R. M. & Pratiwi, A. (2021). Konversi empiris magnitudo menggunakan data gempabumi 1922–2020 di Nusa Tenggara Barat. *Jurnal Sains Teknologi & Lingkungan*, 7(1), 1–12.
- U.S. Geological Survey (2025). *ANSS Comprehensive Earthquake Catalog (ComCat)*.
- Wiemer, S. & Wyss, M. (2000). Minimum magnitude of completeness in earthquake catalogs. *BSSA*, 90(4), 859–869.
""")
