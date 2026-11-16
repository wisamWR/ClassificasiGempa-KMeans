# 🌋 Zona Gempa Indonesia (Earthquake Clustering Viewer)

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
[![CI](https://github.com/wisamWR/ClassificasiGempa-KMeans/actions/workflows/ci.yml/badge.svg?branch=dev-wsm)](https://github.com/wisamWR/ClassificasiGempa-KMeans/actions/workflows/ci.yml)

### 🚀 [Buka aplikasi: Zona Gempa Indonesia](https://classificasigempa-kmeansgit-rcdqnzzovc9qubgl7ravzn.streamlit.app/)

Aplikasi analisis kegempaan Indonesia berbasis katalog BMKG 2008–2025 dan **Machine Learning (K-Means Clustering)**. K-Means dipakai untuk membagi wilayah menjadi **zona sumber gempa**, lalu tiap zona diberi statistik Gutenberg–Richter: b-value, periode ulang, dan peluang gempa M ≥ 5 / M ≥ 6. Pengguna bisa mengecek kotanya, menjelajah zona, dan membaca konteks sebuah gempa.

> ⚠️ Aplikasi ini **tidak memprediksi gempa** dan bukan peta bahaya resmi. Informasi gempa terkini dan peringatan dini: [BMKG](https://www.bmkg.go.id). Kajian risiko bencana: [InaRISK BNPB](https://inarisk.bnpb.go.id).

**Sumber data:** katalog gempa dan feed gempa terkini dari **BMKG** ([data.bmkg.go.id](https://data.bmkg.go.id/gempabumi/)); katalog pembanding [USGS ComCat](https://earthquake.usgs.gov/data/comcat/) (domain publik).

---

## 🌟 Halaman Aplikasi

Aplikasi multipage (`st.navigation`) yang bisa langsung dibuka di **[Streamlit Community Cloud](https://classificasigempa-kmeansgit-rcdqnzzovc9qubgl7ravzn.streamlit.app/)**. Waktu gempa ditampilkan dalam **WIB/WITA/WIT** (perkiraan dari koordinat), dan sidebar setiap halaman memuat disclaimer serta tautan ke BMKG dan InaRISK.

| Halaman | Isi |
| :-- | :-- |
| 📍 **Cek Lokasi** (halaman awal) | Cari kota atau klik peta → zona sumber gempa di bawah lokasi untuk tiap lapisan (di dalam zona atau jaraknya ke tepi zona), peluang gempa utama M ≥ 5 dan M ≥ 6 dalam 10/50 tahun di radius pilihan (25–300 km) beserta peluang dari gempa dangkal saja, jumlah gempa tercatat vs perkiraan model, riwayat & 5 gempa terbesar, intensitas tertinggi yang pernah tercatat beserta saran, dan linimasa gempa. Lokasi yang jauh dari zona dangkal mana pun (mis. Kalimantan Barat) diberi peringatan. |
| 🔴 **Gempa Terkini** | Gempa terbaru, 15 gempa M ≥ 5, dan 15 gempa dirasakan dari feed publik **BMKG** (data.bmkg.go.id, disimpan sementara 5 menit), lengkap dengan shakemap, peringatan bila berpotensi tsunami, perkiraan intensitas, zona sumber, dan seberapa sering gempa sebesar itu terjadi di zonanya. |
| 🧭 **Zona Seismik** | Peta poligon zona per lapisan kedalaman, diwarnai menurut peluang M ≥ 6/M ≥ 5, periode ulang, atau b-value; opsi heatmap. Klik zona untuk detail: statistik, grafik Gutenberg–Richter, gempa terbesar, dan proporsi jenis sesar di zona. |
| 🗺️ **Peta Gempa** | Seluruh gempa (marker cluster, warna = kelas MMI) dengan filter tanggal, Mw, lapisan kedalaman, kelas MMI, profil klaster, gempa utama, dan event semu, plus tombol reset filter. Popup berisi Mw, magnitudo asli, kedalaman, intensitas, zona, profil klaster, status gempa utama, dan waktu setempat. Maksimal 2.000 titik dengan sampel seimbang per kelas intensitas. |
| ⚡ **Kategorikan Gempa** | Masukkan lintang, bujur, kedalaman, dan magnitudo (Mw, M, MLv, ML, atau mb yang dikonversi ke Mw) → kelas intensitas MMI + saran, zona sumber, seberapa sering gempa sebesar itu terjadi di zona tersebut, dan profil klaster K-Means. |
| 🔁 **Gempa Susulan** | 20 deret gempa terbesar hasil declustering: peta **animasi** kemunculan gempa susulan per hari, grafik peluruhan dengan fit **Omori–Utsu** (maximum likelihood, Ogata 1983), dan magnitudo susulan terhadap waktu. Deret yang tidak meluruh seperti gempa susulan biasa (mis. swarm) ditandai. |
| 🐝 **Klaster & Swarm** | Klaster spatio-temporal **ST-DBSCAN** (20 km, 2 hari, ≥ 10 gempa) yang digolongkan menjadi swarm/doublet, deret gempa utama–susulan, atau potongan deret lain; peta, tabel, dan magnitudo terhadap waktu per klaster. |
| 📊 **Eksplorasi Data** | Distribusi magnitudo, grafik Gutenberg–Richter, deret waktu bulanan, Mc dan tingkat verifikasi USGS per tahun, penampang kedalaman (Jawa, Sumatra, Laut Maluku, Laut Banda), tampilan **3D** (pydeck) yang memperlihatkan lempeng menunjam, dan peta **jenis sesar** dari 3.864 mekanisme fokal (klasifikasi Frohlich 1992). |
| 📘 **Metodologi** | Pipeline, metrik model, perbandingan K-Means vs HDBSCAN, keterbatasan, dan referensi. |
| 🛡️ **Panduan Keselamatan** | Panduan sebelum, saat, dan sesudah gempa, serta legenda kelas intensitas MMI. |

**Cara menghitung peluang di Cek Lokasi:** laju tahunan tiap zona dibagi rata ke gempa utamanya, sehingga kontribusi zona = laju zona × (gempa zona dalam radius ÷ seluruh gempa zona). Peluang minimal satu kejadian dalam T tahun = 1 − exp(−laju · T).

---

## 🔬 Pipeline Data & Model

### 🧹 Katalog Bersih

Langkah `python -m gempa.clean` (dokumentasi lengkap di `notebooks/data_cleaning.ipynb`):

- **Homogenisasi magnitudo ke Mw** dengan relasi Taruna & Pratiwi (2021); pada pasangan Mw mekanisme fokal di katalog ini biasnya ≤ 0,02.
- **Magnitude of completeness (Mc)** per tahun (maximum curvature + 0,2): turun dari 4,7 (2008) ke 3,1 (2020–2022).
- **Declustering Gardner & Knopoff (1974)** untuk memisahkan gempa utama dan gempa susulan.
- **Flag kualitas:** kedalaman default dan lokasi kurang terikat.
- **Verifikasi silang dengan USGS ComCat:** gempa Mw ≥ 5 yang tidak punya pasangan di katalog USGS (±60 s, ≤ 150 km), atau yang hanya terekam < 10 fase dengan celah azimut > 180°, ditandai **event semu** (1.530 event). Tingkat kecocokan 95–99% pada tahun normal, tetapi hanya 16–50% pada 2019–2021: katalog BMKG periode itu memuat banyak event fiktif, terutama di Kalimantan/Laut Jawa dan beberapa menit setelah gempa besar di luar wilayah (mis. Selandia Baru, Fiji).
- **Intensitas per event:** I₀ = 1,5·Mw − 3,5·log₁₀(h) + 3 (Gutenberg–Richter 1956), dipetakan ke kelas MMI I-II / III-IV / V-VI / VII+.

### 🤖 Profil Klaster K-Means

Mengelompokkan data gempa (k = 4, dipilih dengan silhouette score) berdasarkan karakteristik fisik:

- **Magnitudo** (Kekuatan Gempa)
- **Kedalaman** (Depth)

Latitude/longitude sengaja tidak dipakai agar klaster tidak terbentuk karena lokasi geografis. Tiap klaster diberi label **Low / Moderate / High / Very High** dengan mengurutkan estimasi intensitas episentrum Gutenberg–Richter (1956) dari centroid-nya.

> ⚠️ Label bersifat relatif antar-klaster, bukan risiko absolut per gempa. Lebih dari seperempat event berlabel *High* berintensitas MMI I–II (tidak terasa), dan gempa besar kedalaman menengah (M7.0 / 150 km) berlabel *Moderate*. Karena itu aplikasi menampilkan label ini sebagai **profil klaster**, sedangkan warna dan rekomendasi memakai kelas MMI per gempa.

### 🧭 Zonasi Seismik

Langkah `python -m gempa.zoning` (dokumentasi di `notebooks/seismic_zoning.ipynb`) membagi wilayah menjadi zona sumber gempa, lalu menghitung statistik hazard tiap zona:

- **Data:** 25.773 gempa utama di atas Mc yang lolos verifikasi.
- **Lapisan kedalaman:** dangkal (< 70 km), menengah (70–300 km), dalam (> 300 km). Clustering dilakukan per lapisan pada koordinat Kartesius 3D (km), sehingga jarak antar-event adalah jarak sebenarnya.
- **K-Means per lapisan:** k = jumlah zona terbanyak yang setiap zonanya masih berisi ≥ 200 event (agar b-value stabil). Hasil: 22 zona dangkal, 14 menengah, 1 dalam. Silhouette tidak dipakai untuk memilih k karena nilainya terus turun dari k = 4.
- **Pembanding HDBSCAN:** di lapisan menengah HDBSCAN sepakat sedang dengan K-Means (ARI 0,45); di lapisan dangkal ia menggabungkan busur yang menerus dan menandai 22% event sebagai noise, sehingga kurang cocok untuk zonasi yang harus menutup seluruh wilayah.
- **Statistik per zona:** Mc, b-value (maximum likelihood Aki 1965, galat Shi & Bolt 1982), laju tahunan, periode ulang, dan probabilitas Poisson M ≥ 5 / M ≥ 6 dalam 10 dan 50 tahun. Laju hanya dihitung dari tahun yang katalognya sudah lengkap pada Mc zona.

- **Hasil:** b-value 0,85–1,42 (median 1,10); periode ulang gempa utama M ≥ 6 dari ±1,5 tahun (Sumatra Selatan, dangkal) sampai ±26 tahun (Flores, menengah).
- **Validasi:** jumlah gempa utama yang tercatat di 89% zona (M ≥ 5) dan 95% zona (M ≥ 6) berada dalam interval 95% Poisson dari prediksi Gutenberg–Richter.

> ⚠️ Probabilitas bersifat statistik katalog 17 tahun (gempa utama, model Poisson), bukan peta bahaya resmi seperti Peta Sumber dan Bahaya Gempa Indonesia (PuSGeN 2017).

### 🔴 Gempa Terkini (`gempa.bmkg`)

- Feed publik BMKG `autogempa`, `gempaterkini`, dan `gempadirasakan` (data.bmkg.go.id) diambil saat halaman dibuka dan disimpan sementara 5 menit, jauh di bawah batas BMKG 60 permintaan/menit. Ada tombol muat ulang; bila BMKG tidak terjangkau, aplikasi hanya menampilkan peringatan.
- Magnitudo BMKG dikonversi ke Mw seperti tipe M di katalog, lalu tiap gempa diberi perkiraan intensitas, zona sumber, dan selang waktu rata-rata gempa sebesar itu di zonanya.
- Sesuai ketentuan BMKG, BMKG dicantumkan sebagai sumber data di aplikasi.

### 🪨 Mekanisme Fokal (`gempa.focal`)

- Bidang nodal (strike/dip/rake) dari 3.905 solusi BMKG diubah menjadi sumbu tekanan (P), tarikan (T), dan netral (B).
- Jenis sesar menurut Frohlich (1992): plunge T > 50° = sesar naik, plunge P > 60° = sesar turun, plunge B > 60° = sesar geser, selain itu oblik. Kedua bidang nodal memberi jenis yang sama pada semua solusi.
- Gempa dangkal didominasi sesar naik (35%) dan sesar geser (27%); gempa dalam didominasi sesar turun dan oblik.

### 🔁 Gempa Susulan (`gempa.stats.omori_mle`)

- Deret gempa berasal dari declustering Gardner–Knopoff.
- Peluruhan dimodelkan dengan Omori–Utsu n(t) = K/(t + c)^p; parameternya ditaksir dengan maximum likelihood (Ogata 1983) pada gempa susulan ≥ Mc antara 0,1 dan 100 hari. Jendela Gardner–Knopoff yang lebih panjang ikut memuat kegempaan latar dan menekan nilai p.
- Deret khas memberi p ≈ 1 (mis. Laut Maluku 2019: 1,09; Laut Flores 2021: 0,99). Deret dengan p di luar 0,5–2 ditandai tidak meluruh seperti gempa susulan biasa.

### 🐝 Klaster Spatio-Temporal (`gempa.swarm`)

- ST-DBSCAN (Birant & Kut 2007) pada gempa ≥ Mc: dua gempa bertetangga bila jarak 3D ≤ 20 km **dan** selisih waktu ≤ 2 hari; klaster minimal 10 gempa.
- Jenis klaster: **swarm/doublet** (selisih Mw terbesar dan kedua < 0,5), **bagian deret gempa lain** (sebagian besar gempanya susulan dari gempa utama yang lebih besar di luar klaster, mis. sepanjang ruptur Palu 2018), atau **gempa utama–susulan**.
- Hasil: 125 klaster (34 swarm/doublet, 45 gempa utama–susulan, 46 bagian deret lain).

---

## 🛠️ Instalasi & Cara Pakai

Pastikan Anda memiliki [Python](https://www.python.org/) yang terinstall di komputer Anda.

1. **Clone Repository (atau download folder project ini):**

   ```bash
   git clone https://github.com/wisamWR/ClassificasiGempa-KMeans.git
   cd ClassificasiGempa-KMeans
   ```

2. **Buat Virtual Environment (Opsional tapi Direkomendasikan):**

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Mac/Linux
   source .venv/bin/activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Jalankan Aplikasi:**
   ```bash
   streamlit run app/streamlit_app.py
   ```
   Pencarian nama kota memakai layanan Nominatim (OpenStreetMap) dan halaman Gempa Terkini memakai feed BMKG, jadi keduanya butuh koneksi internet; titik juga bisa dipilih dengan klik peta.
   Aplikasi akan otomatis terbuka di browser Anda (biasanya di `http://localhost:8501`).

### Melatih Ulang Model

Logika training ada di paket `src/gempa/`. Model bisa dilatih ulang lewat CLI atau notebook; keduanya memakai fungsi yang sama, membaca `data/raw/`, dan menulis ulang artifact di `models/` serta `data/processed/`.

```bash
pip install -r requirements-dev.txt   # termasuk paket gempa (pip install -e .)

# Opsi 1: CLI (dari root project)
python -m gempa.reference  # opsional: unduh ulang katalog pembanding USGS (sudah ada di data/raw/)
python -m gempa.clean      # katalog bersih + laporan (data/processed/catalog_*.{parquet,json})
python -m gempa.train      # sudah menjalankan pembersihan yang sama sebelum training
python -m gempa.zoning     # zonasi seismik + statistik hazard (memakai katalog bersih)
python -m gempa.focal      # jenis sesar dari mekanisme fokal
python -m gempa.swarm      # klaster ST-DBSCAN & swarm (memakai data hasil gempa.train)

# Opsi 2: notebook (jalankan dari folder notebooks/, Restart & Run All)
jupyter notebook notebooks/earthquake_clustering.ipynb
jupyter notebook notebooks/seismic_zoning.ipynb
```

Versi scikit-learn saat training dicatat di `models/metadata.json` dan harus sama dengan versi di `requirements.txt`.

### Deploy ke Streamlit Community Cloud

Aplikasi sudah berjalan di **https://classificasigempa-kmeansgit-rcdqnzzovc9qubgl7ravzn.streamlit.app/**. Untuk deploy ulang atau membuat salinan sendiri: semua data dan model yang dibaca aplikasi sudah ada di repo (±43 MB), jadi tidak perlu langkah build.

1. Masuk ke [share.streamlit.io](https://share.streamlit.io) dengan akun GitHub, lalu **Create app → Deploy a public app from GitHub**.
2. Repository `wisamWR/ClassificasiGempa-KMeans`, branch `dev-wsm`, main file `app/streamlit_app.py`.
3. **Advanced settings → Python version: 3.11** (sama dengan versi saat model dilatih).
4. Deploy. Dependency dibaca dari `requirements.txt`; konfigurasi tampilan dari `.streamlit/config.toml`.

Halaman Cek Lokasi (pencarian nama kota) dan Gempa Terkini butuh akses internet dari server ke Nominatim dan BMKG.

### Menjalankan Test

```bash
ruff check src tests app   # lint
pytest                     # unit test + uji semua halaman aplikasi (AppTest, tanpa akses internet)
```

GitHub Actions (`.github/workflows/ci.yml`) menjalankan keduanya pada setiap push ke `dev-wsm` dan setiap pull request.

---

## 📦 Struktur File

```
app/
  streamlit_app.py                 🧭 Entry point & navigasi halaman
  common.py                        🧩 Loader data/model (cache), konstanta, CSS, komponen bersama
  views/                           📄 Halaman: cek_lokasi, gempa_terkini, zona, peta_gempa, kategorikan,
                                      susulan, swarm, eksplorasi, metodologi, keselamatan
src/gempa/
  data.py                          📥 Load katalog BMKG
  clean.py                         🧹 Pembersihan katalog & laporan kualitas (CLI)
  magnitude.py                     📏 Konversi magnitudo ke Mw
  completeness.py                  📉 Magnitude of completeness (Mc)
  decluster.py                     🔗 Declustering Gardner–Knopoff
  intensity.py                     📐 Intensitas episentrum & kelas MMI
  geo.py                           🌐 Jarak haversine
  reference.py                     🌍 Unduh & cocokkan katalog pembanding USGS (CLI)
  stats.py                         📈 b-value, laju tahunan, periode ulang, probabilitas Poisson, Omori–Utsu
  zoning.py                        🧭 Zonasi seismik & statistik hazard per zona (CLI)
  location.py                      📍 Zona di suatu titik & peluang gempa dalam radius
  bmkg.py                          🔴 Ambil & parse feed gempa publik BMKG
  focal.py                         🪨 Jenis sesar dari mekanisme fokal (CLI)
  swarm.py                         🐝 Klaster spatio-temporal ST-DBSCAN & deteksi swarm (CLI)
  timezones.py                     🕒 Perkiraan zona waktu WIB/WITA/WIT dari koordinat
  train.py                         🤖 Pencarian k, training, pelabelan, simpan artifact (CLI)
  paths.py                         📁 Lokasi file data & model
notebooks/
  data_cleaning.ipynb              🧹 Validasi konversi Mw, Mc, kualitas data, declustering
  earthquake_clustering.ipynb      🔬 Eksplorasi, visualisasi, dan evaluasi K-Means
  seismic_zoning.ipynb             🧭 Zonasi seismik, K-Means vs HDBSCAN, statistik Gutenberg–Richter
tests/                             🧪 Unit test + uji semua halaman aplikasi (AppTest)
  data/bmkg_*.json                 🔴 Contoh feed BMKG untuk test (tanpa akses internet)
data/
  raw/katalog_gempa_v2.tsv         💾 Katalog gempa BMKG (data mentah)
  raw/usgs_comcat_m4.0_2008_2025.csv  🌍 Katalog pembanding USGS ComCat M ≥ 4 (domain publik)
  processed/earthquakes_with_cluster.parquet  🗂️ Data gempa + label klaster (dipakai app)
  processed/catalog_report.json    🧾 Laporan pembersihan katalog
  processed/catalog_clean.parquet  🧹 Katalog bersih lengkap (tidak di-commit, buat dengan gempa.clean)
  processed/zone_events.parquet    🧭 Zona tiap event (eventID → zone_id)
  processed/focal_mechanisms.parquet  🪨 Bidang nodal, plunge P/T/B, dan jenis sesar (gempa.focal)
  processed/st_clusters.parquet    🐝 Ringkasan klaster ST-DBSCAN (+ st_cluster_events.parquet)
models/
  kmeans_pipeline.joblib           🤖 Pipeline MinMaxScaler + K-Means terlatih
  cluster_info.csv                 📊 Rata-rata fitur, severity score, dan label tiap klaster
  metadata.json                    🧾 Parameter, metrik, mapping label, versi library
  zoning_kmeans.joblib             🧭 Model K-Means zonasi per lapisan kedalaman
  zone_stats.csv                   📈 Statistik hazard per zona
  zones.geojson                    🗺️ Poligon zona + statistiknya
  zoning_metadata.json             🧾 Parameter zonasi, scan k, perbandingan K-Means vs HDBSCAN
docs/
  PENJELASAN_PERUBAHAN.md          📝 Catatan perubahan metodologi
  informasi_keselamatan_streamlit.txt  🛡️ Rancangan fitur informasi keselamatan
pyproject.toml                     📦 Konfigurasi paket gempa, pytest & ruff
.streamlit/config.toml             ⚙️ Konfigurasi Streamlit untuk deploy
.github/workflows/ci.yml           ✅ CI: ruff + pytest di setiap push/PR
requirements.txt                   📋 Dependency aplikasi
requirements-dev.txt               🧪 Dependency notebook, training, dan test
```

---

## 🤝 Kontribusi

Pull Request sangat diterima! Jika Anda menemukan bug atau punya ide fitur baru, silakan buka **Issue** atau kirim **Pull Request**.

---

Dirancang dengan ❤️ menggunakan **Streamlit** & **Python**.
