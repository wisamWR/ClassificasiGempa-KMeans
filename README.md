# 🌋 Earthquake Clustering Viewer

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

Visualisasi interaktif dan analisis persebaran gempa bumi menggunakan **Machine Learning (K-Means Clustering)**. Aplikasi ini membantu mengidentifikasi pola kegempaan, tingkat risiko, dan karakteristik setiap klaster gempa di wilayah Indonesia dan sekitarnya.

---

## 👥 Anggota Kelompok 2

| No  | Nama                      | NIM            |
| :-- | :------------------------ | :------------- |
| 1.  | **Adam Haritsa Thahara**  | A11.2024.15556 |
| 2.  | **Angela Echa Naresti**   | A11.2024.15791 |
| 3.  | **Farros Rifantiarno R.** | A11.2024.15694 |
| 4.  | **Nadjwa Salsabila W.**   | A11.2024.15670 |
| 5.  | **Mohammad Wisam W.**     | A11.2024.15739 |

---

## 🌟 Fitur Utama

### 1. 🗺️ Peta Interaktif

Visualisasi titik gempa historis (katalog BMKG, November 2008 – Oktober 2025) di atas peta interaktif. Warna marker menunjukkan **estimasi intensitas di episentrum (skala MMI)** tiap gempa.

- **Marker Clustering:** Titik gempa dikelompokkan saat zoom-out agar peta tetap rapi.
- **Detail Popup:** Mw, magnitudo asli, kedalaman, kelas MMI, profil klaster K-Means, status gempa utama/susulan, dan waktu (UTC).
- **Balanced Sampling (Water Filling):** Peta menampilkan maksimal 2.000 titik. Kuota dibagi rata per kelas intensitas; sisa kuota dari kelas yang datanya sedikit dialihkan ke kelas lain.

### 2. 🧹 Katalog Bersih

Langkah `python -m gempa.clean` (dokumentasi lengkap di `notebooks/data_cleaning.ipynb`):

- **Homogenisasi magnitudo ke Mw** dengan relasi Taruna & Pratiwi (2021); pada pasangan Mw mekanisme fokal di katalog ini biasnya ≤ 0,02.
- **Magnitude of completeness (Mc)** per tahun (maximum curvature + 0,2): turun dari ±4,5 (2008) ke ±3,1 (2020–2022).
- **Declustering Gardner & Knopoff (1974)** untuk memisahkan gempa utama dan gempa susulan.
- **Flag kualitas:** kedalaman default, lokasi kurang terikat, dan event semu (Mw ≥ 5 dengan < 10 fase dan celah azimut > 180°).
- **Intensitas per event:** I₀ = 1,5·Mw − 3,5·log₁₀(h) + 3 (Gutenberg–Richter 1956), dipetakan ke kelas MMI I-II / III-IV / V-VI / VII+.

### 3. 🤖 K-Means Clustering

Mengelompokkan data gempa (k = 4, dipilih dengan silhouette score) berdasarkan karakteristik fisik:

- **Magnitudo** (Kekuatan Gempa)
- **Kedalaman** (Depth)

Latitude/longitude sengaja tidak dipakai agar klaster tidak terbentuk karena lokasi geografis. Tiap klaster diberi label **Low / Moderate / High / Very High** dengan mengurutkan estimasi intensitas episentrum Gutenberg–Richter (1956) dari centroid-nya.

> ⚠️ Label bersifat relatif antar-klaster, bukan risiko absolut per gempa. Lebih dari seperempat event berlabel *High* berintensitas MMI I–II (tidak terasa), dan gempa besar kedalaman menengah (M7.0 / 150 km) berlabel *Moderate*. Karena itu aplikasi menampilkan label ini sebagai **profil klaster**, sedangkan warna dan rekomendasi memakai kelas MMI per gempa.

### 4. 🔍 Filter & Pencarian Lokasi

- **Rentang Tanggal**, **Kelas Intensitas (MMI)**, dan **Profil Klaster (K-Means)**.
- **Hanya gempa utama:** sembunyikan gempa susulan hasil declustering.
- **Sembunyikan event semu** (aktif secara default).
- **Cari Lokasi:** ketik nama kota/daerah dan atur radius (10–500 km). Ringkasan wilayah berisi gempa terbesar, gempa terbaru, laju gempa utama Mw ≥ 5 per tahun, intensitas tertinggi yang pernah tercatat, dan tabel 10 gempa terbesar.

### 5. ⚡ Estimasi Manual

Masukkan **Mw** dan **kedalaman** gempa untuk mendapatkan estimasi kelas intensitas MMI beserta profil klaster K-Means-nya.

### 6. 🛡️ Panduan Keselamatan

Panduan sebelum, saat, dan sesudah gempa.

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
   Aplikasi akan otomatis terbuka di browser Anda (biasanya di `http://localhost:8501`).

### Melatih Ulang Model

Logika training ada di paket `src/gempa/`. Model bisa dilatih ulang lewat CLI atau notebook; keduanya memakai fungsi yang sama, membaca `data/raw/`, dan menulis ulang artifact di `models/` serta `data/processed/`.

```bash
pip install -r requirements-dev.txt   # termasuk paket gempa (pip install -e .)

# Opsi 1: CLI (dari root project)
python -m gempa.clean   # opsional: katalog bersih + laporan (data/processed/catalog_*.{parquet,json})
python -m gempa.train   # sudah menjalankan pembersihan yang sama sebelum training

# Opsi 2: notebook (jalankan dari folder notebooks/, Restart & Run All)
jupyter notebook notebooks/earthquake_clustering.ipynb
```

Versi scikit-learn saat training dicatat di `models/metadata.json` dan harus sama dengan versi di `requirements.txt`.

### Menjalankan Test

```bash
pytest
```

---

## 📦 Struktur File

```
app/
  streamlit_app.py                 🧠 Kode utama aplikasi (UI & logic)
src/gempa/
  data.py                          📥 Load katalog BMKG
  clean.py                         🧹 Pembersihan katalog & laporan kualitas (CLI)
  magnitude.py                     📏 Konversi magnitudo ke Mw
  completeness.py                  📉 Magnitude of completeness (Mc)
  decluster.py                     🔗 Declustering Gardner–Knopoff
  intensity.py                     📐 Intensitas episentrum & kelas MMI
  geo.py                           🌐 Jarak haversine
  train.py                         🤖 Pencarian k, training, pelabelan, simpan artifact (CLI)
  paths.py                         📁 Lokasi file data & model
notebooks/
  data_cleaning.ipynb              🧹 Validasi konversi Mw, Mc, kualitas data, declustering
  earthquake_clustering.ipynb      🔬 Eksplorasi, visualisasi, dan evaluasi K-Means
tests/                             🧪 Unit test (pytest)
data/
  raw/katalog_gempa_v2.tsv         💾 Katalog gempa BMKG (data mentah)
  processed/earthquakes_with_cluster.parquet  🗂️ Data gempa + label klaster (dipakai app)
  processed/catalog_report.json    🧾 Laporan pembersihan katalog
  processed/catalog_clean.parquet  🧹 Katalog bersih lengkap (tidak di-commit, buat dengan gempa.clean)
models/
  kmeans_pipeline.joblib           🤖 Pipeline MinMaxScaler + K-Means terlatih
  cluster_info.csv                 📊 Rata-rata fitur, severity score, dan label tiap klaster
  metadata.json                    🧾 Parameter, metrik, mapping label, versi library
docs/
  PENJELASAN_PERUBAHAN.md          📝 Catatan perubahan metodologi
  informasi_keselamatan_streamlit.txt  🛡️ Rancangan fitur informasi keselamatan
pyproject.toml                     📦 Konfigurasi paket gempa & pytest
requirements.txt                   📋 Dependency aplikasi
requirements-dev.txt               🧪 Dependency notebook, training, dan test
```

---

## 🤝 Kontribusi

Pull Request sangat diterima! Jika Anda menemukan bug atau punya ide fitur baru, silakan buka **Issue** atau kirim **Pull Request**.

---

Dirancang dengan ❤️ menggunakan **Streamlit** & **Python**.
