# Penjelasan Perubahan Metodologi Clustering Gempa Bumi

> **Catatan status:** dokumen ini adalah catatan historis (18–20 Januari 2026).
> Formula severity bobot 70:30 di bagian 2 **sudah diganti** dengan rumus intensitas
> episentrum Gutenberg–Richter (1956), `I0 = 1.5·M − 3.5·log10(h) + 3.0`
> (commit `097e942` dan `52114f4`). Penjelasan rumus terbaru ada di notebook
> `notebooks/earthquake_clustering.ipynb` bagian "Penjelasan Logika Risk Clustering".
>
> Sejak pipeline training dipindah ke `src/gempa/` (November 2026), artifact
> `scaler.joblib`, `kmeans_model.joblib`, dan `cluster_remap.pkl` di bagian 5 diganti
> `models/kmeans_pipeline.joblib` + `models/metadata.json`, dan data olahan memakai
> `data/processed/earthquakes_with_cluster.parquet`.

---

## Update 18 Januari 2026

File: `earthquake_clustering.ipynb`

### Ringkasan Perubahan

**1. Fitur K-Means (Feature Selection)**

- Sebelum: `features = ['latitude', 'longitude', 'mag', 'depth']`
- Sesudah: `clustering_features = ['mag', 'depth']`

Lat/Lon tetap disimpan di `X_full` untuk visualisasi peta.

**2. Formula Severity Score** *(usang, lihat catatan status)*

- Sebelum: `Severity = mag_norm + (1 - depth_norm)` (bobot 50:50)
- Sesudah: `Severity = 0.7*mag_norm + 0.3*(1 - depth_norm)` (bobot 70:30)

### Mengapa Perubahan Ini Diperlukan?

#### Masalah 1: Bias Geografis

Hasil clustering lama menunjukkan:

| Cluster | Label | Mag | Depth | Lokasi |
|---|---|---|---|---|
| 0 | LOW | 3.27 | 32 km | Bali (-8.3°, 115°) |
| 2 | HIGH | 3.51 | 32 km | Maluku (-0.8°, 127°) |
| 3 | VERY HIGH | 3.65 | 35 km | Sumatra (0.2°, 99°) |

**Masalah:** ketiga cluster punya magnitude dan depth yang hampir identik.
Mereka dipisahkan karena lokasi geografis berbeda, bukan risiko.

**Solusi:** hapus Lat/Lon dari proses K-Means. Dengan begitu, clustering fokus
pada karakteristik fisik gempa (mag + depth), bukan lokasi.

**Reasoning:**

- K-Means menggunakan Euclidean distance.
- Jika Lat/Lon masuk, algoritma akan mengelompokkan berdasarkan "pulau mana".
- Padahal gempa M5.0 di Sumatra dan M5.0 di Sulawesi seharusnya punya tingkat
  risiko yang sama dari sisi karakteristik fisik.

#### Masalah 2: Bobot yang Tidak Proporsional *(usang)*

Formula lama `Severity = mag_norm + (1 - depth_norm)` memberi magnitude dan depth
bobot sama (50:50).

**Masalah:** dalam seismologi, magnitude adalah faktor utama kerusakan.
Magnitude adalah skala logaritmik:

- Perbedaan M3 vs M5 = 1000× energi
- Perbedaan M5 vs M7 = 1000× energi

Gempa dalam (deep) M7.0 tetap lebih berbahaya dari gempa dangkal (shallow) M3.0.
Formula 50:50 tidak menangkap ini.

**Solusi:** weighted formula `0.7*mag + 0.3*(1-depth)`; magnitude 70%, depth 30%.

### Mengapa Rasio 70:30 (Bukan 50:50)? *(usang)*

Angka 70:30 didasarkan pada prinsip fisika bagaimana energi gempa dilepaskan:

1. **Skala logaritmik vs linear**
   - Magnitude adalah skala logaritmik. Kenaikan 1 angka (M5 ke M6) = energi naik
     32 kali lipat. Kenaikan 2 angka (M5 ke M7) = energi naik 1000 kali lipat.
   - Depth (kedalaman) efeknya linear. Penurunan dampak guncangan akibat kedalaman
     terjadi secara lebih linear dibanding lonjakan energi magnitude.
2. **Kelemahan 50:50:** model akan menganggap gempa M4.0 (dangkal) sama bahayanya
   dengan gempa M6.0 (agak dalam). Faktanya salah; M6.0 (1000× lebih kuat) tetap
   jauh lebih berbahaya.
3. **Logika 70% (magnitude):** angka 0.7 dipilih agar energi menjadi faktor dominan.
   Ini menjamin gempa M7.0 hampir selalu punya skor lebih tinggi dari M5.0,
   terlepas dari kedalamannya.
4. **Logika 30% (depth):** sisa 0.3 berfungsi sebagai tie-breaker. Jika dua gempa
   punya magnitude sama (misal sama-sama M6.0), gempa yang dangkal skornya naik
   drastis karena faktor 30% ini.

### Catatan Tambahan: Mengapa `mag_type` Tidak Digunakan?

Kolom `mag_type` (tipe magnitudo: MLv, Mw, mb, dll) tidak dimasukkan ke clustering karena:

1. **Itu metode ukur, bukan indikator risiko.** `mag_type` hanya menunjukkan
   metode/alat yang digunakan stasiun gempa untuk mengukur kekuatan. Gempa M5.0
   tetaplah M5.0, tidak peduli diukur dengan MLv atau mb. Mengelompokkan
   berdasarkan alat ukur tidak relevan dengan analisis risiko kerusakan.
2. **Inkonsistensi data.** Dataset tercampur 50:50 antara tipe `MLv` dan `M`. Jika
   fitur ini dipakai, algoritma akan memisahkan gempa yang identik secara fisik
   hanya karena label teknis pencatatannya berbeda.
3. **Kendala teknis (numerik vs kategorikal).** K-Means membutuhkan input numerik.
   `mag_type` adalah data kategorikal; One-Hot Encoding akan menambah dimensi yang
   tidak perlu dan tidak berkorelasi dengan risiko.

Kesimpulan: fokus pada magnitude (sebagai proksi energi) tanpa mempedulikan tipe
pengukurannya adalah pendekatan yang paling efektif untuk kasus ini.

---

## Update 20 Januari 2026

File: `earthquake_clustering.ipynb`

Perubahan berfokus pada visualisasi yang lebih logis, stabilitas kode, dan kinerja.

### 1. Perbaikan Urutan Visualisasi (Risk Sorting)

- **Sebelum:** warna dan label cluster pada grafik batang/pie chart muncul acak
  atau berdasarkan jumlah anggota (misal: Cluster 2 dulu, baru Cluster 0), sehingga
  pembacaan risiko (Low ke High) tidak urut.
- **Sesudah:** visualisasi dipaksa urut secara logis: Low → Moderate → High → Very High,
  dengan mapping ID eksplisit (0 = Low, 1 = Moderate, 2 = High, 3 = Very High) dan
  warna konsisten (hijau → kuning → oranye → merah).

### 2. Perbaikan Logika & Stabilitas Kode

- **Fixed `KeyError: cluster`.** Penyebab: kode analisis dijalankan sebelum kolom
  `cluster` ditempelkan ke dataframe utama. Solusi: langkah eksplisit
  `kmeans.fit_predict` dan simpan ke `df['cluster']` sebelum analisis.
- **Fixed `NameError: cluster_summary`.** Penyebab: definisi tabel ringkasan terhapus
  saat update sebelumnya. Solusi: mengembalikan kode agregasi (groupby) dan
  perhitungan skor severity.

### 3. Optimasi Kinerja

- **Sebelum:** proses "Testing k values" (elbow method) sangat lambat.
- **Penyebab:** parameter `n_init=20` (20× percobaan untuk setiap nilai k).
- **Sesudah:** `n_init` dikurangi menjadi 3 untuk tahap pencarian k.
- **Hasil:** proses 6–7× lebih cepat tanpa mengubah pemilihan k terbaik.

### 4. Prediksi Data Baru (Demo)

Fitur baru di akhir notebook: model bisa langsung dites.
Contoh input Magnitude 5.5, Depth 30 km → output "VERY HIGH RISK".

### 5. Penjelasan Artifacts / File Output

| File | Peran |
|---|---|
| `models/scaler.joblib` | "Penggaris" (normalisasi data 0–1) |
| `models/kmeans_model.joblib` | "Otak" (model clustering) |
| `models/cluster_remap.pkl` | "Penerjemah" (ID klaster → label risiko) |
| `models/cluster_info.csv` | "Legenda" (statistik rata-rata tiap cluster) |
| `data/processed/earthquakes_with_cluster.csv` | "Data peta" (lat/lon + label cluster) |

### Perubahan Teknis: Menghitung Centroid (groupby vs inverse_transform)

Mengapa `centroids = scaler.inverse_transform(kmeans.cluster_centers_)` diganti?

**Masalah:** `inverse_transform` hanya mengembalikan kolom yang dipakai saat training.
Karena Latitude & Longitude sudah dihapus dari training, centroid yang dihasilkan
tidak punya koordinat, sehingga lokasi pusat cluster tidak diketahui.

**Solusi:** hitung rata-rata manual dari dataframe asli yang masih punya semua kolom.

```python
cluster_summary = df.groupby('cluster').agg({
    'mag': 'mean',
    'depth': 'mean',
    'latitude': 'mean',   # bisa dihitung kembali
    'longitude': 'mean'   # bisa dihitung kembali
})
```

**Keuntungan:**

1. **Lengkap:** tetap bisa melihat rata-rata lokasi geografis tiap cluster untuk
   reporting, walaupun model tidak memakai lokasi saat belajar.
2. **Akurat:** nilai rata-rata dihitung dari data aktual anggota cluster, bukan
   estimasi model.
