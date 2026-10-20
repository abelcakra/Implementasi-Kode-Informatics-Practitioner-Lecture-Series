# Implementasi Kode — Informatics Practitioner Lecture Series 2026

Kumpulan implementasi kode dari materi **Informatics Practitioner Lecture Series 2026**, dikemas dalam satu notebook Jupyter/Colab. Setiap sel notebook mewakili satu sesi dan dapat dijalankan secara mandiri.

Fokus repositori ini bukan sekadar menyalin teori, tetapi menerjemahkan konsep dari setiap sesi menjadi studi kasus yang bisa dijalankan: mulai dari analitik bisnis, komputasi paralel, sistem informasi geografis, machine learning end-to-end, perancangan antarmuka, hingga pemantauan deforestasi berbasis citra satelit.

> **Hampir semua data pada notebook bersifat sintetis/ilustratif.** Data dibuat oleh kode agar alur analisis dapat dijalankan tanpa ketergantungan pada data asli. Satu-satunya dataset publik yang dipakai adalah dataset *wine* bawaan scikit-learn pada Sesi 4. Hasilnya tidak dimaksudkan sebagai kesimpulan produksi.

---

## Daftar Isi

- [Ringkasan Sesi](#ringkasan-sesi)
- [Detail Implementasi](#detail-implementasi)
  - [Sesi 1 — Analitik dan AI](#sesi-1--analitik-dan-ai-studi-kasus-pt-maju-bersama)
  - [Sesi 2 — High Performance Computing](#sesi-2--high-performance-computing-hpc)
  - [Sesi 3 — GIS](#sesi-3--gis-dan-analisis-spasial)
  - [Sesi 4 — Machine Learning](#sesi-4--machine-learning-end-to-end)
  - [Sesi 5 — IMK dan Visualisasi Data](#sesi-5--imk-dan-visualisasi-data)
  - [Sesi 6 — Pemantauan Deforestasi](#sesi-6--pemantauan-deforestasi)
- [Struktur Repositori](#struktur-repositori)
- [Cara Menjalankan](#cara-menjalankan)
- [Dependensi](#dependensi)
- [Catatan Data dan Asumsi](#catatan-data-dan-asumsi)
- [Pelajaran Utama](#pelajaran-utama)
- [Kredit](#kredit)

---

## Ringkasan Sesi

| Sesi | Topik | Fokus implementasi | Sel notebook |
|:---:|---|---|:---:|
| 1 | Analitik dan AI | Descriptive, diagnostic, predictive, dan prescriptive analytics pada studi kasus PT Maju Bersama | 0 |
| 2 | High Performance Computing | Mini-LINPACK, simulasi evakuasi tsunami, paralelisasi MPI, alur kerja Slurm | 1 |
| 3 | GIS | Pemilihan lokasi gerai, spatial join, klasifikasi tutupan lahan, luas persil | 2 |
| 4 | Machine Learning | 10 studi kasus supervised, unsupervised, RL, dan semi-supervised | 3 |
| 5 | IMK dan Visualisasi Data | Persona, dasbor, WCAG, prinsip interaksi, Shneiderman, Tufte | 4 |
| 6 | Pemantauan Deforestasi | Optik vs radar, alert perubahan, CNN, evaluasi lintas wilayah, validasi lapangan | 5 |

Setiap sel diawali docstring `SESI n` sehingga urutan dan konteksnya mudah dikenali.

---

## Detail Implementasi

### Sesi 1 — Analitik dan AI: Studi Kasus PT Maju Bersama

Membangun dataset sintetis perusahaan dagang sekaligus menjalankan alur analitik dari hulu ke hilir.

**Dataset yang dibuat**

- 120 pelanggan, 24 produk, dan sekitar 24.000 baris invoice.
- Periode 1 Juli 2024 – 30 Juni 2026.
- Narasi bisnis: pendapatan naik, laba bersih turun, dan piutang naik.

**Yang diimplementasikan**

1. **Pemeriksaan kualitas data** — missing value, duplikasi, kuantitas tidak mungkin, kategori tidak konsisten, relasi pelanggan rusak, dan diskon ekstrem.
2. **Pembersihan data** — deduplikasi, filter nilai tidak valid, normalisasi kategori, imputasi diskon per kategori, serta penandaan anomali untuk ditinjau manusia.
3. **Jebakan granularitas join** — membandingkan join level invoice dengan join yang salah ke tabel pelanggan-bulan, yang menggandakan baris dan merusak total pendapatan.
4. **Descriptive analytics** — tren pendapatan, laba bersih, dan piutang akhir bulan.
5. **Diagnostic analytics** — kenaikan operating expenses, margin per kategori produk, dan keterlambatan pembayaran per segmen pelanggan.
6. **Predictive analytics** — memprediksi invoice yang telat dibayar lebih dari 30 hari:
   - Random Forest dan Logistic Regression;
   - temporal split: latih sampai Desember 2025, uji Januari–Juni 2026;
   - penanganan class imbalance dengan `class_weight`;
   - fitur riwayat keterlambatan pelanggan, fitur kalender, dan promo;
   - demonstrasi data leakage melalui fitur `days_late`;
   - evaluasi accuracy, precision, recall, dan F1.
7. **Prescriptive analytics** — skor risiko per pelanggan, prioritas penagihan, dan simulasi pengurangan piutang dengan asumsi penagihan dini.

**Pelajaran kunci:** accuracy tinggi bisa menyesatkan saat kelas tidak seimbang; data leakage membuat model tampak sempurna; validasi temporal lebih realistis daripada split acak.

---

### Sesi 2 — High Performance Computing (HPC)

Menghubungkan infrastruktur komputasi global dengan tantangan lokal melalui dua demonstrasi.

**A. Mini-LINPACK**

- Menyelesaikan sistem linear padat `Ax = b` menggunakan LU dari LAPACK.
- Mengukur waktu eksekusi, GFLOPS, dan residual ternormalisasi.
- Mengilustrasikan prinsip yang sama dengan benchmark HPL/Top500.

**B. Simulasi evakuasi tsunami**

- Simulasi Monte Carlo berbasis agen untuk mengukur persentase penduduk yang berhasil masuk shelter sebelum gelombang tiba.
- Parameter: jarak ke shelter, kecepatan berjalan, waktu reaksi, jumlah pintu, dan antrean FIFO di pintu shelter.
- Grid skenario: waktu peringatan 10/15/20/30 menit × 1–4 pintu × waktu reaksi 1/3/5 menit.
- Skenario dibagi ke beberapa proses MPI secara round-robin, lalu hasilnya dikumpulkan dengan `gather` dan waktu terlama dihitung dengan `reduce`.
- Jika `mpi4py` tidak tersedia, kode otomatis berjalan serial sehingga tetap aman dijalankan di notebook.

**C. Alur kerja HPC**

- SSH key, login node, `scp`, `sbatch`, dan `squeue`.
- Contoh batch script Slurm untuk menjalankan simulasi dengan beberapa node dan proses MPI.

**Pelajaran kunci:** masalah yang terdiri atas banyak skenario independen cocok diparalelkan; percepatan diukur dengan membandingkan waktu pada jumlah proses yang berbeda.

---

### Sesi 3 — GIS dan Analisis Spasial

Mengimplementasikan operasi GIS dasar hanya dengan NumPy, pandas, scikit-learn, dan `BallTree` haversine — tanpa GeoPandas atau QGIS agar mudah dijalankan di Colab.

**1. Pemilihan lokasi gerai minimarket**

- 600 titik populasi, 25 gerai eksisting, dan 200 kandidat lokasi.
- Menghitung jumlah penduduk dalam radius 1 km.
- Menyaring kandidat yang berjarak minimal 500 m dari gerai lain.
- Memilih tiga lokasi terbaik berdasarkan jumlah penduduk terlayani.

**2. “Snap and solve” laporan warga**

- Enam laporan berkoordinat dihubungkan ke aset kota terdekat menggunakan spatial join.
- Menghasilkan aset terdekat, jenis aset, dan jarak dalam meter.
- Laporan yang terlalu jauh dari aset ditandai untuk diverifikasi.

**3. Klasifikasi tutupan lahan dari citra**

- Lima kelas: air, hutan, sawit, padi, dan permukiman.
- Fitur reflektansi empat band ditambah NDVI.
- Random Forest dengan split stratifikasi 70/30.
- Evaluasi accuracy dan confusion matrix.
- Prediksi diterapkan pada citra sintetis 100 × 100 piksel untuk menghitung luas per kelas dalam hektare.

**4. Luas persil dari koordinat**

- Perhitungan luas poligon dengan formula shoelace pada proyeksi lokal.
- Contoh persil 100 m × 100 m yang mendekati 1 hektare.
- Catatan penggunaan CRS terproyeksi, misalnya UTM 47S/EPSG:32747 untuk Bengkulu.

**Pelajaran kunci:** jarak geodesik, spatial join, dan pemilihan CRS sangat memengaruhi hasil analisis spasial.

---

### Sesi 4 — Machine Learning End-to-End

Sepuluh studi kasus yang mengikuti alur materi, dari preprocessing hingga evaluasi.

| No | Studi kasus | Pendekatan | Evaluasi / output |
|:---:|---|---|---|
| 1 | Prediksi harga rumah | Linear Regression, Random Forest, `ColumnTransformer` | MAE, R², prediksi rumah baru |
| 2 | Prediksi nilai ujian mahasiswa | Linear Regression | MAE, R², koefisien fitur |
| 3a | Deteksi spam email | TF-IDF + Multinomial Naive Bayes | Accuracy, classification report, prediksi teks baru |
| 3b | Persetujuan pinjaman | Logistic Regression, Decision Tree, Random Forest | Accuracy, precision, recall, F1, confusion matrix |
| 4 | Segmentasi pelanggan | K-Means, Agglomerative, DBSCAN, Gaussian Mixture | Inertia, silhouette, profil segmen |
| 5 | Reduksi dimensi | PCA pada dataset wine | Varians kumulatif, akurasi sebelum/sesudah PCA |
| 6 | Deteksi anomali transaksi | Isolation Forest, Z-score | Fraud tertangkap, false alarm |
| 7 | Market basket | Association rules | Support, confidence, lift |
| 8 | Mobil otonom sederhana | Q-Learning pada grid 5 × 5 | Rute hasil belajar dan reward |
| 9 | Semi-supervised learning | Self-training classifier | Perbandingan akurasi: sedikit label vs semi-supervised vs semua label |

**Poin penting yang ditunjukkan**

- Pipeline preprocessing: imputasi missing value, standardisasi, dan one-hot encoding.
- Perbandingan model baseline dan model yang lebih kompleks.
- Overfitting vs underfitting melalui akurasi train dan test pada berbagai kedalaman decision tree.
- Dampak class imbalance dan cara menanganinya.
- Clustering membutuhkan normalisasi sebelum menghitung jarak.
- PCA dapat mempertahankan sebagian besar informasi dengan jumlah komponen yang jauh lebih sedikit.
- Reinforcement learning belajar dari interaksi, reward, dan eksplorasi.

---

### Sesi 5 — IMK dan Visualisasi Data

Membangun dasbor penjualan toko yang dirancang mengikuti konsep interaksi manusia–komputer, bukan hanya “grafik yang terlihat bagus”.

**Persona dan skenario**

- Persona: Rina, staf admin toko yang terbiasa menggunakan Excel.
- Skenario: menyiapkan laporan penjualan bulanan untuk pemilik toko.
- Kebutuhan fungsional dan nonfungsional diturunkan dari persona tersebut.

**Dasbor**

- Overview total penjualan bulanan.
- Filter berdasarkan rentang tanggal dan kategori.
- Detail penjualan per hari.
- Warna konsisten untuk setiap kategori di semua grafik.
- Pesan status sebagai feedback setelah filter dijalankan.
- Validasi rentang tanggal sebagai constraint.

**Konsep yang diterapkan**

- **WCAG (Perceivable):** perhitungan rasio kontras warna, target minimal 4,5:1.
- **Prinsip interaksi:** visibility, feedback, constraints, consistency, dan affordance.
- **Shneiderman:** overview → zoom/filter → details-on-demand.
- **Tufte:** perbandingan chartjunk vs grafik dengan data-ink ratio tinggi.
- **Pemilihan grafik:** line chart, bar chart, scatter plot, dan donut chart.

**Pengujian otomatis**

| Kode | Kebutuhan | Cara uji |
|:---:|---|---|
| F1 | Filter berdasarkan tanggal | Memeriksa batas tanggal hasil filter |
| F2 | Ekspor grafik ke PDF | Memeriksa file PDF ada dan tidak kosong |
| NF1 | Pemuatan grafik cepat | Waktu render < 2 detik pada data diperbesar |
| NF3 | Keterbacaan antarmuka | Kontras warna ≥ 4,5:1 |
| M4 | Constraints | Rentang tanggal terbalik harus ditolak |
| M4 | Feedback | Pesan status muncul setelah filter |

Catatan: kemudahan penggunaan (NF2) tidak dapat diuji otomatis dan tetap memerlukan usability testing dengan pengguna nyata.

**Output runtime**

- `output_imk/laporan_juni.pdf`
- `output_imk/uji_kecepatan.png`
- `output_imk/jenis_grafik.png`
- `output_imk/chartjunk_vs_dataink.png`

---

### Sesi 6 — Pemantauan Deforestasi

Mensimulasikan alur pemantauan deforestasi dari citra satelit hingga validasi lapangan.

**1. Citra optik vs radar**

- Sentinel-2 (optik) terganggu awan; Sentinel-1 (radar) tidak.
- Simulasi tutupan awan pada citra optik dan perhitungan peluang memperoleh citra jernih dalam 10/30 hari.
- Keterbatasan radar dalam mendeteksi objek di bawah kanopi ikut diilustrasikan.

**2. Deforestation alerts**

- Deteksi perubahan per ubin 16 × 16 piksel (2,56 ha pada resolusi 10 m).
- Alert optik: penurunan NDVI > 0,25 pada minimal 5% piksel.
- Jika ubin tertutup awan > 50%, digunakan fallback radar: minimal 10% piksel mengalami penurunan VV lebih dari 1,5 dB.
- Menghitung luas perubahan, alert yang lolos, dan alarm palsu.

**3. Klasifikasi penyebab**

- Enam kelas: pertanian, pertambangan, penebangan, jalan, kebakaran, dan gangguan alami.
- Dataset berlabel sintetis dengan distribusi kelas tidak seimbang (pertanian dominan).
- Model CNN PyTorch: tiga lapisan konvolusi, pooling, dan klasifikasi enam kelas.
- Jika PyTorch tidak tersedia, digunakan Random Forest dengan fitur statistik dan bentuk sebagai fallback.
- Pembobotan kelas untuk mengurangi bias ke kelas dominan.

**4. Evaluasi**

- Overall accuracy, macro F1, F1 per kelas, classification report, dan confusion matrix.
- Menunjukkan bahwa accuracy tinggi belum tentu berarti model baik pada kelas langka.

**5. Generalisasi lintas wilayah**

- Model dilatih di wilayah A dan diuji di wilayah A serta wilayah B.
- Wilayah B memiliki pergeseran kondisi spektral (musim/sensor/vegetasi).
- Penurunan macro F1 menunjukkan pentingnya evaluasi dan penyesuaian model untuk wilayah baru.

**6. Validasi lapangan**

- Alert diklasifikasikan penyebabnya, lalu diurutkan berdasarkan keyakinan model.
- Ubin dengan keyakinan < 0,70 ditandai untuk verifikasi lapangan.
- Trade-off ambang keyakinan 0,5 / 0,7 / 0,9: semakin tinggi ambang, semakin akurat tetapi semakin banyak yang harus dicek.

**7. Frekuensi pembaruan**

- Membandingkan klasifikasi bulanan, alert mingguan, dan sensor termal harian.
- Dengan asumsi kebakaran meluas 25% per hari, jeda deteksi sangat memengaruhi luas area yang terbakar saat ditemukan.

**Output runtime**

- `output_deforestasi/1_optik_vs_radar.png`
- `output_deforestasi/2_contoh_ubin.png`
- `output_deforestasi/3_confusion_matrix.png`
- `output_deforestasi/4_f1_per_kelas.png`
- `output_deforestasi/5_peta_alert.png`

**Pelajaran kunci:** citra tidak menjelaskan penyebab secara lengkap; validasi lapangan dan pemilihan sumber data yang sesuai jenis gangguan tetap diperlukan.

---

## Struktur Repositori

```text
.
├── Implementasi_Kode_Informatics_Practitioner_Lecture_Series (1).ipynb
│   ├── Sel 0 — Sesi 1: Analitik dan AI
│   ├── Sel 1 — Sesi 2: HPC
│   ├── Sel 2 — Sesi 3: GIS
│   ├── Sel 3 — Sesi 4: Machine Learning
│   ├── Sel 4 — Sesi 5: IMK dan Visualisasi Data
│   └── Sel 5 — Sesi 6: Pemantauan Deforestasi
└── README.md
```

Folder `output_imk/` dan `output_deforestasi/` dibuat otomatis saat notebook dijalankan dan tidak disimpan di repositori.

---

## Cara Menjalankan

### Opsi 1 — Google Colab

1. Buka [Google Colab](https://colab.research.google.com/).
2. Pilih **File → Upload notebook**, lalu unggah file `.ipynb` dari repositori ini.
3. Jalankan sel dari atas ke bawah (**Runtime → Run all**).
4. Pada Sesi 5, file PDF akan otomatis diunduh melalui browser saat berjalan di Colab.

### Opsi 2 — Jupyter Notebook lokal

```bash
git clone https://github.com/abelcakra/Implementasi-Kode-Informatics-Practitioner-Lecture-Series.git
cd Implementasi-Kode-Informatics-Practitioner-Lecture-Series

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install numpy pandas scikit-learn matplotlib scipy jupyter
jupyter notebook
```

Buka notebook, lalu jalankan sel secara berurutan. Setiap sel sesi dapat dijalankan sendiri-sendiri, tetapi Sesi 4, 5, dan 6 membutuhkan waktu komputasi lebih lama.

### Opsi 3 — Menjalankan sel sebagai skrip Python

Setiap sel sesi juga dapat disalin ke file `.py` terpisah, lalu dijalankan langsung. Contoh setelah isi sel Sesi 1 disimpan sebagai `sesi1_analitik_ai.py`:

```bash
python sesi1_analitik_ai.py
```

Khusus Sesi 2, jika isinya disimpan sebagai `sesi2_hpc_evakuasi_linpack.py` dan `mpi4py` tersedia, simulasi dapat dijalankan paralel:

```bash
mpirun -n 4 python sesi2_hpc_evakuasi_linpack.py
```

Tanpa `mpi4py`, kode otomatis memakai mode serial.

---

## Dependensi

**Wajib**

| Paket | Digunakan pada |
|---|---|
| Python 3.10+ | Semua sesi |
| NumPy | Semua sesi |
| pandas | Sesi 1, 3, 4, 5, 6 |
| scikit-learn | Sesi 1, 3, 4, 6 |
| Matplotlib | Sesi 5, 6 |
| SciPy | Sesi 6 |
| Jupyter / Google Colab | Menjalankan notebook |

**Opsional**

| Paket | Fungsi |
|---|---|
| `mpi4py` | Paralelisasi Sesi 2; tanpa paket ini berjalan serial |
| `torch` (PyTorch) | CNN Sesi 6; tanpa paket ini memakai fallback Random Forest |
| `google.colab` | Unduh otomatis PDF Sesi 5 di Colab |

Tidak diperlukan GeoPandas, QGIS, atau perangkat lunak GIS eksternal.

---

## Catatan Data dan Asumsi

- Seluruh dataset bisnis, pelanggan, transaksi, citra, dan laporan bersifat **sintetis** dan dibuat oleh kode, kecuali dataset *wine* bawaan scikit-learn pada Sesi 4.
- Nama PT Maju Bersama dan contoh gerai digunakan sebagai ilustrasi materi, bukan data perusahaan sebenarnya.
- Beberapa angka merupakan asumsi eksplisit, antara lain:
  - pengurangan keterlambatan 30% dari penagihan dini pada 20% invoice berisiko tertinggi (Sesi 1);
  - peluang 60% satu akuisisi optik tertutup awan (Sesi 6);
  - pertumbuhan kebakaran 25% per hari (Sesi 6);
  - nilai spektral tiap penyebab deforestasi (Sesi 6).
- Hasil pada notebook ditujukan untuk pembelajaran dan demonstrasi konsep, bukan dasar pengambilan keputusan produksi.
- Untuk data nyata, diperlukan validasi domain, uji kualitas data, evaluasi model yang lebih ketat, serta pertimbangan etika dan privasi.

---

## Pelajaran Utama

1. **Kualitas data menentukan kualitas analisis.** Missing value, duplikasi, dan granularitas join dapat mengubah kesimpulan bisnis.
2. **Accuracy bukan segalanya.** Pada data tidak seimbang, recall, precision, F1, dan confusion matrix lebih informatif.
3. **Hindari data leakage.** Fitur yang hanya diketahui setelah kejadian tidak boleh dipakai untuk prediksi.
4. **Validasi harus mencerminkan dunia nyata.** Temporal split dan evaluasi lintas wilayah lebih jujur daripada split acak.
5. **Paralelisasi cocok untuk beban kerja independen.** Pembagian skenario dan penggabungan hasil adalah pola dasar HPC.
6. **Analisis spasial membutuhkan kesadaran konteks.** Jarak, proyeksi, dan resolusi spasial memengaruhi hasil.
7. **ML adalah alur kerja, bukan hanya algoritma.** Preprocessing, pemilihan model, evaluasi, dan interpretasi saling terkait.
8. **Desain antarmuka harus dapat diuji.** Kebutuhan fungsional dapat diotomasi, tetapi usability dan aksesibilitas tetap memerlukan pengguna nyata.
9. **Model yang baik di satu wilayah belum tentu baik di wilayah lain.** Validasi lapangan tetap menjadi bagian dari sistem.

---

## Kredit

- **Materi:** Informatics Practitioner Lecture Series 2026.
- **Sesi 1:** Andria Arisal.
- **Sesi 2:** Dr. Worawan Diaz Carballo.
- Terima kasih kepada seluruh pemateri dan penyelenggara yang materinya menjadi dasar implementasi ini.

---

Dibuat dan dikelola oleh [abelcakra](https://github.com/abelcakra).
