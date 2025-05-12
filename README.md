# 🧠 Sistem Deteksi Kesehatan Mental Lanjutan

Sistem canggih berbasis AI yang menganalisis teks untuk mendeteksi kondisi kesehatan mental menggunakan Natural Language Processing dan Machine Learning.

## ✨ Fitur

### 🔍 Kemampuan Deteksi Lanjutan
- **Deteksi Multi-kondisi**: Depresi, Kecemasan, Stress, dan kondisi Normal
- **Analisis Sentimen**: Penilaian nada emosional secara real-time
- **Penilaian Level Risiko**: Kategorisasi risiko Tinggi, Sedang, Rendah
- **Skor Kepercayaan**: Metrik kepercayaan keputusan AI yang transparan

### 📊 Analitik Komprehensif
- **Analisis Pola Temporal**: Melacak tren kesehatan mental sepanjang waktu
- **Dashboard Interaktif**: Visualisasi distribusi kondisi dan pola
- **Pelacakan Riwayat**: Menjaga riwayat analisis dengan fungsi ekspor
- **Laporan Detail**: Menghasilkan laporan kesehatan mental yang komprehensif

### 🛡️ Fitur yang Ditingkatkan
- **Pengenalan Emoji**: Menganalisis konteks emosional dari emoji
- **Pembobotan Kata Kunci**: Memprioritaskan indikator klinis
- **Ensemble Multi-model**: Menggabungkan beberapa model AI untuk akurasi
- **Pemrosesan Real-time**: Analisis dan umpan balik instan

## 🚀 Mulai Cepat

### Prasyarat
- Python 3.8 atau lebih tinggi
- pip package manager

### Instalasi

1. **Clone repository**
```bash
git clone https://github.com/yulvianiputeri/mental-health-detection.git
cd mental-health-detection
```

2. **Buat virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Di Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download data NLTK**
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('vader_lexicon')"
```

5. **Latih model**
```bash
python train_model.py
```

6. **Jalankan aplikasi**
```bash
streamlit run app.py
```

## 📁 Struktur Proyek

```
mental-health-detection/
│
├── mental_health_detector.py  # Engine deteksi AI inti
├── train_model.py            # Script pelatihan model
├── app.py                    # Aplikasi web Streamlit
├── requirements.txt          # Dependencies Python
├── README.md                 # Dokumentasi proyek
├── mental_health_model_advanced.pkl  # Model terlatih (dihasilkan)
└── data/                     # Direktori data pelatihan (opsional)
```

## 🧠 Cara Kerja

### 1. Pemrosesan Teks
- Memproses teks input (huruf kecil, penghapusan karakter khusus)
- Ekstraksi fitur linguistik (TF-IDF, n-grams)
- Analisis sentimen menggunakan VADER
- Identifikasi kata kunci kesehatan mental

### 2. Ekstraksi Fitur
- **Fitur Teks**: Panjang, jumlah kata, tanda baca
- **Fitur Sentimen**: Skor positif, negatif, netral
- **Fitur Emoji**: Konteks emosional dari emoji
- **Fitur Kata Kunci**: Indikator kesehatan mental berbobot

### 3. Machine Learning
- **Algoritma**: Gradient Boosting Classifier
- **Cross-validation**: Validasi 5-fold untuk ketahanan
- **Feature Importance**: Mengidentifikasi fitur prediktif utama

### 4. Penilaian Risiko
- Menggabungkan prediksi kondisi dengan skor kepercayaan
- Menghitung level risiko berdasarkan keparahan dan kepastian
- Memberikan rekomendasi yang dapat ditindaklanjuti

## 📊 Performa Model

| Metrik | Skor |
|--------|-------|
| Akurasi Training | 92% |
| Akurasi Testing | 88% |
| Cross-validation | 87% (±3%) |

## 🖥️ Antarmuka Pengguna

### Fitur Utama:
1. **Tab Analisis**: Analisis teks real-time dengan hasil visual
2. **Tab Dashboard**: Analitik dan visualisasi tren
3. **Tab Riwayat**: Analisis masa lalu dengan fungsi ekspor
4. **Tab Sumber Daya**: Sumber daya kesehatan mental dan dukungan krisis

### Metode Input:
- Analisis pesan tunggal
- Pemrosesan riwayat chat batch
- Input suara (segera hadir)

## 🔒 Privasi & Keamanan

- **Tidak Ada Penyimpanan Data**: Analisis dilakukan secara lokal
- **Berbasis Sesi**: Data hanya ada selama sesi aktif
- **Komunikasi Terenkripsi**: Transmisi data yang aman
- **Pemrosesan Anonim**: Tidak memerlukan identifikasi personal

## ⚠️ Disclaimer Penting

Sistem ini:
- **BUKAN alat diagnostik medis**
- **BUKAN pengganti perawatan kesehatan mental profesional**
- **Hanya untuk tujuan screening dan awareness**
- **Harus digunakan bersamaan dengan bimbingan profesional**

## 🆘 Sumber Daya Krisis

Jika Anda mengalami krisis kesehatan mental:
- **Darurat**: 112/119
- **Sejiwa (Hotline Bunuh Diri)**: 119 ext 8
- **Halodoc Konseling**: 500-454
- **LPSP3 UI**: (021) 78842580

## 🤝 Kontribusi

Kontribusi sangat diterima! Silakan kirim Pull Request.

1. Fork repository
2. Buat feature branch (`git checkout -b feature/FiturMenakjubkan`)
3. Commit perubahan (`git commit -m 'Tambah FiturMenakjubkan'`)
4. Push ke branch (`git push origin feature/FiturMenakjubkan`)
5. Buka Pull Request

## 📝 Lisensi

Proyek ini dilisensikan di bawah MIT License - lihat file [LICENSE](LICENSE) untuk detail.

## 🙏 Pengakuan

- NLTK untuk natural language processing
- Scikit-learn untuk algoritma machine learning
- Streamlit untuk antarmuka web
- Plotly untuk visualisasi interaktif

## 📧 Kontak

Untuk pertanyaan atau dukungan:
- Email: yulvianipps02@gmail.com
- Issues: [GitHub Issues](https://github.com/yulvianiputeri/mental-health-detection/issues)

---

**Ingat**: Kesehatan mental Anda penting. Alat ini ada untuk membantu meningkatkan awareness, tetapi dukungan profesional sangat berharga. Jangan ragu untuk menghubungi profesional kesehatan mental ketika diperlukan. 💚

## 🔧 Troubleshooting

### Error Umum dan Solusi

#### 1. **"LookupError: vader_lexicon not found"**
```bash
# Solusi cepat:
python fix_nltk_error.py

# Atau manual:
python -c "import nltk; nltk.download('vader_lexicon')"
```

#### 2. **"The TF-IDF vectorizer is not fitted"**
```bash
# Model belum dilatih, jalankan:
python train_model.py

# Verifikasi fix:
python quick_fix.py
```

#### 3. **"ModuleNotFoundError"**
```bash
# Install dependencies:
pip install -r requirements.txt

# Atau reinstall semua:
pip uninstall streamlit pandas numpy scikit-learn nltk plotly
pip install -r requirements.txt
```

#### 4. **Error SSL di Windows**
```bash
# Jalankan script fix:
python fix_nltk_error.py

# Atau set SSL manual:
python -c "import ssl; ssl._create_default_https_context = ssl._create_unverified_context"
```

### Setup Otomatis

Untuk memudahkan, gunakan script setup otomatis:

```bash
# Setup lengkap (recommended):
python setup_project.py

# Quick fix untuk error umum:
python quick_fix.py

# Download NLTK data saja:
python download_nltk_data.py
```

### Verifikasi Instalasi

```bash
# Cek semua komponen:
python validate_setup.py
```

## 🆘 Jika Masih Bermasalah

1. **Buat ulang virtual environment:**
   ```bash
   # Hapus venv lama
   rm -rf venv  # Linux/Mac
   rmdir /s venv  # Windows
   
   # Buat baru
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate  # Windows
   
   # Install ulang
   pip install -r requirements.txt
   ```

2. **Cek Python dan pip version:**
   ```bash
   python --version  # Minimal 3.8
   pip --version
   ```

3. **Manual download NLTK:**
   ```python
   import nltk
   nltk.download_shell()  # GUI downloader
   ```

## 📋 Langkah-langkah Setup (Detail)

### 1. Setup Environment
```bash
# Clone repository
git clone https://github.com/yulvianiputeri/mental-health-detection.git
cd mental-health-detection

# Virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows

# Dependencies
pip install -r requirements.txt
```

### 2. Download Data NLTK
```bash
# Otomatis
python download_nltk_data.py

# Manual
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt')"
```

### 3. Training Model
```bash
# Training (memerlukan waktu 2-5 menit)
python train_model.py
```

### 4. Running App
```bash
# Jalankan aplikasi
streamlit run app.py
```

### 5. Verifikasi
```bash
# Cek semua OK
python validate_setup.py
```

## 📈 Pengembangan Selanjutnya

- [ ] Dukungan bahasa Indonesia yang lebih baik
- [ ] Integrasi dengan WhatsApp/Telegram
- [ ] Analisis audio real-time
- [ ] Dashboard admin untuk psikolog
- [ ] API untuk integrasi dengan sistem lain
- [ ] Notifikasi push untuk reminder check-in