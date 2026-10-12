# 📰 Fake News Detector (NLP)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fake-news-detector-naive-bayes.streamlit.app/)

Sebuah aplikasi web interaktif berbasis **Machine Learning** yang dirancang untuk mendeteksi kebenaran sebuah artikel berita (khususnya seputar politik dan isu global) menggunakan pendekatan *Natural Language Processing* (NLP). 

Sistem ini dilatih menggunakan algoritma **Multinomial Naive Bayes** yang dikombinasikan dengan ekstraksi fitur **TF-IDF (Term Frequency-Inverse Document Frequency)**, mencapai tingkat akurasi yang tinggi dalam membedakan pola linguistik antara berita asli (True News) dan berita palsu (Fake News).

## ✨ Fitur Utama
- **Klasifikasi Teks Real-Time:** Menghasilkan prediksi instan beserta tingkat probabilitas (Confidence Score).
- **Pipeline NLP Lengkap:** Menerapkan pembersihan teks berlapis, termasuk *lowercasing*, penghapusan URL & karakter khusus, *stopwords filtering*, dan *stemming* menggunakan NLTK.
- **Defensive Design:** Antarmuka dilengkapi dengan validasi input untuk mencegah komputasi *string* kosong atau *noise* yang dapat menyebabkan *error* pada model.
- **Deployment Cloud:** Berjalan secara *serverless* menggunakan ekosistem Streamlit Community Cloud.

## 🛠️ Stack Teknologi
- **Bahasa:** Python 3
- **Machine Learning:** Scikit-Learn (MultinomialNB, TfidfVectorizer)
- **Natural Language Processing:** NLTK (Natural Language Toolkit)
- **Data Manipulation:** Pandas, NumPy
- **Web Framework:** Streamlit
- **Model Serialization:** Joblib

## 🚀 Cara Instalasi dan Menjalankan di Komputer Lokal

Jika Anda ingin menjalankan atau memodifikasi aplikasi ini di komputer Anda sendiri, silakan ikuti langkah-langkah di bawah ini:

### 1. Kloning Repositori
Buka terminal/command prompt Anda dan jalankan perintah berikut untuk mengunduh kode sumber:
```bash
git clone https://github.com/ababiliqbal/fake-news-detector.git
cd fake-news-detector
```

### 2. Buat Virtual Environment (Sangat Direkomendasikan)
Untuk mencegah bentrok versi pustaka dengan proyek Python Anda yang lain, buat lingkungan virtual:

**Untuk Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Untuk Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instal Dependensi
Instal semua pustaka yang dibutuhkan melalui file `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4. Jalankan Aplikasi
Nyalakan server lokal Streamlit:
```bash
streamlit run src/app.py
```
Aplikasi akan secara otomatis terbuka di *browser* *default* Anda pada alamat `http://localhost:8501`.

## ⚠️ Batasan Sistem (System Limitations)
Harap diperhatikan bahwa model inferensi ini memiliki batasan *domain* pelatihan:
1. **Bahasa:** Hanya mendukung teks artikel berbahasa **Inggris**.
2. **Topik:** Dioptimalkan untuk berita terkait **Politik AS, Pemerintahan, dan Isu Global**. Memasukkan teks di luar domain tersebut (misal: olahraga, hiburan, fiksi, atau hasil terjemahan mesin yang buruk) dapat menyebabkan *Domain Shift* dan menurunkan akurasi deteksi secara signifikan.