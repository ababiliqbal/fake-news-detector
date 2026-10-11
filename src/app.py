import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# --- 2. MANAJEMEN MEMORI & PEMUATAN MODEL (CACHING) ---
# Dekorator @st.cache_resource memaksa Streamlit menyimpan objek di RAM cache.
# Model hanya dimuat satu kali saat server pertama kali menyala, bukan setiap tombol ditekan.
@st.cache_resource
def load_resources():
    # Memastikan korpus stopwords tersedia di server/lokal
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    # Pendekatan path dinamis agar aplikasi tidak 'crash' jika dieksekusi dari direktori berbeda
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(BASE_DIR, 'models', 'naive_bayes_model.pkl')
    vectorizer_path = os.path.join(BASE_DIR, 'models', 'tfidf_vectorizer.pkl')
    
    nb_model = joblib.load(model_path)
    tfidf_vectorizer = joblib.load(vectorizer_path)
    
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    
    return nb_model, tfidf_vectorizer, stemmer, stop_words

nb_model, tfidf_vectorizer, stemmer, stop_words = load_resources()

# --- 3. REPLIKASI PIPELINE PREPROCESSING ---
# Harus identik 100% dengan fase eksperimen agar ruang vektor cocok.
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    cleaned_words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(cleaned_words)

# --- 4. ANTARMUKA PENGGUNA (UI) ---
st.title("📰 Detektor Berita Palsu (Fake News)")

st.info("""
**⚠️ PERHATIAN: BATASAN SISTEM (SYSTEM LIMITATIONS)**
1. **Bahasa:** Model ini **HANYA** dapat memproses teks berita berbahasa **Inggris**. Jika memasukkan bahasa lain, prediksi akan gagal secara logika.
2. **Topik Spesifik:** Kecerdasan buatan ini dilatih dan dioptimalkan secara khusus untuk mengidentifikasi berita di domain **Politik AS, Pemerintahan, dan Isu Global (World News)**. Memasukkan teks berita di luar topik tersebut (seperti olahraga, hiburan, atau fiksi) akan menyebabkan penurunan akurasi yang drastis (*Domain Shift*).
""", icon="ℹ️")

st.markdown("Masukkan teks artikel berita sesuai kriteria di atas untuk mengidentifikasi apakah berita tersebut **Asli** atau **Palsu** berdasarkan probabilitas algoritma *Multinomial Naive Bayes*.")

user_input = st.text_area(
    "Teks Berita:", 
    height=250, 
    placeholder="Paste an English political/world news article here..."
)

# --- 5. LOGIKA INFERENSI ---
if st.button("Deteksi Berita"):
    if user_input.strip() == "":
        st.warning("Silakan masukkan teks berita terlebih dahulu.")
    else:
        with st.spinner("Menganalisis pola linguistik..."):
            # 1. Transformasi teks mentah -> teks bersih
            cleaned_text = preprocess_text(user_input)
            
            # 2. Transformasi teks bersih -> Vektor Sparse CSR
            vectorized_text = tfidf_vectorizer.transform([cleaned_text])
            
            # 3. Prediksi dan Kalkulasi Probabilitas
            prediction = nb_model.predict(vectorized_text)[0]
            probabilities = nb_model.predict_proba(vectorized_text)[0]
            
            # 4. Rendering Hasil
            st.divider()
            st.subheader("Hasil Analisis:")
            
            if prediction == 1:
                st.error("🚨 KESIMPULAN: BERITA PALSU (Fake News)")
                st.write(f"**Tingkat Keyakinan Sistem:** {probabilities[1] * 100:.2f}%")
            else:
                st.success("✅ KESIMPULAN: BERITA ASLI (True News)")
                st.write(f"**Tingkat Keyakinan Sistem:** {probabilities[0] * 100:.2f}%")