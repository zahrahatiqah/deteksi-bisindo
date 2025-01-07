import streamlit as st
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import joblib  # Untuk memuat model KNN yang sudah dilatih

# Membuat Streamlit UI
st.set_page_config(page_title="Ruang Belajar KNN", layout="wide", initial_sidebar_state="expanded")

st.title('Ruang Belajar Deployment KNN')

# Load Model KNN
model_path = "scaler.pkl"  # Ganti dengan path model KNN Anda
try:
    knn_model = joblib.load(model_path)
    st.success("Model KNN berhasil dimuat.")
except FileNotFoundError:
    st.error("Model KNN tidak ditemukan. Pastikan model sudah dilatih dan disimpan.")

# Fungsi untuk mengekstrak fitur dari frame
def extract_features(frame):
    # Resize gambar untuk konsistensi ukuran
    resized_frame = cv2.resize(frame, (64, 64))
    # Konversi ke skala abu-abu
    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    # Hitung histogram sebagai fitur
    hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# Fungsi untuk mendeteksi objek
def detect_objects(frame, model):
    features = extract_features(frame)  # Ekstraksi fitur dari frame
    features = features.reshape(1, -1)  # Bentuk ulang untuk prediksi
    label = model.predict(features)[0]  # Prediksi kelas
    confidence = model.predict_proba(features).max()  # Confidence score
    return label, confidence

# Membuat Function untuk proses dan display video
def process_video(source, model, placeholder):
    if source == 'Webcam':
        camera = cv2.VideoCapture(0)  # Kode webcam
    else:
        st.warning("Saat ini hanya mendukung webcam.")
        return

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Deteksi objek menggunakan KNN
        label, confidence = detect_objects(frame, model)

        # Tambahkan label dan confidence ke frame
        cv2.putText(frame, f"{label} ({confidence:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        placeholder.image(frame)

    camera.release()

# Sidebar
with st.sidebar:
    video_source = st.radio('Pilih data video', ['Webcam'])
    process_placeholder = st.empty()

# Process video
with st.sidebar:
    if st.button('Start'):
        process_video(video_source, knn_model, process_placeholder)
with st.sidebar:
    st.image("logo.png", use_column_width=True)
