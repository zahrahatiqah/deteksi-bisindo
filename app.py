import numpy as np
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
import joblib  # Untuk memuat model KNN yang sudah dilatih

# Set judul aplikasi
st.title("Real-Time Object Detection with KNN")

# Load model KNN (pastikan file model sudah tersedia)
model_path = "scaler.pkl"  # Ganti dengan path ke model KNN Anda
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
def detect_objects(frame):
    features = extract_features(frame)  # Ekstraksi fitur dari frame
    features = features.reshape(1, -1)  # Bentuk ulang untuk prediksi
    label = knn_model.predict(features)[0]  # Prediksi kelas
    confidence = knn_model.predict_proba(features).max()  # Confidence score
    return label, confidence

# Stream video dari webcam
run = st.checkbox("Run Webcam")
FRAME_WINDOW = st.image([], channels="BGR")  # Placeholder untuk video

# cap = cv2.VideoCapture(0)  # Akses kamera utama
# if not cap.isOpened():
#     st.error("Webcam tidak dapat diakses!")

# Jalankan deteksi secara real-time
while run:
    ret, frame = cap.read()
    if not ret:
        st.warning("Gagal membaca frame dari webcam.")
        break

    # Deteksi objek menggunakan KNN
    label, confidence = detect_objects(frame)

    # Tambahkan informasi label pada frame
    cv2.putText(frame, f"{label} ({confidence:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Tampilkan frame
    FRAME_WINDOW.image(frame, channels="BGR")

cap.release()
st.write("Webcam stopped.")
