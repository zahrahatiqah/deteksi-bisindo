import cv2
import numpy as np
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Set judul aplikasi
st.title("Real-Time Object Detection with KNN")

# Load model KNN
model_path = "scaler.pkl"  # Path ke file model
try:
    knn_model = joblib.load(model_path)
    st.success("Model KNN berhasil dimuat.")
except FileNotFoundError:
    st.error("Model KNN tidak ditemukan. Pastikan file model tersedia.")

# Fungsi ekstraksi fitur
def extract_features(frame):
    resized_frame = cv2.resize(frame, (64, 64))
    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# Fungsi deteksi objek
def detect_objects(frame):
    features = extract_features(frame)
    features = features.reshape(1, -1)
    label = knn_model.predict(features)[0]
    confidence = knn_model.predict_proba(features).max()
    return label, confidence

# Stream video
run = st.checkbox("Run Webcam")
FRAME_WINDOW = st.image([])  # Placeholder untuk video

cap = None

if run:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Webcam tidak dapat diakses!")
    else:
        while run:
            ret, frame = cap.read()
            if not ret:
                st.warning("Gagal membaca frame dari webcam.")
                break

            # Deteksi objek
            label, confidence = detect_objects(frame)

            # Tambahkan label pada frame
            cv2.putText(frame, f"{label} ({confidence:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            FRAME_WINDOW.image(frame, channels="BGR")

        cap.release()
else:
    if cap and cap.isOpened():
        cap.release()
    st.write("Webcam stopped.")
