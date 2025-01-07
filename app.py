import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Set judul aplikasi
st.title("Real-Time BISINDO Detection with KNN")

# Load model KNN
model_path = "scaler.pkl"  # Ganti dengan path model Anda
try:
    knn_model = joblib.load(model_path)
    st.success("Model KNN berhasil dimuat.")
except FileNotFoundError:
    st.error("Model KNN tidak ditemukan. Pastikan model sudah dilatih dan disimpan.")

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Fungsi untuk mengekstrak landmark
def extract_landmarks(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    hand_landmarks_list = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            hand_landmarks_list.append(hand_landmarks)
    
    return hand_landmarks_list

# Fungsi untuk mendeteksi objek
def detect_bisindo(frame, hand_landmarks_list):
    if len(hand_landmarks_list) == 2:
        hand_1 = hand_landmarks_list[0]
        hand_2 = hand_landmarks_list[1]
        
        combined_landmarks = []
        for landmark in hand_1.landmark:
            combined_landmarks.extend([landmark.x, landmark.y, landmark.z])
        for landmark in hand_2.landmark:
            combined_landmarks.extend([landmark.x, landmark.y, landmark.z])

        prediction = knn_model.predict([combined_landmarks])[0]
        probabilities = knn_model.predict_proba([combined_landmarks])
        accuracy = np.max(probabilities) * 100
        return prediction, accuracy
    return None, None

# Streamlit checkbox untuk menjalankan webcam
run = st.checkbox("Run Webcam")
FRAME_WINDOW = st.image([])  # Placeholder untuk video

cap = cv2.VideoCapture(0)  # Akses kamera utama

if not cap.isOpened():
    st.error("Webcam tidak dapat diakses!")

# Jika Run Webcam dicentang
if run:
    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("Gagal membaca frame dari webcam.")
            break

        frame = cv2.flip(frame, 1)
        hand_landmarks_list = extract_landmarks(frame)
        
        prediction, accuracy = detect_bisindo(frame, hand_landmarks_list)

        if prediction:
            # Tampilkan prediksi dan akurasi pada frame
            cv2.putText(frame, f'{prediction} ({accuracy:.2f}%)', (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Tampilkan frame di Streamlit
        FRAME_WINDOW.image(frame, channels="BGR")

    cap.release()
else:
    st.write("Webcam stopped.")
