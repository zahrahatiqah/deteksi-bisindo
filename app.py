import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import cv2

# Load the face detection model (Haar Cascade)
cascade = cv2.CascadeClassifier("scaler.pkl")

# Define the video processor class
class VideoProcessor:
    def recv(self, frame):
        # Convert the frame to ndarray format (OpenCV uses BGR format)
        frm = frame.to_ndarray(format="bgr24")

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

        # Draw rectangles around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frm, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # Convert the processed frame back to VideoFrame and return
        return av.VideoFrame.from_ndarray(frm, format="bgr24")

# Streamlit setup
st.title("Face Detection with Streamlit WebRTC")

# Start the video stream
webrtc_streamer(
    key="key", 
    video_processor_factory=VideoProcessor, 
    rtc_configuration=RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
)

