import streamlit as st
import cv2
from deepface import DeepFace
import numpy as np
from PIL import Image

st.title("🎭 Real-Time Sentiment Detection")

run = st.checkbox('Start Camera')

FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

def analyze_emotion(frame):
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        return emotion
    except:
        return "No face detected"

while run:
    ret, frame = camera.read()
    if not ret:
        st.warning("Camera not accessible")
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    emotion = analyze_emotion(frame_rgb)
    annotated_frame = cv2.putText(frame_rgb.copy(), f"Emotion: {emotion}", (20, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    FRAME_WINDOW.image(annotated_frame)
else:
    st.write('Stopped')
    camera.release()
