import streamlit as st
import cv2
import numpy as np
import time
from datetime import datetime

# Page config
st.set_page_config(
    page_title="🎭 Face Detector",
    page_icon="🎭",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .status-card {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.2rem;
    }
    .face-detected { 
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border: 3px solid #28a745;
        color: #155724;
    }
    .no-face { 
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border: 3px solid #dc3545;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'detections' not in st.session_state:
    st.session_state.detections = 0
if 'emotions_detected' not in st.session_state:
    st.session_state.emotions_detected = []

# Load face detector
@st.cache_resource
def load_face_detector():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

face_cascade = load_face_detector()

# Simple emotion simulation
def get_random_emotion():
    """Simulate emotion detection with random emotions"""
    import random
    emotions = [
        ('😊', 'Happy'), 
        ('😐', 'Neutral'), 
        ('😲', 'Surprised'), 
        ('🤔', 'Thinking'), 
        ('😌', 'Calm'),
        ('😄', 'Joyful'),
        ('🙂', 'Content')
    ]
    emoji, emotion = random.choice(emotions)
    confidence = round(random.uniform(0.6, 0.95), 2)
    return emoji, emotion, confidence

def detect_faces_and_emotions(frame):
    """Detect faces and simulate emotions"""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))
    
    results = []
    for (x, y, w, h) in faces:
        emoji, emotion, confidence = get_random_emotion()
        results.append({
            'bbox': (x, y, w, h),
            'emoji': emoji,
            'emotion': emotion,
            'confidence': confidence
        })
    
    return results

def draw_face_boxes(frame, detections):
    """Draw bounding boxes and emotion labels"""
    for detection in detections:
        x, y, w, h = detection['bbox']
        emoji = detection['emoji']
        emotion = detection['emotion']
        confidence = detection['confidence']
        
        # Color based on confidence
        if confidence > 0.8:
            color = (0, 255, 0)  # Green
        elif confidence > 0.7:
            color = (0, 255, 255)  # Yellow
        else:
            color = (255, 165, 0)  # Orange
        
        # Draw face rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
        
        # Emotion label
        label = f"{emoji} {emotion} ({confidence})"
        
        # Label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(frame, (x, y-35), (x + label_size[0] + 10, y), color, -1)
        cv2.putText(frame, label, (x + 5, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame

# Main App
st.markdown('<h1 class="main-header">🎭 Real-Time Face & Emotion Detection</h1>', 
           unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; color: #666; margin-bottom: 2rem;'>
    <p><strong>✅ No complex dependencies • Works everywhere • Fast deployment</strong></p>
    <p>Face detection with simulated emotion analysis using OpenCV</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🎛️ Camera Controls")
    camera_on = st.checkbox("🎥 Start Camera", key="camera_control")
    
    st.divider()
    
    st.subheader("⚙️ Settings")
    show_confidence = st.checkbox("Show Confidence Scores", True)
    detection_sensitivity = st.slider("Detection Sensitivity", 1, 5, 3)
    
    st.divider()
    
    st.subheader("📊 Session Stats")
    st.metric("Total Face Detections", st.session_state.detections)
    
    if st.session_state.emotions_detected:
        most_common = max(set(st.session_state.emotions_detected), 
                         key=st.session_state.emotions_detected.count)
        st.metric("Most Common Emotion", most_common)
    
    if st.button("🗑️ Reset Counter"):
        st.session_state.detections = 0
        st.session_state.emotions_detected = []
        st.rerun()

# Main content
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("📹 Live Camera Feed")
    frame_container = st.empty()

with col2:
    st.subheader("📊 Live Status")
    status_container = st.empty()

# Camera processing
if camera_on:
    camera = cv2.VideoCapture(0)
    
    # Set camera properties
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_count = 0
    
    while camera_on:
        ret, frame = camera.read()
        
        if not ret:
            st.error("❌ Camera not accessible! Please check browser permissions.")
            break
        
        frame_count += 1
        
        # Process every few frames for performance
        if frame_count % detection_sensitivity == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces and emotions
            detections = detect_faces_and_emotions(frame_rgb)
            
            # Draw annotations
            annotated_frame = draw_face_boxes(frame_rgb.copy(), detections)
            
            # Add frame info
            cv2.putText(annotated_frame, f"Frame: {frame_count} | Faces: {len(detections)}", 
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display frame
            frame_container.image(annotated_frame, channels="RGB", use_column_width=True)
            
            # Update status
            if detections:
                # Show first detected face info
                detection = detections[0]
                emoji = detection['emoji']
                emotion = detection['emotion']
                confidence = detection['confidence']
                
                status_html = f"""
                <div class="status-card face-detected">
                    <h1>{emoji}</h1>
                    <h2>{emotion}</h2>
                    <h4>Confidence: {confidence}</h4>
                    <p>✅ {len(detections)} face(s) detected</p>
                </div>
                """
                
                # Update counters
                st.session_state.detections += 1
                st.session_state.emotions_detected.append(emotion)
                
                # Keep only last 20 emotions
                if len(st.session_state.emotions_detected) > 20:
                    st.session_state.emotions_detected = st.session_state.emotions_detected[-20:]
                
            else:
                status_html = """
                <div class="status-card no-face">
                    <h2>❌ No Face Detected</h2>
                    <p>Position your face in the camera</p>
                    <p><small>Make sure you have good lighting</small></p>
                </div>
                """
            
            status_container.markdown(status_html, unsafe_allow_html=True)
        
        # Small delay
        time.sleep(0.05)
        
        # Check if user stopped camera
        if not camera_on:
            break
    
    camera.release()
    
else:
    frame_container.info("📷 Click 'Start Camera' in the sidebar to begin detection")
    status_container.empty()

# Control buttons
st.subheader("🎮 Controls")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📷 Take Screenshot", type="secondary"):
        st.success("📷 Screenshot feature - coming soon!")

with col2:
    if st.button("📊 Show Analytics", type="secondary"):
        if st.session_state.emotions_detected:
            st.balloons()
            st.success(f"🎉 Detected {len(set(st.session_state.emotions_detected))} different emotions!")
        else:
            st.info("Start camera to collect emotion data first")

with col3:
    if st.button("🔄 Reset Everything", type="secondary"):
        st.session_state.detections = 0
        st.session_state.emotions_detected = []
        st.success("🔄 Everything reset!")
        st.rerun()

# Information section
with st.expander("📖 How This App Works"):
    st.markdown("""
    ### 🎯 What This App Does
    - **Face Detection**: Uses OpenCV's built-in face detection (Haar Cascades)
    - **Emotion Simulation**: Randomly assigns emotions for demonstration
    - **Real-time Processing**: Analyzes your camera feed live
    - **No Complex Dependencies**: Uses only standard Python libraries
    
    ### ✅ Why This Version Works
    - **No TensorFlow** - Avoids Python version conflicts
    - **No FER library** - Eliminates dependency issues  
    - **Uses OpenCV only** - Pre-installed on Streamlit Cloud
    - **Lightweight** - Fast deployment and performance
    
    ### 🚀 Features
    - Real-time face detection with bounding boxes
    - Simulated emotion recognition with confidence scores
    - Live statistics and counters
    - Adjustable detection sensitivity
    - Works on any device with a camera
    
    ### 🔧 Usage Tips
    1. Click **Start Camera** and allow browser camera access
    2. Position your face clearly in the camera view
    3. Watch as faces are detected with emotion labels
    4. Try adjusting **Detection Sensitivity** for performance
    5. Use **Reset** to clear all counters
    
    ### 📊 Note About Emotions
    The emotions shown are **simulated for demonstration purposes**. 
    In a production app, you would integrate actual ML models for real emotion detection.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🎭 <strong>OpenCV-Based Face Detection</strong> • No Complex Dependencies</p>
    <p><em>Reliable • Fast • Works Everywhere</em></p>
    <p>Built with ❤️ using Streamlit and OpenCV</p>
</div>
""", unsafe_allow_html=True)
