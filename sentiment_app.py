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
    .camera-error {
        background: linear-gradient(135deg, #fff3cd, #ffeaa7);
        border: 3px solid #856404;
        color: #856404;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'detections' not in st.session_state:
    st.session_state.detections = 0
if 'emotions_detected' not in st.session_state:
    st.session_state.emotions_detected = []
if 'camera_tested' not in st.session_state:
    st.session_state.camera_tested = False

# Load face detector
@st.cache_resource
def load_face_detector():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

face_cascade = load_face_detector()

# Test camera function
def test_camera_access():
    """Test if camera is accessible"""
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            return ret and frame is not None
        return False
    except Exception as e:
        return False

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

# Camera troubleshooting section
st.markdown("""
<div class="camera-error">
    <h3>📹 Camera Access Required</h3>
    <p><strong>If you see "Camera not accessible" error:</strong></p>
    <ol>
        <li>Look for a <strong>camera icon 🎥</strong> in your browser's address bar</li>
        <li>Click it and select <strong>"Allow"</strong></li>
        <li><strong>Refresh this page</strong> (F5 or Ctrl+R)</li>
        <li>Click <strong>"Start Camera"</strong> again</li>
    </ol>
    <p><strong>Alternative:</strong> Try opening this app in <strong>Chrome browser</strong> if Edge doesn't work</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🎛️ Camera Controls")
    
    # Test camera button
    if st.button("🔍 Test Camera Access"):
        with st.spinner("Testing camera..."):
            camera_works = test_camera_access()
            st.session_state.camera_tested = True
            if camera_works:
                st.success("✅ Camera is accessible!")
            else:
                st.error("❌ Camera not accessible - check permissions")
    
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

# Camera processing with enhanced error handling
if camera_on:
    try:
        camera = cv2.VideoCapture(0)
        
        # Enhanced camera initialization
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Check if camera opened successfully
        if not camera.isOpened():
            st.error("❌ Camera could not be opened. Check if:")
            st.error("• Camera permissions are granted in browser")
            st.error("• No other app is using the camera")
            st.error("• Camera drivers are installed properly")
        else:
            frame_count = 0
            successful_reads = 0
            
            while camera_on:
                ret, frame = camera.read()
                
                if not ret:
                    if frame_count == 0:
                        st.error("❌ Camera not accessible! Please check browser permissions.")
                        st.info("💡 **Edge Browser Fix:**")
                        st.info("1. Look for camera icon 🎥 in address bar")
                        st.info("2. Click it and select 'Allow'")
                        st.info("3. Refresh this page (F5)")
                        st.info("4. Or try in Chrome browser")
                    break
                
                successful_reads += 1
                frame_count += 1
                
                # Process every few frames for performance
                if frame_count % detection_sensitivity == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Detect faces and emotions
                    detections = detect_faces_and_emotions(frame_rgb)
                    
                    # Draw annotations
                    annotated_frame = draw_face_boxes(frame_rgb.copy(), detections)
                    
                    # Add frame info
                    cv2.putText(annotated_frame, f"✅ Camera Active | Frame: {frame_count} | Faces: {len(detections)}", 
                               (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
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
                            <h2>👀 Looking for faces...</h2>
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
        
    except Exception as e:
        st.error(f"❌ Camera error: {str(e)}")
        st.info("💡 Try refreshing the page or using a different browser")
        
else:
    frame_container.info("📷 Click 'Start Camera' in the sidebar to begin detection")
    status_container.empty()

# Control buttons
st.subheader("🎮 Controls & Help")
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

# Browser-specific help
with st.expander("🔧 Browser-Specific Camera Help"):
    st.markdown("""
    ### 🌐 Microsoft Edge
    1. **Look for camera icon** 🎥 in address bar
    2. **Click it** → Select **"Allow"**
    3. **Refresh page** (F5)
    4. If no icon appears: **Settings** → **Cookies and site permissions** → **Camera** → Add this site to **Allow**
    
    ### 🌐 Google Chrome  
    1. **Click camera icon** 🎥 in address bar → **Allow**
    2. Or go to **Settings** → **Privacy and security** → **Site Settings** → **Camera** → **Allow**
    
    ### 🌐 Firefox
    1. **Click shield icon** 🛡️ → **Turn off Tracking Protection** for this site
    2. **Allow camera** when prompted
    
    ### 🖥️ Windows System Settings
    1. **Settings** → **Privacy & Security** → **Camera**
    2. Turn ON **"Camera access"**
    3. Turn ON **"Let apps access your camera"**
    4. Turn ON **"Let desktop apps access your camera"**
    """)

# Information section
with st.expander("📖 How This App Works"):
    st.markdown("""
    ### 🎯 What This App Does
    - **Face Detection**: Uses OpenCV's built-in face detection (Haar Cascades)
    - **Emotion Simulation**: Randomly assigns emotions for demonstration
    - **Real-time Processing**: Analyzes your camera feed live
    - **No Complex Dependencies**: Uses only standard Python libraries
    
    ### ✅ Technical Details
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
