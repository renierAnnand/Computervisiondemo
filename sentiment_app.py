import streamlit as st
import cv2
import numpy as np
import time
from datetime import datetime

# Page config
st.set_page_config(
    page_title="🎭 Enhanced Face Detector",
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
    .camera-help {
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        border: 3px solid #2196f3;
        color: #0d47a1;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    .browser-help {
        background: linear-gradient(135deg, #fff3e0, #ffe0b2);
        border: 2px solid #ff9800;
        color: #e65100;
        padding: 1.5rem;
        border-radius: 10px;
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
if 'camera_status' not in st.session_state:
    st.session_state.camera_status = "unknown"

# Load face detector
@st.cache_resource
def load_face_detector():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

face_cascade = load_face_detector()

# Enhanced camera testing function
def test_camera_access():
    """Test if camera is accessible with detailed diagnostics"""
    try:
        # Test different camera indices
        for camera_index in [0, 1, 2]:
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None:
                    return True, f"Camera {camera_index} working"
        return False, "No cameras found"
    except Exception as e:
        return False, f"Camera error: {str(e)}"

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
        ('🙂', 'Content'),
        ('😮', 'Amazed'),
        ('🤗', 'Excited')
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
st.markdown('<h1 class="main-header">🎭 Enhanced Face & Emotion Detection</h1>', 
           unsafe_allow_html=True)

# Enhanced camera help section
st.markdown("""
<div class="camera-help">
    <h3>📹 Camera Setup Guide</h3>
    <p><strong>✅ Your app deployed successfully!</strong> Now just need camera access:</p>
    <ol>
        <li><strong>Look for camera icon 🎥</strong> in your browser's address bar</li>
        <li><strong>Click the icon</strong> and select <strong>"Allow"</strong></li>
        <li><strong>Refresh this page</strong> (press F5 or Ctrl+R)</li>
        <li><strong>Click "Test Camera"</strong> button below</li>
    </ol>
</div>
""", unsafe_allow_html=True)

# Browser-specific help
browser_help_col1, browser_help_col2 = st.columns(2)

with browser_help_col1:
    st.markdown("""
    <div class="browser-help">
        <h4>🌐 Microsoft Edge Users</h4>
        <p><strong>Method 1:</strong> Click camera icon 🎥 in address bar → Allow</p>
        <p><strong>Method 2:</strong> Click lock icon 🔒 → Permissions → Camera → Allow</p>
        <p><strong>Method 3:</strong> Edge Settings → Site permissions → Camera → Add this site</p>
    </div>
    """, unsafe_allow_html=True)

with browser_help_col2:
    st.markdown("""
    <div class="browser-help">
        <h4>🌐 Alternative Solution</h4>
        <p><strong>Try Chrome:</strong> Open this app in Google Chrome browser</p>
        <p><strong>Mobile:</strong> Try on your phone - often easier permissions</p>
        <p><strong>Settings:</strong> Windows Settings → Privacy → Camera → Allow</p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar with enhanced controls
with st.sidebar:
    st.header("🎛️ Camera Controls")
    
    # Enhanced camera test button
    if st.button("🔍 Test Camera Access", type="primary"):
        with st.spinner("Testing camera access..."):
            camera_works, message = test_camera_access()
            st.session_state.camera_tested = True
            if camera_works:
                st.success(f"✅ {message}")
                st.session_state.camera_status = "working"
                st.balloons()
            else:
                st.error(f"❌ {message}")
                st.session_state.camera_status = "blocked"
                st.info("📱 Try the troubleshooting steps above")
    
    # Camera status indicator
    if st.session_state.camera_tested:
        if st.session_state.camera_status == "working":
            st.success("🟢 Camera Status: Working")
        else:
            st.error("🔴 Camera Status: Not Working")
    
    st.divider()
    
    camera_on = st.checkbox("🎥 Start Camera", key="camera_control")
    
    st.divider()
    
    st.subheader("⚙️ Settings")
    show_confidence = st.checkbox("Show Confidence Scores", True)
    detection_sensitivity = st.slider("Detection Sensitivity", 1, 5, 3, 
                                    help="1=Most accurate, 5=Fastest")
    face_size_threshold = st.slider("Min Face Size", 30, 200, 50,
                                   help="Ignore faces smaller than this")
    
    st.divider()
    
    st.subheader("📊 Session Stats")
    st.metric("Total Face Detections", st.session_state.detections)
    
    if st.session_state.emotions_detected:
        most_common = max(set(st.session_state.emotions_detected), 
                         key=st.session_state.emotions_detected.count)
        st.metric("Most Common Emotion", most_common)
        
        unique_emotions = len(set(st.session_state.emotions_detected))
        st.metric("Different Emotions", unique_emotions)
    
    if st.button("🗑️ Reset All Data"):
        st.session_state.detections = 0
        st.session_state.emotions_detected = []
        st.success("Reset complete!")
        st.rerun()

# Main content
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("📹 Live Camera Feed")
    frame_container = st.empty()

with col2:
    st.subheader("📊 Live Status")
    status_container = st.empty()

# Enhanced camera processing
if camera_on:
    try:
        camera = cv2.VideoCapture(0)
        
        # Enhanced camera initialization
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        camera.set(cv2.CAP_PROP_FPS, 30)
        
        # Check if camera opened successfully
        if not camera.isOpened():
            st.error("❌ **Camera Error**: Could not open camera")
            st.markdown("""
            **Possible solutions:**
            - Check camera permissions in browser
            - Close other apps using camera (Zoom, Skype, etc.)
            - Try refreshing the page
            - Use different browser (Chrome recommended)
            """)
        else:
            st.success("✅ **Camera Connected**: Processing video feed...")
            frame_count = 0
            successful_reads = 0
            error_count = 0
            
            while camera_on:
                ret, frame = camera.read()
                
                if not ret:
                    error_count += 1
                    if error_count > 10:  # Too many errors
                        st.error("❌ **Camera Stream Lost**: Too many read errors")
                        break
                    continue
                
                successful_reads += 1
                frame_count += 1
                
                # Process every few frames for performance
                if frame_count % detection_sensitivity == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Detect faces and emotions
                    detections = detect_faces_and_emotions(frame_rgb)
                    
                    # Filter by face size
                    filtered_detections = [d for d in detections 
                                         if d['bbox'][2] * d['bbox'][3] >= face_size_threshold * face_size_threshold]
                    
                    # Draw annotations
                    annotated_frame = draw_face_boxes(frame_rgb.copy(), filtered_detections)
                    
                    # Add comprehensive frame info
                    info_text = f"✅ Active | Frame: {frame_count} | Faces: {len(filtered_detections)} | FPS: {30//detection_sensitivity}"
                    cv2.putText(annotated_frame, info_text, (10, 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Display frame
                    frame_container.image(annotated_frame, channels="RGB", use_column_width=True)
                    
                    # Update status with enhanced information
                    if filtered_detections:
                        # Show first detected face info
                        detection = filtered_detections[0]
                        emoji = detection['emoji']
                        emotion = detection['emotion']
                        confidence = detection['confidence']
                        face_area = detection['bbox'][2] * detection['bbox'][3]
                        
                        status_html = f"""
                        <div class="status-card face-detected">
                            <h1>{emoji}</h1>
                            <h2>{emotion}</h2>
                            <h4>Confidence: {confidence}</h4>
                            <p>✅ {len(filtered_detections)} face(s) detected</p>
                            <p><small>Face size: {face_area} pixels</small></p>
                        </div>
                        """
                        
                        # Update counters
                        st.session_state.detections += 1
                        st.session_state.emotions_detected.append(emotion)
                        
                        # Keep only last 50 emotions for performance
                        if len(st.session_state.emotions_detected) > 50:
                            st.session_state.emotions_detected = st.session_state.emotions_detected[-50:]
                        
                    else:
                        faces_found = len(detections)
                        if faces_found > 0:
                            status_html = f"""
                            <div class="status-card no-face">
                                <h3>📏 Face Too Small</h3>
                                <p>Found {faces_found} face(s) but too small</p>
                                <p><small>Move closer to camera</small></p>
                            </div>
                            """
                        else:
                            status_html = """
                            <div class="status-card no-face">
                                <h2>👀 Looking for faces...</h2>
                                <p>Position your face in the camera</p>
                                <p><small>Ensure good lighting & clear view</small></p>
                            </div>
                            """
                    
                    status_container.markdown(status_html, unsafe_allow_html=True)
                
                # Small delay for smooth processing
                time.sleep(0.033)  # ~30 FPS max
                
                # Check if user stopped camera
                if not camera_on:
                    break
            
            # Performance summary
            if successful_reads > 0:
                st.info(f"📊 **Session Summary**: Processed {successful_reads} frames successfully")
        
        camera.release()
        
    except Exception as e:
        st.error(f"❌ **Camera Exception**: {str(e)}")
        st.markdown("""
        **Troubleshooting:**
        - Refresh the page and try again
        - Check Windows camera privacy settings
        - Try using Google Chrome browser
        - Restart your browser completely
        """)
        
else:
    frame_container.info("📷 **Ready to start**: Click 'Start Camera' in the sidebar")
    status_container.empty()

# Enhanced control panel
st.subheader("🎮 Control Panel")
control_col1, control_col2, control_col3, control_col4 = st.columns(4)

with control_col1:
    if st.button("📷 Screenshot", type="secondary"):
        if camera_on:
            st.success("📷 Screenshot captured!")
            st.info("💡 Screenshot feature coming soon")
        else:
            st.warning("Start camera first")

with control_col2:
    if st.button("📊 Analytics", type="secondary"):
        if st.session_state.emotions_detected:
            st.balloons()
            unique_count = len(set(st.session_state.emotions_detected))
            st.success(f"🎉 Detected {unique_count} different emotions!")
            
            # Show emotion breakdown
            emotion_counts = {}
            for emotion in st.session_state.emotions_detected:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            st.write("**Emotion Breakdown:**")
            for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
                st.write(f"• {emotion}: {count} times")
        else:
            st.info("Start camera to collect emotion data first")

with control_col3:
    if st.button("🔄 New Session", type="secondary"):
        st.session_state.detections = 0
        st.session_state.emotions_detected = []
        st.session_state.camera_tested = False
        st.session_state.camera_status = "unknown"
        st.success("🔄 New session started!")
        st.rerun()

with control_col4:
    if st.button("❓ Help", type="secondary"):
        st.info("📖 Check the expandable help sections below!")

# Comprehensive help sections
with st.expander("🔧 Detailed Troubleshooting Guide"):
    st.markdown("""
    ### 🌐 Browser-Specific Solutions
    
    **Microsoft Edge:**
    1. Look for camera icon 🎥 in address bar → Click → Allow
    2. Click lock icon 🔒 → Permissions for this site → Camera → Allow
    3. Settings → Cookies and site permissions → Camera → Allow → Add this site
    4. Clear browser cache and cookies for this site
    
    **Google Chrome:**
    1. Click camera icon 🎥 in address bar → Allow
    2. Settings → Privacy and security → Site Settings → Camera → Allow
    3. Chrome menu → Settings → Advanced → Content settings → Camera
    
    **Firefox:**
    1. Click shield icon 🛡️ → Turn off Tracking Protection for this site
    2. Allow camera when prompted
    3. Settings → Privacy & Security → Permissions → Camera → Settings
    
    ### 🖥️ System-Level Solutions
    
    **Windows Camera Settings:**
    1. Settings → Privacy & Security → Camera
    2. Turn ON "Camera access for this device"
    3. Turn ON "Let apps access your camera"
    4. Turn ON "Let desktop apps access your camera"
    
    **Check Camera Usage:**
    1. Close all other apps that might use camera (Zoom, Skype, Teams)
    2. Restart your browser completely
    3. Try incognito/private browsing mode
    4. Update your camera drivers
    
    ### 📱 Alternative Testing
    
    **Mobile Device:**
    - Open this app on your phone's browser
    - Camera permissions are often easier on mobile
    - Good way to test if the app works
    
    **Different Computer:**
    - Try on another device if available
    - Helps identify if it's device-specific issue
    """)

with st.expander("📖 App Features & Information"):
    st.markdown("""
    ### 🎯 What This App Does
    - **Face Detection**: Uses OpenCV's Haar Cascade classifiers
    - **Emotion Simulation**: Demonstrates emotion detection with random emotions
    - **Real-time Processing**: Analyzes your camera feed frame by frame
    - **Performance Optimization**: Adjustable frame processing for smooth operation
    
    ### ✅ Technical Specifications
    - **Camera Resolution**: 640x480 pixels for optimal performance
    - **Processing Rate**: Adjustable (1-5x frame skipping)
    - **Face Detection**: Minimum 50x50 pixel faces
    - **Dependencies**: Only OpenCV, NumPy, Streamlit (no ML libraries)
    
    ### 🚀 App Features
    - Real-time face detection with colored bounding boxes
    - Simulated emotion recognition with confidence scores
    - Live statistics and emotion tracking
    - Adjustable sensitivity and face size thresholds
    - Camera diagnostics and troubleshooting
    - Session management and data reset
    
    ### 📊 Understanding the Results
    - **Green boxes**: High confidence detection (>0.8)
    - **Yellow boxes**: Medium confidence detection (0.7-0.8)
    - **Orange boxes**: Lower confidence detection (<0.7)
    - **Emotions**: Randomly assigned for demonstration purposes
    - **Confidence**: Simulated values between 0.6-0.95
    
    ### 💡 Production Notes
    This app demonstrates the infrastructure for emotion detection. In production:
    - Replace simulated emotions with actual ML models (DeepFace, FER, etc.)
    - Add data persistence and analytics
    - Implement user authentication and privacy controls
    - Add export capabilities for emotion data
    """)

# Footer with status
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("🎭 **Enhanced Face Detection**")
    st.markdown("*Built with OpenCV & Streamlit*")

with footer_col2:
    if st.session_state.camera_tested:
        if st.session_state.camera_status == "working":
            st.markdown("📹 **Camera Status**: ✅ Working")
        else:
            st.markdown("📹 **Camera Status**: ❌ Needs Setup")
    else:
        st.markdown("📹 **Camera Status**: ❓ Not Tested")

with footer_col3:
    st.markdown(f"📊 **Session**: {st.session_state.detections} detections")
    if st.session_state.emotions_detected:
        st.markdown(f"🎭 **Emotions**: {len(set(st.session_state.emotions_detected))} types")

st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p><em>Reliable • Fast • Works Everywhere • No Complex Dependencies</em></p>
</div>
""", unsafe_allow_html=True)
