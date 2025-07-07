import streamlit as st
import cv2
import numpy as np
from fer import FER
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import base64
from io import BytesIO
from PIL import Image

# Page config
st.set_page_config(
    page_title="🎭 Emotion Detector",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .emotion-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        text-align: center;
    }
    .confidence-high { 
        color: #28a745; 
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
    }
    .confidence-medium { 
        color: #ffc107; 
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
    }
    .confidence-low { 
        color: #dc3545; 
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
    }
    .metric-container {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'emotion_history' not in st.session_state:
    st.session_state.emotion_history = []
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
if 'total_detections' not in st.session_state:
    st.session_state.total_detections = 0

# Load emotion detector with caching
@st.cache_resource
def load_emotion_detector():
    """Load and cache the FER emotion detector"""
    return FER(mtcnn=True)

@st.cache_data
def get_emotion_config():
    """Get emotion colors and emojis"""
    return {
        'colors': {
            'angry': '#FF6B6B',
            'disgust': '#4ECDC4', 
            'fear': '#45B7D1',
            'happy': '#96CEB4',
            'sad': '#FFEAA7',
            'surprise': '#DDA0DD',
            'neutral': '#B0B0B0'
        },
        'emojis': {
            'angry': '😠',
            'disgust': '🤢',
            'fear': '😨',
            'happy': '😊',
            'sad': '😢',
            'surprise': '😲',
            'neutral': '😐'
        }
    }

class EmotionAnalyzer:
    def __init__(self):
        self.detector = load_emotion_detector()
        self.config = get_emotion_config()
        
    def analyze_frame(self, frame):
        """Analyze emotions in frame with optimization"""
        try:
            # Resize frame for faster processing
            height, width = frame.shape[:2]
            if width > 640:
                scale = 640 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Detect emotions
            emotions = self.detector.detect_emotions(frame)
            
            if emotions and len(emotions) > 0:
                emotion_data = emotions[0]
                emotion_scores = emotion_data['emotions']
                dominant_emotion = max(emotion_scores, key=emotion_scores.get)
                confidence = emotion_scores[dominant_emotion]
                bbox = emotion_data.get('box', None)
                
                return {
                    'dominant_emotion': dominant_emotion,
                    'confidence': confidence,
                    'all_emotions': emotion_scores,
                    'bbox': bbox,
                    'face_detected': True,
                    'face_count': len(emotions)
                }
            else:
                return {
                    'dominant_emotion': 'No face detected',
                    'confidence': 0.0,
                    'all_emotions': {},
                    'bbox': None,
                    'face_detected': False,
                    'face_count': 0
                }
        except Exception as e:
            return {
                'dominant_emotion': f'Error: {str(e)[:30]}...',
                'confidence': 0.0,
                'all_emotions': {},
                'bbox': None,
                'face_detected': False,
                'face_count': 0
            }
    
    def draw_annotations(self, frame, result):
        """Draw emotion annotations on frame"""
        if not result['face_detected']:
            # Draw error or no detection message
            cv2.putText(frame, result['dominant_emotion'], (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame
        
        emotion = result['dominant_emotion']
        confidence = result['confidence']
        
        # Get confidence-based color
        if confidence >= 0.7:
            color = (0, 255, 0)  # Green
        elif confidence >= 0.4:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 0, 255)  # Red
        
        # Draw bounding box around face
        if result['bbox']:
            x, y, w, h = result['bbox']
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Add emotion label above face
            emoji = self.config['emojis'].get(emotion, '❓')
            label = f"{emoji} {emotion.title()}: {confidence:.2f}"
            
            # Calculate label size and position
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            label_y = max(y - 10, label_size[1] + 10)
            
            # Draw label background
            cv2.rectangle(frame, (x, label_y - label_size[1] - 5), 
                         (x + label_size[0] + 10, label_y + 5), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x + 5, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw main emotion info
        main_label = f"Emotion: {emotion.title()}"
        cv2.putText(frame, main_label, (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        conf_label = f"Confidence: {confidence:.2f}"
        cv2.putText(frame, conf_label, (20, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw confidence bar
        bar_width = int(200 * confidence)
        cv2.rectangle(frame, (20, 80), (220, 100), (255, 255, 255), 2)
        cv2.rectangle(frame, (20, 80), (20 + bar_width, 100), color, -1)
        
        # Draw face count if multiple faces
        if result['face_count'] > 1:
            faces_label = f"Faces detected: {result['face_count']}"
            cv2.putText(frame, faces_label, (20, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame

def create_emotion_chart(emotions_dict):
    """Create a bar chart of current emotion scores"""
    if not emotions_dict:
        return None
    
    config = get_emotion_config()
    emotions_df = pd.DataFrame(list(emotions_dict.items()), 
                              columns=['Emotion', 'Score'])
    emotions_df['Color'] = emotions_df['Emotion'].map(config['colors'])
    emotions_df['Emoji'] = emotions_df['Emotion'].map(config['emojis'])
    emotions_df['Label'] = emotions_df['Emoji'] + ' ' + emotions_df['Emotion'].str.title()
    
    fig = px.bar(emotions_df, 
                 x='Label', 
                 y='Score', 
                 color='Emotion',
                 color_discrete_map=config['colors'],
                 title="Current Emotion Breakdown")
    
    fig.update_layout(
        height=300, 
        showlegend=False,
        xaxis_title="Emotions",
        yaxis_title="Confidence Score",
        yaxis=dict(range=[0, 1])
    )
    return fig

def create_emotion_timeline():
    """Create timeline chart of emotion history"""
    if not st.session_state.emotion_history:
        return None
    
    df = pd.DataFrame(st.session_state.emotion_history)
    config = get_emotion_config()
    
    fig = go.Figure()
    
    for emotion in df['emotion'].unique():
        emotion_data = df[df['emotion'] == emotion]
        fig.add_trace(go.Scatter(
            x=emotion_data['timestamp'],
            y=emotion_data['confidence'],
            mode='lines+markers',
            name=f"{config['emojis'].get(emotion, '❓')} {emotion.title()}",
            line=dict(color=config['colors'].get(emotion, '#000000')),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title="Emotion Timeline",
        xaxis_title="Time",
        yaxis_title="Confidence",
        height=400,
        yaxis=dict(range=[0, 1])
    )
    
    return fig

def save_screenshot(frame, emotion, confidence):
    """Create download link for screenshot"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"emotion_{emotion}_{confidence:.2f}_{timestamp}.jpg"
    
    # Convert frame to PIL Image
    pil_image = Image.fromarray(frame)
    
    # Save to bytes
    img_buffer = BytesIO()
    pil_image.save(img_buffer, format='JPEG', quality=95)
    img_bytes = img_buffer.getvalue()
    
    # Create download button
    st.download_button(
        label="📷 Download Screenshot",
        data=img_bytes,
        file_name=filename,
        mime="image/jpeg"
    )

# Main App Layout
st.markdown('<h1 class="main-header">🎭 FER-Based Emotion Detection</h1>', 
           unsafe_allow_html=True)

# Sidebar controls
with st.sidebar:
    st.header("🎛️ Controls")
    
    # Camera controls
    camera_on = st.checkbox("🎥 Start Camera", key="camera_toggle")
    
    st.divider()
    
    # Settings
    st.subheader("⚙️ Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.1,
                                    help="Minimum confidence to record emotions")
    
    show_bbox = st.checkbox("Show Face Bounding Box", True,
                           help="Draw rectangle around detected faces")
    
    save_history = st.checkbox("Save Emotion History", True,
                              help="Record emotions for analytics")
    
    frame_skip = st.selectbox("Processing Speed", 
                             options=[1, 2, 3, 5], 
                             index=1,
                             help="Higher = faster but less accurate")
    
    max_history = st.slider("Max History Records", 50, 200, 100, 10,
                           help="Maximum emotion records to keep")
    
    st.divider()
    
    # Statistics
    if st.session_state.emotion_history:
        st.subheader("📊 Session Stats")
        df = pd.DataFrame(st.session_state.emotion_history)
        
        # Calculate statistics
        most_common = df['emotion'].mode().iloc[0] if not df.empty else "None"
        avg_confidence = df['confidence'].mean() if not df.empty else 0
        session_duration = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 60 if len(df) > 1 else 0
        
        config = get_emotion_config()
        emoji = config['emojis'].get(most_common, '❓')
        
        st.metric("Most Common Emotion", f"{emoji} {most_common.title()}")
        st.metric("Average Confidence", f"{avg_confidence:.2f}")
        st.metric("Total Detections", len(df))
        st.metric("Session Duration", f"{session_duration:.1f} min")
        
        if st.button("🗑️ Clear History", type="secondary"):
            st.session_state.emotion_history = []
            st.session_state.total_detections = 0
            st.rerun()
    else:
        st.info("No emotion data yet.\nStart camera to begin!")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Live Camera Feed")
    frame_placeholder = st.empty()
    
    # Control buttons
    button_col1, button_col2, button_col3 = st.columns(3)
    
    with button_col1:
        screenshot_requested = st.button("📷 Screenshot", type="secondary")
    
    with button_col2:
        analytics_toggle = st.button("📊 Toggle Analytics", type="secondary")
    
    with button_col3:
        if st.button("🔄 Reset Session", type="secondary"):
            st.session_state.emotion_history = []
            st.session_state.frame_count = 0
            st.session_state.total_detections = 0
            st.success("Session reset!")
            st.rerun()

with col2:
    st.subheader("📈 Real-time Analysis")
    emotion_chart_placeholder = st.empty()
    current_emotion_placeholder = st.empty()

# Initialize analyzer
analyzer = EmotionAnalyzer()

# Camera processing
if camera_on:
    if not st.session_state.camera_active:
        st.session_state.camera_active = True
        camera = cv2.VideoCapture(0)
        
        # Set camera properties for better performance
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 15)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Camera loop
    if st.session_state.camera_active:
        camera = cv2.VideoCapture(0)
        
        # Performance tracking
        start_time = time.time()
        frames_processed = 0
        
        while camera_on and st.session_state.camera_active:
            ret, frame = camera.read()
            
            if not ret:
                st.error("❌ Camera not accessible. Please check permissions.")
                break
            
            st.session_state.frame_count += 1
            
            # Process every nth frame based on frame_skip setting
            if st.session_state.frame_count % frame_skip == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = analyzer.analyze_frame(frame_rgb)
                
                # Draw annotations
                if show_bbox:
                    annotated_frame = analyzer.draw_annotations(frame_rgb.copy(), result)
                else:
                    annotated_frame = frame_rgb.copy()
                    # Just add basic text without bounding box
                    emotion = result['dominant_emotion']
                    confidence = result['confidence']
                    cv2.putText(annotated_frame, f"{emotion}: {confidence:.2f}", 
                               (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display frame
                frame_placeholder.image(annotated_frame, channels="RGB", use_column_width=True)
                
                # Update emotion chart
                if result['all_emotions']:
                    chart = create_emotion_chart(result['all_emotions'])
                    if chart:
                        emotion_chart_placeholder.plotly_chart(chart, use_container_width=True)
                
                # Display current emotion with styling
                emotion = result['dominant_emotion']
                confidence = result['confidence']
                
                if result['face_detected']:
                    # Determine confidence level for styling
                    if confidence >= 0.7:
                        confidence_class = "confidence-high"
                        status_text = "High Confidence"
                    elif confidence >= 0.4:
                        confidence_class = "confidence-medium"
                        status_text = "Medium Confidence"
                    else:
                        confidence_class = "confidence-low"
                        status_text = "Low Confidence"
                    
                    config = get_emotion_config()
                    emoji = config['emojis'].get(emotion, '❓')
                    
                    current_emotion_placeholder.markdown(f"""
                    <div class="emotion-card {confidence_class}">
                        <h2>{emoji} {emotion.title()}</h2>
                        <h4>Confidence: {confidence:.2f}</h4>
                        <p>{status_text}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Save to history
                    if save_history and confidence >= confidence_threshold:
                        st.session_state.emotion_history.append({
                            'timestamp': datetime.now(),
                            'emotion': emotion,
                            'confidence': confidence
                        })
                        st.session_state.total_detections += 1
                        
                        # Limit history size
                        if len(st.session_state.emotion_history) > max_history:
                            st.session_state.emotion_history = st.session_state.emotion_history[-max_history:]
                else:
                    current_emotion_placeholder.markdown(f"""
                    <div class="emotion-card confidence-low">
                        <h3>❓ {emotion}</h3>
                        <p>Position your face clearly in the camera</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Screenshot functionality
                if screenshot_requested and result['face_detected']:
                    save_screenshot(annotated_frame, emotion, confidence)
                
                frames_processed += 1
                
                # Calculate and display FPS every 30 frames
                if frames_processed % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = 30 / elapsed if elapsed > 0 else 0
                    st.sidebar.metric("Performance", f"{fps:.1f} FPS")
                    start_time = time.time()
            
            # Small delay to prevent overwhelming
            time.sleep(0.033)  # ~30 FPS max
            
            # Check if user stopped camera
            if not camera_on:
                break
        
        camera.release()
        st.session_state.camera_active = False
else:
    st.session_state.camera_active = False
    frame_placeholder.info("📷 Click 'Start Camera' in the sidebar to begin emotion detection")
    emotion_chart_placeholder.empty()
    current_emotion_placeholder.empty()

# Analytics section
if st.session_state.get('show_analytics', False) or (st.button("📊 Show Detailed Analytics") if st.session_state.emotion_history else False):
    st.header("📊 Emotion Analytics Dashboard")
    
    if st.session_state.emotion_history:
        df = pd.DataFrame(st.session_state.emotion_history)
        config = get_emotion_config()
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_detections = len(df)
            st.metric("Total Detections", total_detections)
        
        with col2:
            avg_confidence = df['confidence'].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.2f}")
        
        with col3:
            most_common = df['emotion'].mode().iloc[0] if not df.empty else "None"
            emoji = config['emojis'].get(most_common, '❓')
            st.metric("Dominant Emotion", f"{emoji} {most_common.title()}")
        
        with col4:
            session_time = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 60 if len(df) > 1 else 0
            st.metric("Session Time", f"{session_time:.1f} min")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Timeline chart
            timeline_chart = create_emotion_timeline()
            if timeline_chart:
                st.plotly_chart(timeline_chart, use_container_width=True)
        
        with col2:
            # Emotion distribution pie chart
            emotion_counts = df['emotion'].value_counts()
            fig_pie = px.pie(
                values=emotion_counts.values, 
                names=[f"{config['emojis'].get(emotion, '❓')} {emotion.title()}" for emotion in emotion_counts.index],
                title="Emotion Distribution",
                color=emotion_counts.index,
                color_discrete_map=config['colors']
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Confidence analysis
        st.subheader("📈 Confidence Analysis")
        fig_conf = px.histogram(df, x='confidence', bins=20, title="Confidence Distribution")
        fig_conf.update_layout(xaxis_title="Confidence Level", yaxis_title="Frequency")
        st.plotly_chart(fig_conf, use_container_width=True)
        
        # Recent detections table
        st.subheader("📋 Recent Detections")
        display_df = df.tail(20).copy()
        display_df['Time'] = display_df['timestamp'].dt.strftime('%H:%M:%S')
        display_df['Emotion'] = display_df.apply(lambda row: f"{config['emojis'].get(row['emotion'], '❓')} {row['emotion'].title()}", axis=1)
        display_df['Confidence'] = display_df['confidence'].round(3)
        
        st.dataframe(
            display_df[['Time', 'Emotion', 'Confidence']], 
            use_container_width=True,
            hide_index=True
        )
        
        # Export functionality
        st.subheader("📥 Export Data")
        col1, col2 = st.columns(2)
        
        with col1:
            # CSV export
            csv = df.to_csv(index=False)
            st.download_button(
                label="📄 Download CSV",
                data=csv,
                file_name=f"emotion_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # JSON export
            json_data = df.to_json(orient='records', date_format='iso')
            st.download_button(
                label="📄 Download JSON",
                data=json_data,
                file_name=f"emotion_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    else:
        st.info("📊 No emotion data available yet. Start the camera to begin collecting data!")

# Help and information
with st.expander("❓ Help & Usage Guide"):
    st.markdown("""
    ### 🚀 Quick Start
    1. **Click 'Start Camera'** in the sidebar
    2. **Allow camera access** when prompted by your browser
    3. **Position your face** clearly in the camera view
    4. **Watch real-time emotions** appear on screen
    
    ### ⚙️ Settings Explained
    - **Confidence Threshold**: Only emotions above this confidence level are saved to history
    - **Processing Speed**: Higher values process fewer frames (faster but less accurate)
    - **Show Face Bounding Box**: Draws rectangles around detected faces
    - **Save Emotion History**: Records emotions for analytics and charts
    - **Max History Records**: Limits memory usage by keeping only recent detections
    
    ### 📊 Features
    - **Real-time Detection**: Live emotion analysis with confidence scores
    - **Visual Feedback**: Color-coded confidence levels and emoji indicators
    - **Analytics Dashboard**: Charts and statistics of your emotion patterns
    - **Screenshot Capture**: Save moments with emotion data
    - **Data Export**: Download your emotion data as CSV or JSON
    
    ### 🔧 Troubleshooting
    - **Camera not working**: Check browser permissions and refresh the page
    - **Poor accuracy**: Ensure good lighting and clear face visibility
    - **Slow performance**: Increase processing speed or reduce frame quality
    - **No face detected**: Move closer to camera and improve lighting
    
    ### 🎯 Tips for Best Results
    - **Good lighting**: Face the camera with adequate light
    - **Stable position**: Keep your face centered and at a consistent distance
    - **Natural expressions**: Don't force emotions, let them happen naturally
    - **Clean background**: Minimize distractions behind you
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🎭 FER-Based Emotion Detection • Built with Streamlit & Computer Vision</p>
    <p>Real-time emotion analysis using Facial Emotion Recognition (FER) library</p>
</div>
""", unsafe_allow_html=True)
