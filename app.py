import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import time
from pathlib import Path
import requests
from io import BytesIO
import pyttsx3
import threading

# Try importing CV models
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except:
    YOLO_AVAILABLE = False

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except:
    MEDIAPIPE_AVAILABLE = False

# Page Configuration
st.set_page_config(
    page_title="✨ AI Vision Studio",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Cool Pastel Aesthetic Theme (Blues, Lavenders, Mint)
st.markdown("""
<style>
    /* Main Background with Gradient */
    .stApp {
        background: linear-gradient(135deg, #E0F7FA 0%, #E8EAF6 50%, #F3E5F5 100%);
        animation: gradientShift 15s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Glowing Title Animation */
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(45deg, #7986CB, #BA68C8, #4FC3F7, #7986CB);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientAnimation 4s ease infinite;
        text-shadow: 2px 2px 4px rgba(121, 134, 203, 0.3);
        margin-bottom: 1rem;
    }
    
    @keyframes gradientAnimation {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Subtitle */
    .subtitle {
        font-size: 1.3rem;
        text-align: center;
        color: #5C6BC0;
        font-weight: 600;
        margin-bottom: 2rem;
        animation: fadeInUp 1s ease;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Card Styling with Hover Effects */
    .stButton > button {
        background: linear-gradient(135deg, #B3E5FC, #E1BEE7);
        color: #37474F;
        border: none;
        border-radius: 25px;
        padding: 12px 30px;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(129, 212, 250, 0.3);
        transition: all 0.3s ease;
        animation: pulse 2s infinite;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(129, 212, 250, 0.5);
        background: linear-gradient(135deg, #E1BEE7, #B3E5FC);
        color: #37474F;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #E0F7FA, #F3E5F5);
        border-right: 3px solid #B2EBF2;
    }
    
    /* Info Box */
    .stAlert {
        background: rgba(255,255,255,0.9);
        border-left: 5px solid #BA68C8;
        border-radius: 15px;
        animation: slideIn 0.5s ease;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(-50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Metric Cards */
    .metric-card {
        background: rgba(255,255,255,0.95);
        padding: 20px;
        border-radius: 20px;
        box-shadow: 0 8px 25px rgba(121, 134, 203, 0.2);
        border: 2px solid #C5CAE9;
        transition: all 0.3s ease;
        animation: floatIn 0.6s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 35px rgba(121, 134, 203, 0.3);
    }
    
    @keyframes floatIn {
        from {
            opacity: 0;
            transform: scale(0.8);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    /* Select Box */
    .stSelectbox {
        animation: fadeIn 0.8s ease;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    /* File Uploader */
    .stFileUploader {
        background: rgba(255,255,255,0.8);
        padding: 20px;
        border-radius: 15px;
        border: 2px dashed #90CAF9;
        animation: borderPulse 2s infinite;
    }
    
    @keyframes borderPulse {
        0%, 100% { border-color: #90CAF9; }
        50% { border-color: #CE93D8; }
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #7986CB !important;
    }
    
    /* Success Message */
    .success-box {
        background: linear-gradient(135deg, #A5D6A7, #C8E6C9);
        padding: 15px;
        border-radius: 15px;
        color: #1B5E20;
        font-weight: 600;
        text-align: center;
        animation: bounceIn 0.6s ease;
    }
    
    @keyframes bounceIn {
        0% { transform: scale(0); }
        50% { transform: scale(1.1); }
        100% { transform: scale(1); }
    }
    
    /* Sparkle Effect */
    .sparkle {
        animation: sparkle 1.5s infinite;
    }
    
    @keyframes sparkle {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Header decoration */
    .header-decoration {
        text-align: center;
        font-size: 2rem;
        animation: rotate 3s linear infinite;
    }
    
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

# Initialize TTS Engine
def init_tts():
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 0.9)
        return engine
    except:
        return None

# Text to Speech Function
def speak_text(text, engine):
    if engine:
        def speak():
            engine.say(text)
            engine.runAndWait()
        thread = threading.Thread(target=speak)
        thread.start()

# Load Models
@st.cache_resource
def load_yolo_model():
    if YOLO_AVAILABLE:
        try:
            model = YOLO('yolov8n.pt')  # Nano model for speed
            return model
        except:
            return None
    return None

@st.cache_resource
def load_mediapipe_models():
    if MEDIAPIPE_AVAILABLE:
        models = {}
        try:
            models['face_detection'] = mp.solutions.face_detection.FaceDetection(
                min_detection_confidence=0.5
            )
            models['hands'] = mp.solutions.hands.Hands(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            models['pose'] = mp.solutions.pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            return models
        except:
            return {}
    return {}

# Process Image with YOLO
def process_with_yolo(image, model):
    results = model(image)
    annotated_image = results[0].plot()
    
    # Extract detection info
    detections = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            name = model.names[cls]
            detections.append({
                'class': name,
                'confidence': conf
            })
    
    return annotated_image, detections

# Process Image with MediaPipe
def process_with_mediapipe(image, model_type, mp_models):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    annotated_image = image.copy()
    results_data = []
    
    if model_type == 'Face Detection' and 'face_detection' in mp_models:
        results = mp_models['face_detection'].process(image_rgb)
        if results.detections:
            for detection in results.detections:
                mp.solutions.drawing_utils.draw_detection(annotated_image, detection)
                results_data.append('Face detected')
    
    elif model_type == 'Hand Tracking' and 'hands' in mp_models:
        results = mp_models['hands'].process(image_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    mp.solutions.hands.HAND_CONNECTIONS
                )
                results_data.append('Hand detected')
    
    elif model_type == 'Pose Detection' and 'pose' in mp_models:
        results = mp_models['pose'].process(image_rgb)
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp.solutions.pose.POSE_CONNECTIONS
            )
            results_data.append('Pose detected')
    
    return annotated_image, results_data

# Process Video Chunks
def process_video_chunks(video_path, model, model_type, chunk_size=30):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    chunks_data = []
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % chunk_size == 0:
            if model_type == 'YOLO Object Detection' and model:
                _, detections = process_with_yolo(frame, model)
                chunks_data.append({
                    'chunk': frame_idx // chunk_size,
                    'time': frame_idx / fps,
                    'detections': len(detections)
                })
        
        frame_idx += 1
    
    cap.release()
    return chunks_data

# Main App
def main():
    # Animated Header
    st.markdown('<div class="header-decoration">✨🎨✨</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="main-title">AI Vision Studio</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">✨ Experience the Magic of Computer Vision ✨</p>', unsafe_allow_html=True)
    
    # Initialize TTS
    tts_engine = init_tts()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎨 Control Panel")
        st.markdown("---")
        
        # Model Selection
        st.markdown("#### 🤖 Choose Your AI Model")
        model_choice = st.selectbox(
            "",
            ['YOLO Object Detection', 'Face Detection', 'Hand Tracking', 'Pose Detection'],
            key='model_select'
        )
        
        st.markdown("---")
        
        # Input Source
        st.markdown("#### 📥 Input Source")
        input_source = st.radio(
            "",
            ['📷 Camera', '📁 File Upload', '🔗 URL', '🎥 Live Detection'],
            key='input_radio'
        )
        
        st.markdown("---")
        
        # TTS Option
        enable_tts = st.checkbox("🔊 Enable Text-to-Speech", value=False)
        
        st.markdown("---")
        st.markdown("### 🌟 Features")
        st.markdown("""
        - ✅ Real-time Detection
        - ✅ Video Processing
        - ✅ Multiple Input Sources
        - ✅ Text-to-Speech
        - ✅ Beautiful UI
        """)
    
    # Load Models
    yolo_model = None
    mp_models = {}
    
    if model_choice == 'YOLO Object Detection':
        with st.spinner('🎨 Loading YOLO model...'):
            yolo_model = load_yolo_model()
            if not yolo_model:
                st.error("❌ YOLO model not available. Please install ultralytics.")
                return
    else:
        with st.spinner('🎨 Loading MediaPipe models...'):
            mp_models = load_mediapipe_models()
            if not mp_models:
                st.error("❌ MediaPipe not available. Please install mediapipe.")
                return
    
    # Main Content Area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🖼️ Processing Area")
        
        # Input Source Handling
        if input_source == '📁 File Upload':
            uploaded_file = st.file_uploader(
                "Upload Image or Video",
                type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov'],
                key='file_uploader'
            )
            
            if uploaded_file:
                file_type = uploaded_file.type.split('/')[0]
                
                if file_type == 'image':
                    # Process Image
                    image = Image.open(uploaded_file)
                    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    
                    st.image(image, caption='Original Image', use_column_width=True)
                    
                    if st.button('🚀 Analyze Image', key='analyze_btn'):
                        with st.spinner('🎨 Analyzing...'):
                            if model_choice == 'YOLO Object Detection':
                                result_img, detections = process_with_yolo(image_cv, yolo_model)
                                st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), 
                                        caption='Detected Objects', use_column_width=True)
                                
                                # Display Results
                                with col2:
                                    st.markdown("### 📊 Results")
                                    st.markdown(f'<div class="metric-card"><h3>{len(detections)}</h3><p>Objects Detected</p></div>', 
                                              unsafe_allow_html=True)
                                    
                                    if detections:
                                        for det in detections[:5]:
                                            st.metric(det['class'], f"{det['confidence']:.2%}")
                                        
                                        if enable_tts and tts_engine:
                                            text = f"Detected {len(detections)} objects"
                                            speak_text(text, tts_engine)
                            else:
                                result_img, results = process_with_mediapipe(
                                    image_cv, model_choice, mp_models
                                )
                                st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB),
                                        caption=f'{model_choice} Results', use_column_width=True)
                                
                                with col2:
                                    st.markdown("### 📊 Results")
                                    st.markdown(f'<div class="metric-card"><h3>{len(results)}</h3><p>Detections</p></div>',
                                              unsafe_allow_html=True)
                                    
                                    if enable_tts and tts_engine and results:
                                        speak_text(f"Detected {len(results)} items", tts_engine)
                
                elif file_type == 'video':
                    # Process Video
                    st.video(uploaded_file)
                    
                    if st.button('🎬 Process Video Chunks', key='process_video_btn'):
                        # Save video temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                            tmp_file.write(uploaded_file.read())
                            video_path = tmp_file.name
                        
                        with st.spinner('🎨 Processing video chunks...'):
                            if model_choice == 'YOLO Object Detection' and yolo_model:
                                chunks = process_video_chunks(video_path, yolo_model, model_choice)
                                
                                with col2:
                                    st.markdown("### 📊 Video Analysis")
                                    st.markdown(f'<div class="metric-card"><h3>{len(chunks)}</h3><p>Chunks Processed</p></div>',
                                              unsafe_allow_html=True)
                                    
                                    for chunk in chunks[:5]:
                                        st.write(f"⏱️ Time: {chunk['time']:.1f}s - Detected: {chunk['detections']}")
                                    
                                    if enable_tts and tts_engine:
                                        speak_text(f"Processed {len(chunks)} video chunks", tts_engine)
        
        elif input_source == '🔗 URL':
            url = st.text_input("🔗 Enter Image URL", key='url_input')
            
            if url and st.button('🚀 Load & Analyze', key='url_btn'):
                try:
                    response = requests.get(url)
                    image = Image.open(BytesIO(response.content))
                    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    
                    st.image(image, caption='Loaded Image', use_column_width=True)
                    
                    with st.spinner('🎨 Analyzing...'):
                        if model_choice == 'YOLO Object Detection' and yolo_model:
                            result_img, detections = process_with_yolo(image_cv, yolo_model)
                            st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB),
                                    caption='Detected Objects', use_column_width=True)
                            
                            with col2:
                                st.markdown("### 📊 Results")
                                st.markdown(f'<div class="metric-card"><h3>{len(detections)}</h3><p>Objects Detected</p></div>',
                                          unsafe_allow_html=True)
                                
                                if enable_tts and tts_engine:
                                    speak_text(f"Detected {len(detections)} objects", tts_engine)
                except Exception as e:
                    st.error(f"❌ Error loading URL: {str(e)}")
        
        elif input_source == '📷 Camera':
            st.markdown("### 📸 Camera Input")
            camera_image = st.camera_input("Take a picture", key='camera_input')
            
            if camera_image:
                image = Image.open(camera_image)
                image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                if st.button('🚀 Analyze', key='camera_analyze'):
                    with st.spinner('🎨 Analyzing...'):
                        if model_choice == 'YOLO Object Detection' and yolo_model:
                            result_img, detections = process_with_yolo(image_cv, yolo_model)
                            st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB),
                                    caption='Detection Results', use_column_width=True)
                            
                            with col2:
                                st.markdown("### 📊 Results")
                                st.markdown(f'<div class="metric-card"><h3>{len(detections)}</h3><p>Objects</p></div>',
                                          unsafe_allow_html=True)
        
        elif input_source == '🎥 Live Detection':
            st.markdown("### 🎥 Live Detection")
            st.info("💡 Use 'Camera' mode for real-time capture and detection!")
            
            run_live = st.checkbox("▶️ Start Live Detection", key='live_check')
            
            if run_live:
                stframe = st.empty()
                cap = cv2.VideoCapture(0)
                
                stop_button = st.button("⏹️ Stop", key='stop_live')
                
                while run_live and not stop_button:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if model_choice == 'YOLO Object Detection' and yolo_model:
                        result_frame, _ = process_with_yolo(frame, yolo_model)
                        stframe.image(cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB),
                                    channels='RGB', use_column_width=True)
                    else:
                        result_frame, _ = process_with_mediapipe(frame, model_choice, mp_models)
                        stframe.image(cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB),
                                    channels='RGB', use_column_width=True)
                    
                    time.sleep(0.1)
                
                cap.release()
    
    with col2:
        if input_source not in ['📁 File Upload', '🔗 URL']:
            st.markdown("### 💡 Quick Tips")
            st.markdown("""
            <div class="metric-card">
            <p><strong>🎯 Model Info:</strong></p>
            <ul>
            <li>YOLO: Fast object detection</li>
            <li>Face: Detect faces</li>
            <li>Hands: Track hand movements</li>
            <li>Pose: Body pose estimation</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown('<p style="text-align: center; color: #5C6BC0; font-weight: 600;">✨ Built with Love using Free AI Models ✨</p>',
               unsafe_allow_html=True)

if __name__ == "__main__":
    main()
