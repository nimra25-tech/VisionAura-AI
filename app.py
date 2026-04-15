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

# ... (the rest of your dreamy pastel app code stays exactly the same) ...
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
    page_icon="🪻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Dreamy Lavender Aesthetic
st.markdown("""
<style>
    /* Main Background with Soft Gradient */
    .stApp {
        background: linear-gradient(135deg, #F3E5F5 0%, #E8EAF6 50%, #E0F7FA 100%);
        animation: gradientShift 15s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Decorative Lavender Borders */
    .lavender-corner-top-left {
        position: fixed; top: 0; left: 0; font-size: 6rem; z-index: 100;
        pointer-events: none; opacity: 0.7; animation: gentleSway 6s ease-in-out infinite;
        transform-origin: top left;
    }
    .lavender-corner-top-right {
        position: fixed; top: 0; right: 0; font-size: 6rem; z-index: 100;
        pointer-events: none; opacity: 0.7; animation: gentleSway 7s ease-in-out infinite alternate;
        transform-origin: top right;
    }
    .floral-divider {
        text-align: center; letter-spacing: 15px; font-size: 1.5rem;
        color: #BA68C8; opacity: 0.8; margin: 10px 0;
    }
    
    @keyframes gentleSway {
        0% { transform: rotate(-3deg); }
        50% { transform: rotate(3deg); }
        100% { transform: rotate(-3deg); }
    }
    
    /* Glowing Title Animation */
    .main-title {
        font-size: 4rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(45deg, #9C27B0, #7986CB, #4FC3F7, #BA68C8);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientAnimation 5s ease infinite;
        text-shadow: 3px 3px 6px rgba(156, 39, 176, 0.2);
        margin-bottom: 0.5rem;
    }
    
    /* Glassmorphism Quote Card */
    .quote-card {
        background: rgba(255, 255, 255, 0.4);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.5);
        border-radius: 20px;
        padding: 25px;
        text-align: center;
        margin: 0 auto 2rem auto;
        max-width: 800px;
        box-shadow: 0 8px 32px rgba(121, 134, 203, 0.15);
        animation: floatIn 1.5s ease;
    }
    
    .quote-text {
        font-size: 1.25rem;
        color: #5C6BC0;
        font-style: italic;
        font-weight: 500;
        line-height: 1.6;
    }
    
    .quote-author {
        display: block;
        margin-top: 10px;
        font-size: 0.9rem;
        color: #7986CB;
        font-weight: 600;
    }
    
    /* Card Styling with Hover Effects */
    .stButton > button {
        background: linear-gradient(135deg, #CE93D8, #9FA8DA);
        color: white;
        border: none;
        border-radius: 30px;
        padding: 12px 30px;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(156, 39, 176, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 25px rgba(156, 39, 176, 0.4);
        background: linear-gradient(135deg, #BA68C8, #7986CB);
        color: white;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, rgba(243,229,245,0.9), rgba(224,247,250,0.9));
        backdrop-filter: blur(10px);
        border-right: 2px solid rgba(255,255,255,0.6);
    }
    
    /* Metric Cards (Glassmorphism) */
    .metric-card {
        background: rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(121, 134, 203, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.5);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        background: rgba(255, 255, 255, 0.8);
        box-shadow: 0 12px 40px rgba(121, 134, 203, 0.25);
    }
    
    /* File Uploader */
    .stFileUploader {
        background: rgba(255,255,255,0.5);
        backdrop-filter: blur(5px);
        padding: 20px;
        border-radius: 20px;
        border: 2px dashed #BA68C8;
    }
</style>

<div class="lavender-corner-top-left">🌿🪻</div>
<div class="lavender-corner-top-right">🪻🌿</div>
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
    # Animated Header & Motivational Quote
    st.markdown('<div class="floral-divider">🪻 ✧ 🌸 ✧ 🪻 ✧ 🌸 ✧ 🪻</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="main-title">AI Vision Studio</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="quote-card">
        <span class="quote-text">"The real magic of discovery lies not in seeking new landscapes, but in having new eyes."</span>
        <span class="quote-author">— Marcel Proust</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize TTS
    tts_engine = init_tts()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🪻 Control Panel")
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
        - 🪻 Real-time Detection
        - 🪻 Video Processing
        - 🪻 Multiple Input Sources
        - 🪻 Text-to-Speech Audio
        """)
    
    # Load Models
    yolo_model = None
    mp_models = {}
    
    if model_choice == 'YOLO Object Detection':
        with st.spinner('✨ Conjuring YOLO model...'):
            yolo_model = load_yolo_model()
            if not yolo_model:
                st.error("❌ YOLO model not available. Please install ultralytics.")
                return
    else:
        with st.spinner('✨ Conjuring MediaPipe models...'):
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
    st.markdown('<div class="floral-divider">🪻 ✧ 🌸 ✧ 🪻</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #7986CB; font-weight: 600;">✨ Built with Magic & AI ✨</p>',
               unsafe_allow_html=True)

if __name__ == "__main__":
    main()
