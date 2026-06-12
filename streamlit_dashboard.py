import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import pandas as pd
from pathlib import Path

# Set Page Config
st.set_page_config(
    page_title="DeepFER - Optimized Facial Emotion Recognition",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Premium Design
st.markdown("""
<style>
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4B79FF, #8FBAFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.25rem;
        color: #555555;
        margin-bottom: 2rem;
    }
    .opt-badge {
        background-color: #e6f0fa;
        color: #0b57d0;
        padding: 4px 8px;
        border-radius: 5px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-right: 5px;
    }
</style>
""", unsafe_allow_html=True)

# App Configuration
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
EMOTION_COLORS = {
    'angry': (255, 0, 0),       # Red
    'disgust': (0, 128, 0),     # Dark Green
    'fear': (128, 0, 128),     # Purple
    'happy': (255, 255, 0),     # Yellow
    'neutral': (128, 128, 128), # Gray
    'sad': (0, 0, 255),         # Blue
    'surprise': (255, 165, 0)   # Orange
}

# Training class weights for prior scaling optimization
CLASS_WEIGHTS = np.array([1.5, 5.0, 1.5, 0.8, 1.2, 1.2, 1.8])

# Sidebar - Settings & Optimizations
st.sidebar.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=80)
st.sidebar.title("DeepFER Controls")
st.sidebar.write("Configure model thresholds and prediction optimizations.")

confidence_threshold = st.sidebar.slider(
    "Min Emotion Confidence",
    min_value=0.0,
    max_value=1.0,
    value=0.30,
    step=0.05
)

st.sidebar.markdown("---")
st.sidebar.subheader("Prediction Optimizations")

enable_clahe = st.sidebar.checkbox(
    "Enable CLAHE Contrast Optimization",
    value=True,
    help="Applies Contrast Limited Adaptive Histogram Equalization to normalize face lighting and shadows before prediction."
)

enable_prior_scaling = st.sidebar.checkbox(
    "Enable Prior Probability Scaling",
    value=True,
    help="Adjusts predictions using class weights to compensate for the dataset imbalance (improves detection of rare classes like disgust/fear)."
)

# Load Models
@st.cache_resource
def load_emotion_model():
    model_path = Path.cwd() / 'models' / 'best_model.h5'
    if not model_path.exists():
        return None
    try:
        model = tf.keras.models.load_model(str(model_path), compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def load_face_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

emotion_model = load_emotion_model()
face_cascade = load_face_cascade()

# Title Section
st.markdown('<div class="main-title">DeepFER Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Optimized Facial Emotion Recognition Engine</div>', unsafe_allow_html=True)

# Status indicators
opt_status_html = ""
if enable_clahe:
    opt_status_html += '<span class="opt-badge">⚡ CLAHE Active</span>'
if enable_prior_scaling:
    opt_status_html += '<span class="opt-badge">⚖️ Prior-Scaling Active</span>'
if opt_status_html:
    st.markdown(f"**Active Optimizations:** {opt_status_html}", unsafe_allow_html=True)

if emotion_model is None:
    st.warning("⚠️ **Trained Model Weights Not Found**: Please ensure that `models/best_model.h5` exists in your workspace directory.")
    
    # Create a dummy model structure for interface rendering
    class DummyModel:
        def __init__(self):
            self.input_shape = (None, 48, 48, 3)
            self.name = "dummy_cnn"
        def predict(self, face_batch, verbose=0):
            probs = np.random.dirichlet(np.ones(7), size=1)
            return probs
    emotion_model = DummyModel()

# Core Processing Functions
def preprocess_face(face_img, model, apply_clahe):
    target_h = model.input_shape[1] if model.input_shape[1] is not None else 48
    target_w = model.input_shape[2] if model.input_shape[2] is not None else 48
    # Read channels from model (e.g. 1 for grayscale, 3 for RGB)
    target_c = model.input_shape[3] if len(model.input_shape) > 3 and model.input_shape[3] is not None else 3
    
    # Resize
    face_resized = cv2.resize(face_img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    
    # Apply CLAHE optimization to handle shadows/lighting
    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # Convert to LAB to equalize only L channel (luminance), keeping colors consistent
        lab = cv2.cvtColor(face_resized, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        face_resized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
    if target_c == 1:
        # Grayscale preprocessing
        face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY).astype('float32')
        face = face_gray / 255.0
        face = np.expand_dims(face, axis=-1)
    else:
        # RGB preprocessing
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB).astype('float32')
        model_name = getattr(model, 'name', '').lower()
        if target_h <= 64:
            face = face_rgb / 255.0
        elif 'resnet' in model_name:
            face = tf.keras.applications.resnet_v2.preprocess_input(face_rgb)
        else:
            face = tf.keras.applications.efficientnet_v2.preprocess_input(face_rgb)
        
    return np.expand_dims(face, axis=0)

def detect_and_classify(image):
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
    
    if len(faces) == 0:
        return img_bgr, None
        
    results = []
    for (x, y, w, h) in faces:
        pad_x = int(w * 0.1)
        pad_y = int(h * 0.1)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(img_bgr.shape[1], x + w + pad_x)
        y2 = min(img_bgr.shape[0], y + h + pad_y)
        
        face_crop = img_bgr[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue
            
        face_input = preprocess_face(face_crop, emotion_model, enable_clahe)
        predictions = emotion_model.predict(face_input, verbose=0)[0]
        
        # Apply prior scaling optimization if enabled
        if enable_prior_scaling:
            predictions = predictions * CLASS_WEIGHTS
            predictions = predictions / np.sum(predictions)  # re-normalize
            
        max_idx = np.argmax(predictions)
        label = EMOTION_LABELS[max_idx]
        conf = predictions[max_idx]
        
        results.append({
            'bbox': (x1, y1, x2, y2),
            'label': label,
            'confidence': conf,
            'probabilities': predictions
        })
        
        color = EMOTION_COLORS.get(label, (0, 255, 0))
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img_bgr, f"{label} ({conf:.2%})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
                    
    return img_bgr, results

# Tabs Interface
tab1, tab2 = st.tabs(["🖼️ Image Upload", "📷 Live Webcam Capture"])

with tab1:
    uploaded_file = st.file_uploader("Choose a facial image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Original & Bounding Box View")
            processed_img_bgr, results = detect_and_classify(image)
            processed_img_rgb = cv2.cvtColor(processed_img_bgr, cv2.COLOR_BGR2RGB)
            st.image(processed_img_rgb, use_container_width=True)
            
        with col2:
            st.subheader("Emotion Analysis Reports")
            if results is None or len(results) == 0:
                st.info("No faces detected in the uploaded image. Please try another photo with clear frontal lighting.")
            else:
                for idx, res in enumerate(results):
                    st.write(f"### Face #{idx + 1}")
                    
                    if res['confidence'] < confidence_threshold:
                        st.warning(f"Detected emotion **{res['label']}** is below the confidence threshold ({res['confidence']:.2%} < {confidence_threshold:.0%})")
                    else:
                        st.success(f"**Primary Emotion Detected**: **{res['label'].upper()}** with **{res['confidence']:.2%}** confidence!")
                    
                    df = pd.DataFrame({
                        'Emotion': EMOTION_LABELS,
                        'Probability': res['probabilities']
                    }).sort_values('Probability', ascending=False)
                    
                    st.bar_chart(df.set_index('Emotion'))

with tab2:
    st.subheader("Capture Image from Webcam")
    webcam_image = st.camera_input("Take a photo")
    
    if webcam_image is not None:
        image = Image.open(webcam_image)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Processed Feed")
            processed_img_bgr, results = detect_and_classify(image)
            processed_img_rgb = cv2.cvtColor(processed_img_bgr, cv2.COLOR_BGR2RGB)
            st.image(processed_img_rgb, use_container_width=True)
            
        with col2:
            st.subheader("Emotion Analysis Reports")
            if results is None or len(results) == 0:
                st.info("No faces detected. Please align your face clearly in front of the camera.")
            else:
                for idx, res in enumerate(results):
                    st.write(f"### Face #{idx + 1}")
                    if res['confidence'] < confidence_threshold:
                        st.warning(f"Detected emotion **{res['label']}** is below confidence threshold ({res['confidence']:.2%} < {confidence_threshold:.0%})")
                    else:
                        st.success(f"**Primary Emotion Detected**: **{res['label'].upper()}** with **{res['confidence']:.2%}** confidence!")
                    
                    df = pd.DataFrame({
                        'Emotion': EMOTION_LABELS,
                        'Probability': res['probabilities']
                    }).sort_values('Probability', ascending=False)
                    
                    st.bar_chart(df.set_index('Emotion'))
