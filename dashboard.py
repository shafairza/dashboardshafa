import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    st.warning("PyTorch tidak tersedia")

try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    st.warning("TensorFlow tidak tersedia")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    st.warning("Ultralytics YOLO tidak tersedia. Install: pip install ultralytics")

st.set_page_config(
    page_title="ML Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS STYLING ---
st.markdown("""
    <style>
    /* Completely hide and disable sidebar collapse button */
    [data-testid="collapsedControl"] {
        display: none !important;
        visibility: hidden !important;
        pointer-events: none !important;
    }

    /* Force sidebar to always be expanded */
    [data-testid="stSidebar"] {
        position: relative !important;
    }

    /* Remove collapse button from DOM */
    section[data-testid="stSidebar"] > div:first-child > button {
        display: none !important;
    }

    /* Hide any button in sidebar header area */
    [data-testid="stSidebar"] > div > div:first-child > div > button {
        display: none !important;
    }
    </style>
""", unsafe_allow_html=True)

def load_css():
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

    /* Global Reset & Base */
    * {
        font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        margin: 0;
        padding: 0;
    }

    /* Root Variables - Purple Theme */
    :root {
        --primary: #a855f7;
        --primary-dark: #9333ea;
        --secondary: #c084fc;
        --background: #0a1929;
        --surface: rgba(168, 85, 247, 0.05);
        --text-primary: #ffffff;
        --text-secondary: #d8b4fe;
        --text-muted: #c084fc;
        --border: rgba(168, 85, 247, 0.3);
        --success: #00e676;
        --error: #ff1744;
        --warning: #ffc400;
        --info: #a855f7;
    }

    /* Main Background - Purple Theme */
    .main {
        background:
            linear-gradient(135deg, #0a1929 0%, #1a0d2e 25%, #2d1b4e 50%, #1e0d3a 75%, #0a1929 100%),
            radial-gradient(ellipse at top left, rgba(168, 85, 247, 0.12) 0%, transparent 50%),
            radial-gradient(ellipse at bottom right, rgba(192, 132, 252, 0.08) 0%, transparent 50%);
        color: var(--text-primary);
        min-height: 100vh;
        position: relative;
        overflow-x: hidden;
    }

    .main::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background:
            radial-gradient(circle at 20% 30%, rgba(168, 85, 247, 0.15) 0%, transparent 60%),
            radial-gradient(circle at 80% 70%, rgba(192, 132, 252, 0.12) 0%, transparent 60%),
            radial-gradient(circle at 50% 50%, rgba(147, 51, 234, 0.08) 0%, transparent 70%);
        pointer-events: none;
        z-index: 0;
        animation: backgroundShift 20s ease-in-out infinite;
    }

    @keyframes backgroundShift {
        0%, 100% { 
            transform: translateX(0) translateY(0) scale(1);
            opacity: 1;
        }
        25% { 
            transform: translateX(-10px) translateY(-5px) scale(1.02);
            opacity: 0.8;
        }
        50% { 
            transform: translateX(5px) translateY(-10px) scale(0.98);
            opacity: 0.9;
        }
        75% { 
            transform: translateX(-5px) translateY(5px) scale(1.01);
            opacity: 0.85;
        }
    }

    .main > div {
        position: relative;
        z-index: 1;
    }

    /* Sidebar - Purple Theme */
    [data-testid="stSidebar"] {
        background: rgba(26, 13, 46, 0.98) !important;
        backdrop-filter: blur(40px) saturate(180%);
        -webkit-backdrop-filter: blur(40px) saturate(180%);
        border-right: 1px solid var(--border);
        box-shadow: 4px 0 24px rgba(168, 85, 247, 0.2);
    }

    [data-testid="stSidebar"] > div {
        background: transparent !important;
        padding: 2rem 1.5rem !important;
    }

    [data-testid="stSidebar"] * {
        color: var(--text-primary) !important;
    }

    /* Typography - Black for main content */
    .main h1 {
        font-weight: 700 !important;
        color: #000000 !important;
        font-size: 2.25rem !important;
        line-height: 1.2 !important;
        letter-spacing: -0.03em !important;
        margin-bottom: 0.5rem !important;
    }

    .main h2 {
        font-weight: 600 !important;
        color: #000000 !important;
        font-size: 1.75rem !important;
        line-height: 1.3 !important;
        letter-spacing: -0.02em !important;
        margin-top: 2rem !important;
    }

    .main h3 {
        font-weight: 600 !important;
        color: #000000 !important;
        font-size: 1.125rem !important;
        line-height: 1.5 !important;
    }

    /* Sidebar headings remain white */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }

    /* Main content text - BLACK */
    .main p,
    .main label,
    .main span,
    .main div {
        color: #000000 !important;
        line-height: 1.6 !important;
    }

    /* Ensure readability for all text in main */
    .main .stMarkdown p,
    .main .stMarkdown span,
    .main .stMarkdown div {
        color: #000000 !important;
    }

    /* Placeholder text */
    input::placeholder {
        color: #666666 !important;
    }

    /* Sidebar text remains white */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        color: #ffffff !important;
    }

    /* Glass Card Base - Ultra Premium */
    .glass-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0.02) 100%) !important;
        backdrop-filter: blur(20px) saturate(200%);
        -webkit-backdrop-filter: blur(20px) saturate(200%);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 24px;
        box-shadow:
            0 12px 40px rgba(0, 0, 0, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.2),
            0 0 0 1px rgba(139, 92, 246, 0.1);
        transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }

    .glass-card:hover {
        border-color: rgba(139, 92, 246, 0.6);
        box-shadow:
            0 20px 60px rgba(139, 92, 246, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.3),
            0 0 0 1px rgba(139, 92, 246, 0.2);
        transform: translateY(-8px) scale(1.02);
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.12) 0%, rgba(255, 255, 255, 0.04) 100%) !important;
    }

    /* Buttons - Purple Theme */
    .stButton > button {
        background: linear-gradient(135deg, rgba(168, 85, 247, 0.9) 0%, rgba(147, 51, 234, 0.9) 100%) !important;
        backdrop-filter: blur(10px);
        color: #ffffff !important;
        font-weight: 600 !important;
        border: 1px solid rgba(168, 85, 247, 0.4) !important;
        border-radius: 14px !important;
        padding: 0.875rem 2rem !important;
        font-size: 0.9375rem !important;
        letter-spacing: 0.01em !important;
        box-shadow:
            0 4px 20px rgba(168, 85, 247, 0.5),
            inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, rgba(192, 132, 252, 1) 0%, rgba(168, 85, 247, 1) 100%) !important;
        box-shadow:
            0 8px 30px rgba(168, 85, 247, 0.7),
            inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
        transform: translateY(-2px) scale(1.02) !important;
        border-color: rgba(192, 132, 252, 0.6) !important;
    }

    /* Radio Button Styling */
    .stRadio > div {
        gap: 0.75rem !important;
        display: flex !important;
        flex-direction: column !important;
    }

    .stRadio > div > label {
        background: var(--surface) !important;
        backdrop-filter: blur(15px);
        border: 1px solid var(--border) !important;
        border-radius: 16px !important;
        padding: 1rem 1.25rem !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer !important;
        position: relative !important;
        overflow: hidden !important;
        min-height: 60px !important;
        height: 60px !important;
        width: 100% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
    }

    .stRadio > div > label::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        width: 4px;
        height: 100%;
        background: transparent;
        transition: all 0.3s ease;
        border-radius: 0 4px 4px 0;
    }

    .stRadio > div > label:hover {
        background: rgba(255, 255, 255, 0.08) !important;
        border-color: var(--primary) !important;
        transform: translateX(4px) !important;
        box-shadow: 0 6px 20px rgba(139, 92, 246, 0.2) !important;
    }

    .stRadio > div > label:hover::before {
        background: linear-gradient(180deg, var(--primary) 0%, var(--secondary) 100%);
    }

    .stRadio > div > label[data-checked="true"] {
        background: linear-gradient(135deg, rgba(168, 85, 247, 0.3) 0%, rgba(147, 51, 234, 0.2) 100%) !important;
        border-color: var(--primary) !important;
        box-shadow: 0 8px 25px rgba(168, 85, 247, 0.4) !important;
        transform: translateX(4px) !important;
    }

    .stRadio > div > label[data-checked="true"]::before {
        background: linear-gradient(180deg, var(--primary) 0%, var(--secondary) 100%);
    }

    /* Hide default radio button circle */
    .stRadio > div > label > div:first-child {
        display: none !important;
    }

    .stRadio > div > label > div {
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        font-size: 1rem !important;
        display: flex !important;
        align-items: center !important;
        gap: 0rem !important;
        margin-left: 0rem !important;
        width: 100% !important;
    }

    /* File Uploader - Glass with Purple Theme */
    [data-testid="stFileUploader"] {
        background: rgba(168, 85, 247, 0.08);
        backdrop-filter: blur(10px);
        border: 2px dashed rgba(168, 85, 247, 0.5);
        border-radius: 20px;
        padding: 3rem 2rem;
        transition: all 0.3s ease;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: rgba(192, 132, 252, 0.7);
        background: rgba(168, 85, 247, 0.12);
        box-shadow: 0 8px 30px rgba(168, 85, 247, 0.3);
    }

    /* File Uploader Text - BLACK */
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] div {
        color: #000000 !important;
        font-weight: 500 !important;
    }

    [data-testid="stFileUploader"] small {
        color: #333333 !important;
    }

    /* Metrics - Glass Style */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(16px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem !important;
        box-shadow:
            0 4px 20px rgba(0, 0, 0, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }

    [data-testid="stMetric"]:hover {
        border-color: rgba(139, 92, 246, 0.3);
        box-shadow:
            0 8px 30px rgba(139, 92, 246, 0.15),
            inset 0 1px 0 rgba(255, 255, 255, 0.15);
        transform: translateY(-2px);
    }

    [data-testid="stMetricValue"] {
        font-weight: 700 !important;
        color: var(--text-primary) !important;
        font-size: 2rem !important;
        letter-spacing: -0.02em !important;
    }

    [data-testid="stMetricLabel"] {
        font-weight: 500 !important;
        color: var(--text-secondary) !important;
        font-size: 0.8125rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        margin-bottom: 0.5rem !important;
    }

    /* Alert Boxes - Glass with Better Visibility */
    .stAlert {
        backdrop-filter: blur(10px);
        border-radius: 14px;
        border: 1px solid;
        padding: 1rem 1.5rem;
        font-weight: 600;
    }

    /* All Alert text should be BLACK */
    .stAlert div,
    .stAlert p,
    .stAlert span {
        color: #000000 !important;
    }

    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%) !important;
    }

    .stProgress > div > div {
        background: rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
    }

    /* Dataframe - Glass */
    .dataframe {
        background: rgba(255, 255, 255, 0.04) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 14px !important;
        overflow: hidden !important;
    }

    .dataframe thead tr {
        background: rgba(139, 92, 246, 0.15) !important;
    }

    .dataframe th {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        font-size: 0.75rem !important;
        letter-spacing: 0.08em !important;
        padding: 1rem !important;
        border-color: rgba(255, 255, 255, 0.08) !important;
    }

    .dataframe td {
        color: var(--text-secondary) !important;
        padding: 0.875rem 1rem !important;
        border-color: rgba(255, 255, 255, 0.05) !important;
    }

    .dataframe tbody tr:hover {
        background: rgba(139, 92, 246, 0.08) !important;
    }

    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Image Container */
    [data-testid="stImage"] {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Divider */
    hr {
        border-color: rgba(255, 255, 255, 0.08) !important;
        margin: 2rem 0 !important;
    }
    </style>
    """

st.markdown(load_css(), unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'total_predictions' not in st.session_state:
    st.session_state.total_predictions = 0
if 'accuracy_score' not in st.session_state:
    st.session_state.accuracy_score = 95.7
if 'task_type' not in st.session_state:
    st.session_state.task_type = "Image Classification"
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Dashboard"

# --- FUNGSI PROCESS IMAGE ---
def process_image(uploaded_file):
    """Fungsi untuk memproses file gambar yang diupload"""
    try:
        image = Image.open(uploaded_file)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    """Memuat model YOLO (PyTorch) dan Keras (TensorFlow)"""
    yolo_model = None
    classifier = None
    
    try:
        if YOLO_AVAILABLE:
            yolo_model = YOLO("models/Shafa_Laporan 4.pt")
            st.success("✅ Model YOLO berhasil dimuat!")
        else:
            st.warning("⚠️ Ultralytics YOLO tidak tersedia")
    except Exception as e:
        st.warning(f"⚠️ Model YOLO tidak dapat dimuat: {e}")

    try:
        if TENSORFLOW_AVAILABLE:
            classifier = tf.keras.models.load_model("models/Shafa_Laporan 2.h5")
            st.success("✅ Model Klasifikasi TensorFlow berhasil dimuat!")
        else:
            st.warning("⚠️ TensorFlow tidak tersedia")
    except Exception as e:
        st.warning(f"⚠️ Model Klasifikasi tidak dapat dimuat: {e}")

    return yolo_model, classifier

# Load models saat aplikasi dimulai
with st.spinner("Loading models..."):
    yolo_model, classifier = load_models()
    MODEL_LOAD_SUCCESS = (yolo_model is not None or classifier is not None)

# --- FUNGSI PREDIKSI KLASIFIKASI ---
def predict_classification(img, classifier):
    """Fungsi prediksi klasifikasi gambar (jenis beras)"""
    
    CLASSIFICATION_LABELS = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]

    if classifier is None:
        st.error("❌ Model Klasifikasi tidak tersedia!")
        simulated_probs = np.random.dirichlet(np.ones(len(CLASSIFICATION_LABELS)))
        simulated_label = CLASSIFICATION_LABELS[np.argmax(simulated_probs)]
        simulated_conf = float(np.max(simulated_probs) * 100)
        return {
            "class": simulated_label,
            "confidence": simulated_conf,
            "probabilities": {label: float(prob * 100) for label, prob in zip(CLASSIFICATION_LABELS, simulated_probs)},
            "task_type": "Classification (Simulated)"
        }

    try:
        target_size = (128, 128)
        img_resized = img.resize(target_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        prediction = classifier.predict(img_array, verbose=0)
        class_index = np.argmax(prediction)
        confidence = float(np.max(prediction) * 100)

        predicted_label = CLASSIFICATION_LABELS[class_index]

        return {
            "class": predicted_label,
            "confidence": confidence,
            "probabilities": {
                label: float(prob * 100) for label, prob in zip(CLASSIFICATION_LABELS, prediction[0])
            },
            "task_type": "Classification"
        }

    except Exception as e:
        st.error(f"Error saat klasifikasi: {str(e)}")
        simulated_probs = np.random.dirichlet(np.ones(len(CLASSIFICATION_LABELS)))
        simulated_label = CLASSIFICATION_LABELS[np.argmax(simulated_probs)]
        simulated_conf = float(np.max(simulated_probs) * 100)
        return {
            "class": simulated_label,
            "confidence": simulated_conf,
            "probabilities": {label: float(prob * 100) for label, prob in zip(CLASSIFICATION_LABELS, simulated_probs)},
            "task_type": "Classification (Simulated)"
        }

# --- FUNGSI PREDIKSI DETEKSI ---
def predict_detection(img, yolo_model):
    """Fungsi prediksi deteksi objek (YOLO)"""
    
    TARGET_DETECTION_CLASSES = ["smoking", "notsmoking"]

    if yolo_model is None:
        st.error("❌ Model YOLO tidak tersedia!")
        simulated_objects = [
            {"class": "smoking", "confidence": 0.93},
            {"class": "notsmoking", "confidence": 0.81},
        ]
        return {
            "objects": simulated_objects,
            "total_objects": len(simulated_objects),
            "task_type": "Detection (Simulated)",
            "annotated_image": np.array(img),
            "target_found": False
        }

    try:
        results = yolo_model(img, conf=0.25)
        class_names = getattr(yolo_model, "names", {})
        detected_objects = []
        target_detections_found = False

        for r in results:
            if hasattr(r, "boxes") and hasattr(r.boxes, "cls"):
                detected_indices = r.boxes.cls.cpu().numpy().tolist()
                detected_class_names = [class_names[int(i)] for i in detected_indices]
                
                for i, cls_name in enumerate(detected_class_names):
                    conf = None
                    if hasattr(r.boxes, "conf"):
                        conf = float(r.boxes.conf[i].cpu().item())
                    
                    detected_objects.append({
                        "class": cls_name,
                        "confidence": conf
                    })

                    if cls_name in TARGET_DETECTION_CLASSES:
                        target_detections_found = True

        annotated_img = results[0].plot() if results else np.array(img)

        return {
            "objects": detected_objects,
            "total_objects": len(detected_objects),
            "task_type": "Detection",
            "target_found": target_detections_found,
            "annotated_image": annotated_img
        }

    except Exception as e:
        st.error(f"Error saat deteksi: {str(e)}")
        simulated_objects = [
            {"class": "smoking", "confidence": 0.93},
            {"class": "notsmoking", "confidence": 0.81},
        ]
        return {
            "objects": simulated_objects,
            "total_objects": len(simulated_objects),
            "task_type": "Detection (Simulated)",
            "annotated_image": np.array(img),
            "target_found": False
        }

# --- FUNGSI PREDICT IMAGE (ROUTER) ---
def predict_image(image, task_type, model_type):
    """Fungsi utama untuk routing prediksi berdasarkan task type"""
    
    if task_type == "Klasifikasi Gambar":
        return predict_classification(image, classifier)
    elif task_type == "Deteksi Objek (YOLO)":
        return predict_detection(image, yolo_model)
    else:
        st.error(f"Task type tidak dikenal: {task_type}")
        return None

# --- FUNGSI VISUALISASI ---
def create_confidence_chart(probabilities):
    sorted_probs = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)[:5]
    categories = [item[0] for item in sorted_probs]
    values = [item[1] for item in sorted_probs]

    colors = ['rgba(168, 85, 247, 0.9)', 'rgba(192, 132, 252, 0.9)', 'rgba(147, 51, 234, 0.9)', 'rgba(216, 180, 254, 0.9)', 'rgba(139, 92, 246, 0.9)']
    
    fig = go.Figure(data=[
        go.Bar(
            x=values,
            y=categories,
            orientation='h',
            marker=dict(
                color=colors[:len(categories)],
                line=dict(color='rgba(255, 255, 255, 0.3)', width=2),
            ),
            text=[f'{v:.1f}%' for v in values],
            textposition='auto',
            textfont=dict(color='white', size=12, family='DM Sans'),
            hovertemplate='<b>%{y}</b><br>Confidence: %{x:.1f}%<extra></extra>',
        )
    ])

    fig.update_layout(
        title={
            'text': 'Confidence Distribution',
            'font': {'size': 18, 'color': '#FFFFFF', 'family': 'DM Sans'},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Confidence (%)',
        yaxis_title='Category',
        font=dict(size=12, color='#B4B4B4', family='DM Sans'),
        plot_bgcolor='rgba(255, 255, 255, 0.03)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=450,
        margin=dict(l=20, r=20, t=80, b=20),
        xaxis=dict(
            range=[0, 100],
            gridcolor='rgba(168, 85, 247, 0.2)',
