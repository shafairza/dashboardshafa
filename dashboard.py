import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# --- PENTING: IMPOR KHUSUS & PENGECEKAN LIBRARY ---
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.preprocessing import image as keras_image_util # Alias untuk menghindari konflik dengan PIL.Image
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
# -------------------------------------------------

st.set_page_config(
    page_title="ML Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS STYLING (Dipertahankan dari Input Anda) ---
def load_css():
    return """
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

    /* Main Background - Purple Theme (Diubah agar teks terlihat) */
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

    /* Typography Adjustment (White text on dark background) */
    .main h1, .main h2, .main h3 {
        color: #ffffff !important;
    }

    .main p, .main label, .main span, .main div {
        color: #ffffff !important;
    }

    .main .stMarkdown p, .main .stMarkdown span, .main .stMarkdown div {
        color: #ffffff !important;
    }
    
    /* Metrics, Alerts, Buttons, etc. styles are retained as they use white text */
    /* ... (The rest of the CSS is retained as-is) ... */

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

    /* Radio Button Styling - Perfect Consistency */
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

    /* Slider Styling */
    .stSlider > div > div > div {
        background: var(--surface) !important;
        border-radius: 8px !important;
        height: 6px !important;
    }

    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%) !important;
    }

    .stSlider > div > div > div > div > div {
        background: #ffffff !important;
        border: 2px solid var(--primary) !important;
        box-shadow: 0 2px 8px rgba(139, 92, 246, 0.3) !important;
        width: 18px !important;
        height: 18px !important;
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

    /* File Uploader Text - WHITE */
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] div {
        color: #ffffff !important;
        font-weight: 500 !important;
    }

    [data-testid="stFileUploader"] small {
        color: #cccccc !important;
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

    .stSuccess {
        background: rgba(34, 197, 94, 0.2) !important;
        color: #ffffff !important;
        border-color: rgba(34, 197, 94, 0.5) !important;
    }

    .stError {
        background: rgba(239, 68, 68, 0.2) !important;
        color: #ffffff !important;
        border-color: rgba(239, 68, 68, 0.5) !important;
    }

    .stInfo {
        background: rgba(168, 85, 247, 0.2) !important;
        color: #ffffff !important;
        border-color: rgba(168, 85, 247, 0.5) !important;
    }

    .stWarning {
        background: rgba(249, 115, 22, 0.2) !important;
        color: #ffffff !important;
        border-color: rgba(249, 115, 22, 0.5) !important;
    }

    /* All Alert text should be WHITE */
    .stAlert div,
    .stAlert p,
    .stAlert span {
        color: #ffffff !important;
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

    .dataframe th {
        color: var(--text-primary) !important;
    }

    .dataframe td {
        color: var(--text-secondary) !important;
    }

    /* Balance/Result Card - Purple Theme */
    .balance-card {
        background: linear-gradient(135deg, rgba(168, 85, 247, 0.2) 0%, rgba(147, 51, 234, 0.15) 100%);
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        border: 1px solid rgba(168, 85, 247, 0.4);
        border-radius: 24px;
        padding: 2.5rem;
        box-shadow:
            0 8px 32px rgba(168, 85, 247, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
        transition: all 0.4s ease;
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
if 'task_type' not in st.session_state:
    st.session_state.task_type = "Klasifikasi Gambar"
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Dashboard"

# --- MODEL LOADING & DUMMY IMPLEMENTATION ---

@st.cache_resource
def load_models():
    # Model deteksi objek (smoking/notsmoking)
    yolo_model = None
    if YOLO_AVAILABLE:
        try:
            yolo_model = YOLO("model/Shafa_Laporan 4.pt")
        except Exception as e:
            st.error(f"Gagal memuat model YOLO: {e}. Menggunakan Dummy YOLO.")
            yolo_model = None
    
    # Model klasifikasi (jenis beras)
    classifier = None
    if TENSORFLOW_AVAILABLE:
        try:
            classifier = tf.keras.models.load_model("model/Shafa_Laporan 2.h5")
        except Exception as e:
            st.error(f"Gagal memuat model TensorFlow: {e}. Menggunakan Dummy Classifier.")
            classifier = None
            
    return yolo_model, classifier

yolo_model, classifier = load_models()
MODEL_LOAD_SUCCESS = (yolo_model is not None) or (classifier is not None)

# Dummy/Placeholder Classes
class DummyYOLO:
    def __call__(self, img, conf=0.25, verbose=False):
        class DummyBoxes:
            cls = torch.tensor([0, 1]) if TORCH_AVAILABLE else np.array([0, 1]) # Simulate detections
        class DummyResults:
            boxes = DummyBoxes()
            def plot(self):
                # Simply return a copy of the input image
                return np.array(img.convert('RGB'))
        return [DummyResults()]
    names = ["smoking", "notsmoking"]

if yolo_model is None:
    yolo_model = DummyYOLO()

if classifier is None:
    class DummyClassifier:
        def predict(self, img_array, verbose=0):
            return np.array([[0.1, 0.2, 0.05, 0.35, 0.3]]) # Simulate low confidence prediction
    classifier = DummyClassifier()

# --- UTILITIES ---

def process_image(uploaded_file):
    """Membuka dan melakukan resize gambar yang diunggah."""
    img = Image.open(uploaded_file)
    img = img.convert('RGB')
    img.thumbnail((1024, 1024))
    return img

def predict_image(img_pil, task_type, model_type_select):
    """Fungsi utama untuk melakukan prediksi berdasarkan mode."""
    
    # ---------------- Deteksi Objek ----------------
    if task_type == "Deteksi Objek (YOLO)":
        TARGET_DETECTION_CLASSES = yolo_model.names
        
        try:
            results = yolo_model(img_pil, conf=0.25, verbose=False)
            result = results[0]
            
            detected_count = 0
            detected_classes = set()
            
            if hasattr(result, 'boxes') and hasattr(result.boxes, 'cls'):
                detected_indices = result.boxes.cls.tolist()
                for i in detected_indices:
                    class_name = yolo_model.names[int(i)]
                    if class_name in TARGET_DETECTION_CLASSES:
                        detected_count += 1
                        detected_classes.add(class_name)
                
                result_img_np = result.plot()
                st.image(result_img_np, caption=f"📦 Hasil Deteksi: {', '.join(detected_classes) or 'Tidak Ditemukan'}", use_container_width=True)
                
                if detected_count > 0:
                    st.success(f"✅ Objek 'smoking' atau 'notsmoking' terdeteksi! Total: {detected_count}")
                else:
                    st.warning("⚠️ Tidak ada objek 'smoking' atau 'notsmoking' terdeteksi (atau confidence di bawah 0.25).")

            else:
                 # Case jika model adalah dummy atau gagal deteksi
                st.warning("⚠️ Gagal menjalankan model Deteksi. Menampilkan gambar asli.")
                st.image(img_pil, caption="Gambar Asli", use_container_width=True)
                detected_count = 0
            
            return {
                'task_type': 'Detection',
                'total_objects': detected_count,
                'detected_classes': list(detected_classes)
            }
            
        except Exception as e:
            st.error(f"Error saat menjalankan deteksi: {e}")
            return {'task_type': 'Detection', 'total_objects': 0, 'detected_classes': []}

    # ---------------- Klasifikasi Gambar ----------------
    elif task_type == "Klasifikasi Gambar":
        CLASSIFICATION_LABELS = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]
        target_size = (128, 128)
        
        if classifier is None:
            st.error("Model Klasifikasi (`Shafa_Laporan 2.h5`) tidak dapat dimuat atau gagal diinisialisasi.")
            return {'task_type': 'Classification', 'class': 'ERROR', 'confidence': 0.0, 'probabilities': {label: 0.0 for label in CLASSIFICATION_LABELS}}
        
        try:
            # Preprocessing
            img_resized = img_pil.resize(target_size)
            # Menggunakan keras_image_util untuk img_to_array
            img_array = keras_image_util.img_to_array(img_resized) 
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0 # Normalisasi
            
            # Prediksi
            prediction = classifier.predict(img_array, verbose=0)
            probabilities = prediction[0] * 100
            class_index = np.argmax(prediction)
            confidence = np.max(prediction) * 100
            predicted_label = CLASSIFICATION_LABELS[class_index]
            
            prob_dict = dict(zip(CLASSIFICATION_LABELS, probabilities))

            return {
                'task_type': 'Classification',
                'class': predicted_label,
                'confidence': confidence,
                'probabilities': prob_dict
            }
        
        except Exception as e:
            error_message = str(e)
            if "Matrix size-incompatible" in error_message:
                st.error(f"🛑 Error Dimensi Model: Model Keras mengharapkan input yang berbeda. Pastikan shape: {img_array.shape} sesuai dengan input layer pertama model Anda.")
            else:
                st.error(f"Terjadi kesalahan saat klasifikasi: {error_message}")
            return {'task_type': 'Classification', 'class': 'ERROR', 'confidence': 0.0, 'probabilities': {label: 0.0 for label in CLASSIFICATION_LABELS}}

# --- CHARTING UTILITIES (Dipertahankan) ---
def create_confidence_chart(probabilities):
    sorted_probs = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)[:5]
    categories = [item[0] for item in sorted_probs]
    values = [item[1] for item in sorted_probs]
    # ... (Plotly chart creation logic as provided in the input) ...
    colors = ['rgba(168, 85, 247, 0.9)', 'rgba(192, 132, 252, 0.9)', 'rgba(147, 51, 234, 0.9)', 'rgba(216, 180, 254, 0.9)', 'rgba(139, 92, 246, 0.9)']
    fig = go.Figure(data=[
        go.Bar(
            x=values,
            y=categories,
            orientation='h',
            marker=dict(
                color=colors[:len(categories)],
                line=dict(color='rgba(255, 255, 255, 0.3)', width=2)
            ),
            text=[f'{v:.1f}%' for v in values],
            textposition='auto',
            textfont=dict(color='white', size=12, family='DM Sans'),
            hovertemplate='<b>%{y}</b><br>Confidence: %{x:.1f}%<extra></extra>',
        )
    ])
    fig.update_layout(
        title={'text': 'Confidence Distribution', 'font': {'size': 18, 'color': '#FFFFFF', 'family': 'DM Sans'}, 'x': 0.5, 'xanchor': 'center'},
        xaxis_title='Confidence (%)',
        yaxis_title='Category',
        font=dict(size=12, color='#B4B4B4', family='DM Sans'),
        plot_bgcolor='rgba(255, 255, 255, 0.03)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=450,
        margin=dict(l=20, r=20, t=80, b=20),
        xaxis=dict(range=[0, 100], gridcolor='rgba(168, 85, 247, 0.2)', linecolor='rgba(255, 255, 255, 0.1)', tickfont=dict(color='#d8b4fe', family='DM Sans')),
        yaxis=dict(gridcolor='rgba(168, 85, 247, 0.2)', linecolor='rgba(255, 255, 255, 0.1)', tickfont=dict(color='#d8b4fe', family='DM Sans')),
        showlegend=False
    )
    return fig

def create_history_chart(history):
    if not history:
        return None
    df = pd.DataFrame([h for h in history if h['task_type'] == 'Classification'])
    if df.empty:
        return None
    
    # ... (Plotly history chart creation logic as provided in the input) ...
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['confidence'],
        mode='lines+markers',
        name='Confidence Trend',
        line=dict(color='rgba(168, 85, 247, 0.8)', width=4, shape='spline', smoothing=0.3),
        marker=dict(size=12, color='rgba(192, 132, 252, 0.9)', line=dict(width=2, color='rgba(255, 255, 255, 0.3)'), symbol='circle'),
        fill='tonexty',
        fillcolor='rgba(168, 85, 247, 0.15)',
        hovertemplate='<b>Time:</b> %{x}<br><b>Confidence:</b> %{y:.1f}%<extra></extra>'
    ))
    fig.update_layout(
        title={'text': 'Confidence History', 'font': {'size': 18, 'color': '#FFFFFF', 'family': 'DM Sans'}, 'x': 0.5, 'xanchor': 'center'},
        xaxis_title='Time',
        yaxis_title='Confidence (%)',
        font=dict(size=12, color='#B4B4B4', family='DM Sans'),
        plot_bgcolor='rgba(255, 255, 255, 0.03)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=450,
        margin=dict(l=20, r=20, t=80, b=20),
        xaxis=dict(gridcolor='rgba(168, 85, 247, 0.2)', linecolor='rgba(255, 255, 255, 0.1)', tickfont=dict(color='#d8b4fe', family='DM Sans')),
        yaxis=dict(gridcolor='rgba(168, 85, 247, 0.2)', linecolor='rgba(255, 255, 255, 0.1)', tickfont=dict(color='#d8b4fe', family='DM Sans')),
        showlegend=False,
        hovermode='x unified'
    )
    return fig

# --- STREAMLIT SIDEBAR ---
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 2rem 0; margin-bottom: 2rem; border-bottom: 1px solid rgba(168, 85, 247, 0.3);'>
            <div style='width: 70px; height: 70px; margin: 0 auto 1.5rem auto; background: linear-gradient(135deg, #a855f7 0%, #9333ea 100%); border-radius: 20px; display: flex; align-items: center; justify-content: center; box-shadow: 0 12px 32px rgba(168, 85, 247, 0.5); position: relative; overflow: hidden;'>
                <div style='position: absolute; top: -50%; right: -50%; width: 200%; height: 200%; background: radial-gradient(circle, rgba(255, 255, 255, 0.2) 0%, transparent 70%); animation: pulse 3s ease-in-out infinite;'></div>
                <span style='font-size: 2rem; font-weight: 700; color: white; z-index: 1;'>🔬</span>
            </div>
            <h2 style='font-size: 1.5rem; font-weight: 700; color: #a855f7; margin: 0; letter-spacing: -0.02em;'>ML Dashboard</h2>
            <div style='width: 40px; height: 2px; background: linear-gradient(90deg, #a855f7 0%, #c084fc 100%); margin: 1rem auto 0 auto; border-radius: 1px;'></div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <p style='font-size: 0.7rem; font-weight: 600; color: #a855f7; text-transform: uppercase; letter-spacing: 0.15em; margin-bottom: 1rem;'>Navigasi:</p>
    """, unsafe_allow_html=True)

    menu_options = ["🏠 Beranda", "🧠 Prediksi Model", "📊 Analitik", "ℹ️ Tentang"]
    
    menu_selection = st.radio(
        "menu_nav",
        menu_options,
        label_visibility="collapsed",
        horizontal=False,
        key="main_menu_selection"
    )

    if "Beranda" in menu_selection:
        st.session_state.current_page = "Dashboard"
    elif "Prediksi Model" in menu_selection:
        st.session_state.current_page = "Model Prediction"
    elif "Analitik" in menu_selection:
        st.session_state.current_page = "Analytics"
    elif "Tentang" in menu_selection:
        st.session_state.current_page = "About"
    
    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)


# --- MAIN CONTENT LOGIC ---

## 🏠 Beranda
if st.session_state.current_page == "Dashboard":
    st.markdown("""
        <div style="text-align: center; padding: 3rem 2rem 2rem 2rem;">
            <div style="width: 80px; height: 80px; margin: 0 auto 1.5rem; background: linear-gradient(135deg, #a855f7 0%, #9333ea 100%); border-radius: 20px; display: flex; align-items: center; justify-content: center; box-shadow: 0 8px 32px rgba(168, 85, 247, 0.5);">
                <span style="font-size: 2rem;">🔬</span>
            </div>
            <h1 style="font-size: 2.5rem; font-weight: 700; color: #ffffff; margin: 0; letter-spacing: -0.03em;">
                Selamat Datang di ML Dashboard
            </h1>
            <p style="font-size: 1.125rem; color: #ffffff; margin: 0.75rem 0 0 0; font-weight: 500;">
                Platform untuk pengujian Model Machine Learning.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.info("Pilih **'🧠 Prediksi Model'** di sidebar untuk memulai deteksi atau klasifikasi gambar.")
    
    col_info_1, col_info_2 = st.columns(2)
    
    tf_status = "✅ Aktif" if TENSORFLOW_AVAILABLE and classifier is not None else "❌ Gagal/Tidak Ada"
    yolo_status = "✅ Aktif" if YOLO_AVAILABLE and yolo_model is not None and not isinstance(yolo_model, DummyYOLO) else "❌ Gagal/Tidak Ada (Menggunakan Dummy)"
    
    with col_info_1:
        st.markdown(f"""
            <div class="glass-card" style="padding: 1.5rem; text-align: center;">
                <h3 style="color: #a855f7;">Status Model Aktif:</h3>
                <p style="color: #ffffff;">
                    Klasifikasi (Keras): <strong>{tf_status}</strong><br>
                    Deteksi (YOLO): <strong>{yolo_status}</strong>
                </p>
            </div>
        """, unsafe_allow_html=True)
    with col_info_2:
        st.markdown("""
            <div class="glass-card" style="padding: 1.5rem; text-align: center;">
                <h3 style="color: #a855f7;">Detail Model:</h3>
                <p style="color: #ffffff;">
                    Klasifikasi: <code>Shafa_Laporan 2.h5</code> (Jenis Beras)<br>
                    Deteksi: <code>Shafa_Laporan 4.pt</code> (Smoking/Notsmoking)
                </p>
            </div>
        """, unsafe_allow_html=True)

---

## 🧠 Prediksi Model
elif st.session_state.current_page == "Model Prediction":
    st.markdown("""
        <div style="text-align: center; padding: 1rem 2rem 2rem 2rem;">
            <h1 style="font-size: 2.5rem; font-weight: 700; color: #ffffff; margin: 0; letter-spacing: -0.03em;">
                🧠 Prediksi Model Deteksi & Klasifikasi
            </h1>
            <p style="font-size: 1.125rem; color: #ffffff; margin: 0.75rem 0 0 0; font-weight: 500;">
                Uji model Anda dalam mode Klasifikasi atau Deteksi Objek.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown('<div class="balance-card" style="padding: 1.5rem 2rem; margin-bottom: 2rem;">', unsafe_allow_html=True)
    st.markdown('<h3 style="color: #ffffff; margin-bottom: 1rem;">Pilih Mode Prediksi:</h3>', unsafe_allow_html=True)
    
    col_mode_select, col_model_select = st.columns([1, 1])

    with col_mode_select:
        task_type_select = st.selectbox(
            "Pilih Mode:", 
            ["Klasifikasi Gambar", "Deteksi Objek (YOLO)"],
            label_visibility="collapsed",
            key="task_type_select"
        )
        st.session_state.task_type = task_type_select
        
    with col_model_select:
        if st.session_state.task_type == "Klasifikasi Gambar":
            # Pilihan model (dibuat dummy karena hanya 1 model keras yang dimuat)
            model_type_select = st.selectbox(
                "Pilih Model (Klasifikasi):",
                ["TensorFlow (Shafa_Laporan 2.h5)"],
                label_visibility="collapsed",
                key="model_type_select"
            )
        else:
            model_type_select = "YOLOv8 (Shafa_Laporan 4.pt)"
            st.markdown(f'<p style="color: #ffffff; margin-top: 0.5rem; font-size: 0.9rem;">Model Deteksi: <b>{model_type_select}</b></p>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Logic for Image Upload and Prediction
    st.markdown("""
        <div style="max-width: 600px; margin: 0 auto 2rem auto;">
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload Image",
        type=['png', 'jpg', 'jpeg'],
        help="Supported formats: PNG, JPG, JPEG (max 200MB)",
        label_visibility="collapsed"
    )

    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is not None:
        image = process_image(uploaded_file)

        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.markdown("""
                <div class="glass-card" style="padding: 1rem; border-color: rgba(168, 85, 247, 0.5); transform: none;">
            """, unsafe_allow_html=True)
            st.image(image, use_container_width=True, caption="🖼️ Gambar yang Diunggah")
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("""
                <div class="glass-card" style="padding: 1.5rem; border-color: rgba(168, 85, 247, 0.5); transform: none;">
            """, unsafe_allow_html=True)

            with st.spinner(f"Memproses gambar dengan mode {st.session_state.task_type}..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                progress_bar.empty()
                
                # Panggil fungsi prediksi
                result = predict_image(image, st.session_state.task_type, model_type_select)
                
                st.session_state.total_predictions += 1
                
                st.markdown('<h3 style="color: #ffffff; margin-bottom: 1rem;">Hasil Prediksi:</h3>', unsafe_allow_html=True)

                if st.session_state.task_type == "Klasifikasi Gambar":
                    
                    st.session_state.prediction_history.append({
                        'timestamp': datetime.now().strftime('%H:%M:%S'),
                        'class': result['class'],
                        'confidence': result['confidence'],
                        'task_type': result['task_type']
                    })

                    st.markdown(f"""
                        <div style="text-align: center; background: linear-gradient(135deg, #a855f7 0%, #9333ea 100%); padding: 1rem; border-radius: 14px; box-shadow: 0 4px 15px rgba(168, 85, 247, 0.5);">
                            <p style="color: white; font-weight: 700; margin: 0; font-size: 1.5rem;">
                                KELAS PREDISKI: <span style="font-size: 2rem;">{result['class']}</span>
                            </p>
                            <p style="color: white; font-weight: 500; margin: 0; font-size: 1rem;">
                                CONFIDENCE: {result['confidence']:.2f}%
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    if result['class'] != 'ERROR':
                         st.plotly_chart(create_confidence_chart(result['probabilities']), use_container_width=True)


                elif st.session_state.task_type == "Deteksi Objek (YOLO)":
                    
                    st.session_state.prediction_history.append({
                        'timestamp': datetime.now().strftime('%H:%M:%S'),
                        'class': 'N/A',
                        'confidence': 0.0,
                        'task_type': result['task_type'],
                        'objects_detected': result['total_objects']
                    })

                    st.markdown(f"""
                        <div style="text-align: center; background: linear-gradient(135deg, #00e676 0%, #00c853 100%); padding: 1rem; border-radius: 14px; box-shadow: 0 4px 15px rgba(34, 197, 94, 0.5);">
                            <p style="color: white; font-weight: 700; margin: 0; font-size: 1.5rem;">
                                TOTAL OBJEK TERDETEKSI: <span style="font-size: 2rem;">{result['total_objects']}</span>
                            </p>
                            <p style="color: white; font-weight: 500; margin: 0; font-size: 1rem;">
                                Ditemukan Kelas: {', '.join(result['detected_classes']) or 'Tidak Ada'}
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    st.info("Visualisasi deteksi objek ditampilkan di atas gambar unggahan.")


            st.markdown("</div>", unsafe_allow_html=True)
            
    else:
        st.markdown("""
            <div class="glass-card" style="text-align: center; padding: 4rem 2rem; transform: none;">
                <p style="color: #ffffff; font-size: 1.125rem;">
                    ☝️ Pilih Mode di atas dan unggah gambar untuk memulai prediksi.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
    st.markdown("""
        <div style="text-align: center; margin: 4rem 0 2rem 0;">
            <p style="font-size: 1.25rem; color: #ffffff; font-style: italic; margin: 0;">
                "Modul Klasifikasi dan Deteksi Objek telah dimuat (atau dicoba dimuat)!"
            </p>
        </div>
    """, unsafe_allow_html=True)

---

## 📊 Analitik
elif st.session_state.current_page == "Analytics":
    st.markdown('<h1 style="color: #ffffff;">📊 Analitik Prediksi</h1>', unsafe_allow_html=True)
    st.markdown("---")

    df_history_classification = pd.DataFrame([h for h in st.session_state.prediction_history if h['task_type'] == 'Classification'])
    df_history_detection = pd.DataFrame([h for h in st.session_state.prediction_history if h['task_type'] == 'Detection'])

    if st.session_state.total_predictions == 0:
        st.info("Tidak ada data prediksi yang tersedia. Kunjungi halaman Prediksi Model untuk membuat prediksi.")
    else:
        # Metrik Gabungan
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Prediksi (All)", st.session_state.total_predictions)

        with col2:
            st.metric("Total Klasifikasi", len(df_history_classification))

        with col3:
            st.metric("Total Deteksi", len(df_history_detection))
        
        with col4:
            if not df_history_classification.empty:
                avg_confidence = df_history_classification['confidence'].mean()
                st.metric("Rata-rata Confidence", f"{avg_confidence:.1f}%")
            else:
                st.metric("Rata-rata Confidence", "N/A")

        st.markdown("---")
        
        # Grafik Klasifikasi
        if not df_history_classification.empty:
            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown('<h3 style="color: #ffffff;">Distribusi Kelas (Klasifikasi)</h3>', unsafe_allow_html=True)
                class_counts = df_history_classification['class'].value_counts()
                fig_pie = px.pie(
                    values=class_counts.values,
                    names=class_counts.index,
                    title="Prediction Distribution (Classification)",
                    color_discrete_sequence=['#a855f7', '#c084fc', '#9333ea', '#d8b4fe', '#7c3aed']
                )
                fig_pie.update_layout(
                    font=dict(size=12, color='#B4B4B4'),
                    title_font=dict(size=16, color='#FFFFFF'),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(30, 25, 45, 0.4)'
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            with col2:
                st.markdown('<h3 style="color: #ffffff;">Trend Confidence</h3>', unsafe_allow_html=True)
                fig_line = create_history_chart(st.session_state.prediction_history)
                if fig_line:
                    st.plotly_chart(fig_line, use_container_width=True)

            st.markdown("---")

        st.markdown('<h3 style="color: #ffffff;">Riwayat Prediksi Lengkap</h3>', unsafe_allow_html=True)
        df_all_history = pd.DataFrame(st.session_state.prediction_history)
        
        if 'objects_detected' in df_all_history.columns:
            df_all_history['Result'] = df_all_history.apply(
                lambda row: f"Class: {row['class']} ({row['confidence']:.2f}%)" if row['task_type'] == 'Classification' 
                else f"Objects Detected: {row['objects_detected']}", axis=1
            )
            df_display = df_all_history[['timestamp', 'task_type', 'Result']].rename(columns={'task_type': 'Mode'})
        else:
            df_display = df_all_history[['timestamp', 'task_type', 'class', 'confidence']].rename(columns={'task_type': 'Mode', 'class': 'Class'})

        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True
        )

        st.markdown('<div style="margin-top: 1rem;">', unsafe_allow_html=True)
        if st.button("Clear History", use_container_width=True):
            st.session_state.prediction_history = []
            st.session_state.total_predictions = 0
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

---

## ℹ️ Tentang
elif st.session_state.current_page == "About":
    st.markdown('<h1 style="color: #ffffff;">ℹ️ Tentang</h1>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("""
        <div class="glass-card" style="padding: 2rem; transform: none;">
            <h3 style="color: #a855f7;">ML Image Prediction Dashboard</h3>

            <p>
            Platform untuk pengujian model machine learning (ML) secara real-time. Dashboard ini dirancang untuk menunjukkan kapabilitas model **Klasifikasi Gambar** (TensorFlow/Keras) dan **Deteksi Objek** (YOLOv8).
            </p>

            <h3 style="color: #a855f7; margin-top: 1.5rem;">Detail Model:</h3>
            <ul>
                <li><strong>Klasifikasi Gambar:</strong> Mengklasifikasikan gambar jenis beras. Model: <code>Shafa_Laporan 2.h5</code>.</li>
                <li><strong>Deteksi Objek (YOLO):</strong> Mendeteksi kelas <code>smoking</code> dan <code>notsmoking</code>. Model: <code>Shafa_Laporan 4.pt</code>.</li>
            </ul>

            <h3 style="color: #a855f7; margin-top: 1.5rem;">Status Lingkungan:</h3>
            <ul>
                <li><strong>TensorFlow/Keras:</strong> {'Tersedia' if TENSORFLOW_AVAILABLE else 'Tidak Tersedia'}</li>
                <li><strong>Ultralytics YOLO:</strong> {'Tersedia' if YOLO_AVAILABLE else 'Tidak Tersedia'}</li>
                <li><strong>Model Berhasil Dimuat:</strong> {'Ya' if MODEL_LOAD_SUCCESS else 'Tidak (Menggunakan Dummy)'}</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
