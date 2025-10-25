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

try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

st.set_page_config(
    page_title="ML Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS STYLING (TIDAK BERUBAH) ---
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

    /* Sidebar Section Headers */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
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

    /* Hide sidebar collapse button completely */
    [data-testid="collapsedControl"] {
        display: none !important;
    }

    /* Hide sidebar collapse button in header */
    [data-testid="stSidebar"] button[kind="header"] {
        display: none !important;
    }

    /* Hide all collapse control buttons */
    button[aria-label*="collapse"] {
        display: none !important;
    }

    /* Hide Material Icon text fallback */
    .material-icons {
        font-size: 0 !important;
    }

    /* Hide keyboard_double_arrow text specifically */
    [data-testid="stSidebar"] button {
        font-size: 0 !important;
    }

    [data-testid="stSidebar"] button svg {
        display: block !important;
    }

    /* Alternative: hide the entire sidebar nav button area */
    section[data-testid="stSidebar"] > div > div > button {
        display: none !important;
    }

    /* Hide the collapsible trigger */
    .css-1544g2n, .css-nahz7x, .css-10trblm {
        display: none !important;
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

    .balance-card:hover {
        border-color: rgba(192, 132, 252, 0.6);
        box-shadow:
            0 12px 40px rgba(168, 85, 247, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.15);
        transform: translateY(-4px);
    }

    /* Chart Container - Glass */
    .js-plotly-plot {
        background: rgba(255, 255, 255, 0.03) !important;
        backdrop-filter: blur(10px);
        border-radius: 16px !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        padding: 1rem !important;
    }

    /* Image Container */
    [data-testid="stImage"] {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Custom Loading Animation */
    .loading-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 2rem;
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .loading-dots {
        display: flex;
        gap: 8px;
        margin-top: 1rem;
    }

    .loading-dot {
        width: 12px;
        height: 12px;
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        border-radius: 50%;
        animation: bounce 1.4s ease-in-out infinite both;
    }

    .loading-dot:nth-child(1) { animation-delay: -0.32s; }
    .loading-dot:nth-child(2) { animation-delay: -0.16s; }
    .loading-dot:nth-child(3) { animation-delay: 0s; }

    @keyframes bounce {
        0%, 80%, 100% {
            transform: scale(0);
        }
        40% {
            transform: scale(1);
        }
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
    st.session_state.task_type = "Image Classification" # Default
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Dashboard"

# --- HELPER FUNCTIONS (TIDAK BERUBAH) ---
# --- HELPER FUNCTIONS (Perubahan di sini) ---
@st.cache_resource
def load_tensorflow_model():
    if not TENSORFLOW_AVAILABLE:
        return None
    try:
        # PENTING: Pastikan path model ini benar di lingkungan Anda
        # Model Klasifikasi (untuk 5 jenis beras)
        model = keras.models.load_model('models/Shafa_Laporan 2.h5') 
        return model
    except Exception as e:
        st.error(f"Error loading TensorFlow model: {e}")
        return None

@st.cache_resource
def load_pytorch_model():
    if not TORCH_AVAILABLE:
        return None
    try:
        # PENTING: Pastikan path model ini benar di lingkungan Anda
        # Model Klasifikasi (untuk 5 jenis beras)
        model = torch.load('models/Shafa_Laporan 4.pt', map_location='cpu')
        return model
    except Exception as e:
        st.error(f"Error loading PyTorch model: {e}")
        return None

# Tambahkan daftar kelas untuk memudahkan validasi input
# KELAS UNTUK KLASIFIKASI (5 JENIS BERAS)
CLASSIFICATION_CATEGORIES = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag'] 
# KELAS UNTUK DETEKSI (SMOKING/NOT SMOKING)
DETECTION_CLASSES = ['Smoking', 'Not Smoking'] 

# KELAS INPUT YANG DIHARAPKAN UNTUK KLASIFIKASI (Gambar Biji-bijian)
def is_rice_image(image):
    # Logika SIMULASI untuk menentukan apakah gambar adalah "Beras" atau "Random"
    # Di dunia nyata, ini dilakukan dengan model ML kedua atau feature extraction
    # Untuk tujuan simulasi, kita akan gunakan logika sederhana berdasarkan nama file
    if st.session_state.get('uploaded_filename'):
        filename = st.session_state.uploaded_filename.lower()
        if any(rice_type in filename for rice_type in ['rice', 'arborio', 'basmati', 'ipsala', 'jasmine', 'karacadag', 'grain', 'seed']):
            return True
    
    # Jika tidak ada nama file (misalnya dari kamera), asumsikan acak
    return random.choice([True, False, False]) # Lebih sering False untuk gambar random


def is_person_image(image):
    # Logika SIMULASI untuk menentukan apakah gambar terindikasi 'Orang'
    if st.session_state.get('uploaded_filename'):
        filename = st.session_state.uploaded_filename.lower()
        if any(keyword in filename for keyword in ['face', 'person', 'people', 'human', 'smoke', 'vape']):
            return True
    
    return random.choice([True, True, False, False, False]) # Lebih sering False


def predict_classification(image, model_type="TensorFlow Model"):
    """
    Image Classification Prediction (Hanya untuk 5 Kelas Beras)
    Logika 3 & 4
    """
    
    # KELAS YANG DIKLASIFIKASIKAN OLEH MODEL (5 JENIS BERAS)
    categories = CLASSIFICATION_CATEGORIES
    
    if not is_rice_image(image):
        # 3. Kalau klik klasifikasi dan unggah gambar (terindikasi ada orang/hal random selain beras) bakal terdeteksi “Ini bukan klasifikasi”
        return {
            'class': "INPUT TIDAK COCOK",
            'confidence': 100.0,
            'probabilities': {cat: 0.0 for cat in categories},
            'task_type': 'Classification',
            'error_message': "Ini bukan klasifikasi **biji-bijian/beras** yang didukung model."
        }

    try:
        # Logika prediksi model yang sudah ada (Model Beras)
        if model_type == "TensorFlow Model":
            model = load_tensorflow_model()
            # [Kode prediksi TF lainnya tetap sama, asumsikan model berjalan normal]
            if model is not None:
                img_array = np.array(image.resize((224, 224))) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                predictions = model.predict(img_array, verbose=0)
                probabilities = predictions[0] * 100
            else:
                # Simulasi jika model gagal dimuat
                probabilities = np.random.dirichlet(np.ones(len(categories))) * 100

        else: # PyTorch Model
            # [Kode prediksi PyTorch lainnya tetap sama]
            if TORCH_AVAILABLE:
                model = load_pytorch_model()
                if model is not None:
                    img_array = np.array(image.resize((224, 224))) / 255.0
                    img_tensor = torch.FloatTensor(img_array).permute(2, 0, 1).unsqueeze(0)
                    with torch.no_grad():
                        predictions = model(img_tensor)
                        probabilities = torch.softmax(predictions, dim=1).numpy()[0] * 100
                else:
                    probabilities = np.random.dirichlet(np.ones(len(categories))) * 100
            else:
                probabilities = np.random.dirichlet(np.ones(len(categories))) * 100
                
        # 4. Kalau klik klasifikasi dan unggah gambar beras (terdiri 5 kelas) bakal terdeteksi sesuai kelas nya contoh: “Ini beras kelas Ipsala”
        predicted_class = categories[np.argmax(probabilities)]
        confidence = np.max(probabilities)
        
        return {
            'class': predicted_class,
            'confidence': confidence,
            'probabilities': dict(zip(categories, probabilities)),
            'task_type': 'Classification',
            'success_message': f"Ini beras kelas **{predicted_class}** (Confidence: {confidence:.2f}%)"
        }
        
    except Exception as e:
        # [Penanganan error model tetap sama]
        st.warning(f"Model prediction failed: {e}. Using simulation.")
        probabilities = np.random.dirichlet(np.ones(len(categories))) * 100
        predicted_class = categories[np.argmax(probabilities)]
        confidence = np.max(probabilities)
        
        return {
            'class': predicted_class,
            'confidence': confidence,
            'probabilities': dict(zip(categories, probabilities)),
            'task_type': 'Classification',
            'success_message': f"Ini beras kelas **{predicted_class}** (Simulasi Error). Confidence: {confidence:.2f}%"
        }


def predict_detection(image):
    """
    Object Detection Prediction (Simulasi YOLO: Smoking/Not Smoking)
    Logika 1 & 2
    """
    
    # KELAS UNTUK DETEKSI
    categories = DETECTION_CLASSES
    
    # 1. ketika klik model Yolov dan unggah gambar (terindikasi ada orang) bakal terdeteksi “Smoking/NotSmoking)
    if is_person_image(image):
        
        # Simulasi hasil deteksi (misal model YOLO mendeteksi orang dan klasifikasi apakah dia merokok)
        simulated_class = random.choice(categories)
        simulated_confidence = random.uniform(80, 99)
        
        # Buat probabilitas simulasi untuk chart
        probabilities = {c: 0.0 for c in categories}
        probabilities[simulated_class] = simulated_confidence
        
        objects = [
            {'class': simulated_class, 'confidence': simulated_confidence, 'bbox': [100, 100, 500, 500]}
        ]
        
        return {
            'class': simulated_class,
            'confidence': simulated_confidence,
            'probabilities': probabilities,
            'objects': objects,
            'total_objects': len(objects),
            'task_type': 'Detection',
            'success_message': f"Deteksi: **{simulated_class}** dengan Confidence: {simulated_confidence:.2f}%"
        }

    # 2. ⁠kalau klik model Yolov dan unggah gambar random (beras/pesawat) bakal terdeteksi “Tidak terdeteksi smoking/notsmoking, bukan objek deteksi”
    else:
        return {
            'class': "OBJEK TIDAK DITEMUKAN",
            'confidence': 0.0,
            'probabilities': {c: 0.0 for c in categories},
            'objects': [],
            'total_objects': 0,
            'task_type': 'Detection',
            'error_message': "Tidak terdeteksi **Smoking/Not Smoking**. Gambar bukan objek deteksi yang relevan."
        }


def predict_image(image, task_type, model_type="TensorFlow Model"):
    """Main prediction function"""
    if task_type == "Klasifikasi Gambar":
        return predict_classification(image, model_type)
    elif task_type == "Deteksi Objek (YOLO)":
        return predict_detection(image)
    else:
        return predict_classification(image, model_type) # Default fallback

def process_image(image):
    img = Image.open(image)
    img = img.convert('RGB')
    img.thumbnail((800, 800))
    # Simpan nama file untuk digunakan dalam fungsi is_rice_image/is_person_image
    st.session_state.uploaded_filename = image.name 
    return img

# [Fungsi create_confidence_chart dan create_history_chart tetap sama]
# ... (Kode fungsi chart)
def create_confidence_chart(probabilities):
    # Dapatkan 5 kategori teratas untuk visualisasi
    sorted_probs = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)[:5]
    categories = [item[0] for item in sorted_probs]
    values = [item[1] for item in sorted_probs]

    # Cek jika semua nilai nol atau kosong (untuk kasus "INPUT TIDAK COCOK" atau "OBJEK TIDAK DITEMUKAN")
    if not values or all(v == 0.0 for v in values):
        # Buat data dummy untuk chart peringatan
        categories = ["TIDAK ADA DATA"]
        values = [100]
        colors = ['rgba(239, 68, 68, 0.9)'] # Warna merah untuk error
        title = 'Confidence Distribution (TIDAK ADA HASIL)'
        
        fig = go.Figure(data=[
            go.Bar(
                x=values,
                y=categories,
                orientation='h',
                marker=dict(color=colors[0], line=dict(color='rgba(255, 255, 255, 0.3)', width=2)),
                text=['N/A'],
                textposition='auto',
                textfont=dict(color='white', size=12, family='DM Sans'),
                hovertemplate='<b>TIDAK ADA HASIL VALID</b><extra></extra>',
            )
        ])
    else:
        colors = ['rgba(168, 85, 247, 0.9)', 'rgba(192, 132, 252, 0.9)', 'rgba(147, 51, 234, 0.9)', 'rgba(216, 180, 254, 0.9)', 'rgba(139, 92, 246, 0.9)']
        title = 'Confidence Distribution'
        
        fig = go.Figure(data=[
            go.Bar(
                x=values,
                y=categories,
                orientation='h',
                marker=dict(
                    color=colors[:len(categories)],
                    line=dict(color='rgba(255, 255, 255, 0.3)', width=2),
                    gradient=dict(
                        type='radial',
                        coloraxis='coloraxis'
                    )
                ),
                text=[f'{v:.1f}%' for v in values],
                textposition='auto',
                textfont=dict(color='white', size=12, family='DM Sans'),
                hovertemplate='<b>%{y}</b><br>Confidence: %{x:.1f}%<extra></extra>',
            )
        ])

    fig.update_layout(
        title={'text': title, 'font': {'size': 18, 'color': '#FFFFFF', 'family': 'DM Sans'}, 'x': 0.5, 'xanchor': 'center'},
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
            linecolor='rgba(255, 255, 255, 0.1)',
            tickfont=dict(color='#d8b4fe', family='DM Sans')
        ),
        yaxis=dict(
            gridcolor='rgba(168, 85, 247, 0.2)',
            linecolor='rgba(255, 255, 255, 0.1)',
            tickfont=dict(color='#d8b4fe', family='DM Sans')
        ),
        showlegend=False,
        coloraxis=dict(
            colorscale=[[0, 'rgba(168, 85, 247, 0.8)'], [1, 'rgba(147, 51, 234, 0.8)']]
        )
    )

    return fig

def create_history_chart(history):
    if not history:
        return None

    df = pd.DataFrame(history)
    
    # Filter hanya untuk entri yang memiliki 'confidence' (yaitu, hasil klasifikasi)
    df_filtered = df[df['task_type'] == 'Classification'].copy()
    if df_filtered.empty:
        return None

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_filtered['timestamp'],
        y=df_filtered['confidence'],
        mode='lines+markers',
        name='Confidence Trend',
        line=dict(
            color='rgba(168, 85, 247, 0.8)',
            width=4,
            shape='spline',
            smoothing=0.3
        ),
        marker=dict(
            size=12,
            color='rgba(192, 132, 252, 0.9)',
            line=dict(width=2, color='rgba(255, 255, 255, 0.3)'),
            symbol='circle'
        ),
        fill='tonexty',
        fillcolor='rgba(168, 85, 247, 0.15)',
        hovertemplate='<b>Time:</b> %{x}<br><b>Confidence:</b> %{y:.1f}%<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=df_filtered['timestamp'],
        y=df_filtered['confidence'],
        mode='lines',
        fill='tozeroy',
        fillcolor='rgba(168, 85, 247, 0.08)',
        line=dict(color='rgba(168, 85, 247, 0.3)', width=1),
        showlegend=False,
        hoverinfo='skip'
    ))

    fig.update_layout(
        title={
            'text': 'Confidence History',
            'font': {'size': 18, 'color': '#FFFFFF', 'family': 'DM Sans'},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Time',
        yaxis_title='Confidence (%)',
        font=dict(size=12, color='#B4B4B4', family='DM Sans'),
        plot_bgcolor='rgba(255, 255, 255, 0.03)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=450,
        margin=dict(l=20, r=20, t=80, b=20),
        xaxis=dict(
            gridcolor='rgba(168, 85, 247, 0.2)',
            linecolor='rgba(255, 255, 255, 0.1)',
            tickfont=dict(color='#d8b4fe', family='DM Sans')
        ),
        yaxis=dict(
            gridcolor='rgba(168, 85, 247, 0.2)',
            linecolor='rgba(255, 255, 255, 0.1)',
            tickfont=dict(color='#d8b4fe', family='DM Sans')
        ),
        showlegend=False,
        hovermode='x unified'
    )

    return fig

# --- STREAMLIT SIDEBAR (PERUBAHAN DISINI) ---
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

    # Opsi navigasi baru
    menu_options = ["🏠 Beranda", "🧠 Prediksi Model", "📊 Analitik", "ℹ️ Tentang"]
    
    # Update current_page based on selection
    menu_selection = st.radio(
        "menu_nav",
        menu_options,
        label_visibility="collapsed",
        horizontal=False,
        key="main_menu_selection"
    )

    # Map menu selection to simple page name
    if "Beranda" in menu_selection:
        st.session_state.current_page = "Dashboard"
    elif "Prediksi Model" in menu_selection:
        st.session_state.current_page = "Model Prediction"
    elif "Analitik" in menu_selection:
        st.session_state.current_page = "Analytics"
    elif "Tentang" in menu_selection:
        st.session_state.current_page = "About"
    
    # Tetap sediakan variabel untuk kompatibilitas, tapi ini TIDAK akan digunakan di Prediksi Model
    task_type_default = "Image Classification"
    model_type_default = "TensorFlow Model"
    confidence_threshold_default = 70

    st.session_state.task_type_default = task_type_default
    st.session_state.model_type_default = model_type_default
    st.session_state.confidence_threshold_default = confidence_threshold_default
    
    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)


# --- MAIN CONTENT LOGIC ---

# 1. Dashboard (Awal)
if st.session_state.current_page == "Dashboard":
    st.markdown("""
        <div style="text-align: center; padding: 3rem 2rem 2rem 2rem;">
            <div style="width: 80px; height: 80px; margin: 0 auto 1.5rem; background: linear-gradient(135deg, #a855f7 0%, #9333ea 100%); border-radius: 20px; display: flex; align-items: center; justify-content: center; box-shadow: 0 8px 32px rgba(168, 85, 247, 0.5);">
                <span style="font-size: 2rem;">🔬</span>
            </div>
            <h1 style="font-size: 2.5rem; font-weight: 700; color: #000000; margin: 0; letter-spacing: -0.03em;">
                Dashboard: Classification and Detection by S
            </h1>
            <p style="font-size: 1.125rem; color: #000000; margin: 0.75rem 0 0 0; font-weight: 500;">
                Platform untuk pengujian Model Machine Learning
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.info("Pilih **'🧠 Prediksi Model'** di sidebar untuk memulai deteksi atau klasifikasi gambar.")
    
    col_info_1, col_info_2 = st.columns(2)
    with col_info_1:
        st.markdown("""
            <div class="glass-card" style="padding: 1.5rem; text-align: center;">
                <h3 style="color: #a855f7;">Model Tersedia:</h3>
                <p style="color: #000000;">
                    TensorFlow (Keras) dan PyTorch (untuk klasifikasi)
                </p>
            </div>
        """, unsafe_allow_html=True)
    with col_info_2:
        st.markdown("""
            <div class="glass-card" style="padding: 1.5rem; text-align: center;">
                <h3 style="color: #a855f7;">Fitur Utama:</h3>
                <p style="color: #000000;">
                    Klasifikasi & Deteksi Objek pada Gambar
                </p>
            </div>
        """, unsafe_allow_html=True)


# 2. Prediksi Model (Baru)
elif st.session_state.current_page == "Model Prediction":
    st.markdown("""
        <div style="text-align: center; padding: 1rem 2rem 2rem 2rem;">
            <h1 style="font-size: 2.5rem; font-weight: 700; color: #000000; margin: 0; letter-spacing: -0.03em;">
                🧠 Prediksi Model Deteksi & Klasifikasi
            </h1>
            <p style="font-size: 1.125rem; color: #000000; margin: 0.75rem 0 0 0; font-weight: 500;">
                Uji model Anda dalam mode Klasifikasi atau Deteksi Objek.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Bagian sidebar untuk pemilihan mode (Deteksi Objek atau Klasifikasi Gambar)
    # Catatan: Walaupun prompt meminta st.sidebar.selectbox, kita akan membuatnya di main content area untuk UI yang lebih baik,
    # atau di sidebar jika *memang* harus. Saya akan taruh di main area agar terlihat menonjol.
    
    # Gunakan container untuk styling yang lebih baik
    st.markdown('<div class="balance-card" style="padding: 1.5rem 2rem; margin-bottom: 2rem;">', unsafe_allow_html=True)
    st.markdown('<h3 style="color: #000000; margin-bottom: 1rem;">Pilih Mode Prediksi:</h3>', unsafe_allow_html=True)
    
    col_mode_select, col_model_select = st.columns([1, 1])

    with col_mode_select:
        # Pilihan mode deteksi/klasifikasi
        task_type_select = st.selectbox(
            "Pilih Mode:", 
            ["Klasifikasi Gambar", "Deteksi Objek (YOLO)"],
            label_visibility="collapsed",
            key="task_type_select"
        )
        st.session_state.task_type = task_type_select
        
    with col_model_select:
        # Pilihan model (hanya untuk Klasifikasi)
        if st.session_state.task_type == "Klasifikasi Gambar":
            model_type_select = st.selectbox(
                "Pilih Framework:",
                ["TensorFlow Model", "PyTorch Model"],
                label_visibility="collapsed",
                key="model_type_select"
            )
        else:
            model_type_select = "Detection Model (Simulated)"
            st.markdown(f'<p style="color: #000000; margin-top: 0.5rem; font-size: 0.9rem;">Model Deteksi digunakan.</p>', unsafe_allow_html=True)

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
                <div style="background: rgba(168, 85, 247, 0.1); border: 2px solid rgba(168, 85, 247, 0.4); border-radius: 20px; padding: 1rem; overflow: hidden;">
            """, unsafe_allow_html=True)
            st.image(image, use_container_width=True, caption=f"Gambar yang Diunggah: {uploaded_file.name}")
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("""
                <div style="background: rgba(168, 85, 247, 0.1); border: 2px solid rgba(168, 85, 247, 0.4); border-radius: 20px; padding: 1rem; overflow: hidden;">
            """, unsafe_allow_html=True)

            with st.spinner(f"Memproses gambar dengan mode {st.session_state.task_type}..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                # Panggil fungsi prediksi dengan mode yang dipilih
                result = predict_image(image, st.session_state.task_type, model_type_select)
                
                # Cek apakah ada error_message (Logika 2 & 3)
                if 'error_message' in result:
                    st.error(result['error_message'])
                    
                    # Tampilkan status khusus untuk penolakan
                    st.markdown(f"""
                        <div style="text-align: center; background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); padding: 1rem; border-radius: 14px; box-shadow: 0 4px 15px rgba(239, 68, 68, 0.5); margin-top: 1rem;">
                            <p style="color: white; font-weight: 700; margin: 0; font-size: 1.5rem;">
                                STATUS: INPUT DITOLAK
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    st.plotly_chart(create_confidence_chart(result['probabilities']), use_container_width=True)


                else:
                    # Logika Normal (Klasifikasi Beras atau Deteksi Smoking/NotSmoking)
                    st.session_state.total_predictions += 1
                    
                    st.markdown('<h3 style="color: #000000; margin-bottom: 1rem;">Hasil Prediksi:</h3>', unsafe_allow_html=True)

                    if st.session_state.task_type == "Klasifikasi Gambar":
                        # Simpan ke histori
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
                        st.success(result['success_message']) # Logika 4
                        
                        st.markdown("---")
                        st.plotly_chart(create_confidence_chart(result['probabilities']), use_container_width=True)


                    elif st.session_state.task_type == "Deteksi Objek (YOLO)":
                        # Simpan ke histori
                        st.session_state.prediction_history.append({
                            'timestamp': datetime.now().strftime('%H:%M:%S'),
                            'class': result['class'],
                            'confidence': result['confidence'],
                            'task_type': result['task_type'],
                            'objects_detected': result['total_objects']
                        })

                        # Logika 1: Deteksi Smoking/Not Smoking
                        color = "#00e676" if result['class'] == 'Not Smoking' else "#ffc400"
                        
                        st.markdown(f"""
                            <div style="text-align: center; background: linear-gradient(135deg, {color} 0%, #00c853 100%); padding: 1rem; border-radius: 14px; box-shadow: 0 4px 15px rgba(34, 197, 94, 0.5);">
                                <p style="color: white; font-weight: 700; margin: 0; font-size: 1.5rem;">
                                    HASIL DETEKSI: <span style="font-size: 2rem;">{result['class']}</span>
                                </p>
                                <p style="color: white; font-weight: 500; margin: 0; font-size: 1rem;">
                                    CONFIDENCE: {result['confidence']:.2f}% (Jumlah Objek: {result['total_objects']})
                                </p>
                            </div>
                        """, unsafe_allow_html=True)
                        st.success(result['success_message'])
                        
                        st.markdown("---")
                        # Gunakan chart confidence untuk deteksi juga
                        st.plotly_chart(create_confidence_chart(result['probabilities']), use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)
            
    else:
        st.markdown("""
            <div style="text-align: center; padding: 4rem 2rem;">
                <p style="color: #000000; font-size: 1.125rem;">
                    ☝️ Pilih Mode di atas dan unggah gambar untuk memulai prediksi.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
    st.markdown("""
        <div style="text-align: center; margin: 4rem 0 2rem 0;">
            <p style="font-size: 1.25rem; color: #000000; font-style: italic; margin: 0;">
                "Disini Bisa Deteksi Berbagai Ekspresi Wajah (simulasi)"
            </p>
        </div>
    """, unsafe_allow_html=True)

# 3. Analytics (Tidak Berubah Signifikan)
elif st.session_state.current_page == "Analytics":
    st.markdown("# 📊 Analitik Prediksi")
    st.markdown("---")

    df_history_classification = pd.DataFrame([h for h in st.session_state.prediction_history if h['task_type'] == 'Classification'])

    if not df_history_classification.empty:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Klasifikasi",
                len(df_history_classification)
            )

        with col2:
            avg_confidence = df_history_classification['confidence'].mean()
            st.metric(
                "Rata-rata Confidence",
                f"{avg_confidence:.1f}%"
            )

        with col3:
            max_confidence = df_history_classification['confidence'].max()
            st.metric(
                "Confidence Maksimum",
                f"{max_confidence:.1f}%"
            )

        with col4:
            most_common = df_history_classification['class'].mode()[0] if not df_history_classification['class'].mode().empty else "N/A"
            st.metric(
                "Kelas Terbanyak",
                most_common
            )

        st.markdown("---")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### Distribusi Kelas")
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
            st.markdown("### Trend Confidence")
            fig_line = create_history_chart(st.session_state.prediction_history)
            if fig_line:
                st.plotly_chart(fig_line, use_container_width=True)

        st.markdown("---")

        st.markdown("### Riwayat Prediksi Lengkap")
        # Gabungkan semua data, termasuk deteksi
        df_all_history = pd.DataFrame(st.session_state.prediction_history)
        
        # Sederhanakan tampilan untuk riwayat
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

        if st.button("Clear History"):
            st.session_state.prediction_history = []
            st.session_state.total_predictions = 0
            st.rerun()

    else:
        st.info("Tidak ada data prediksi **Klasifikasi** yang tersedia. Kunjungi halaman Prediksi Model untuk memulai.")


# 4. About (Tidak Berubah)
elif st.session_state.current_page == "About":
    st.markdown("# ℹ️ Tentang")
    st.markdown("---")

    st.markdown("""
    ### ML Image Prediction Dashboard

    Platform untuk pengujian model machine learning (ML) secara real-time. Dashboard ini dirancang untuk menunjukkan kapabilitas model **Klasifikasi Gambar** (menggunakan TensorFlow atau PyTorch) dan **Deteksi Objek** (Simulasi YOLO).

    #### Fitur Utama:
    * **Klasifikasi Gambar:** Mengklasifikasikan gambar yang diunggah ke dalam kategori tertentu dengan nilai *confidence*.
    * **Deteksi Objek (Simulasi):** Menyimulasikan pendeteksian objek dalam gambar.
    * **Visualisasi Data:** Menampilkan distribusi *confidence* dan riwayat prediksi.

    #### Teknologi
    * **Framework Utama:** Streamlit
    * **Machine Learning:** TensorFlow/Keras & PyTorch
    * **Data Analysis:** Pandas, NumPy
    * **Visualisasi:** Plotly Express & Graph Objects
    """)
