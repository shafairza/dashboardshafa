import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ==========================
# PAGE CONFIG
# ==========================

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

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/Shafa_Laporan 4.pt")
    classifier = tf.keras.models.load_model("model/Shafa_Laporan 2.h5")
    return yolo_model, classifier

try:
    yolo_model, classifier = load_models()
    MODEL_LOAD_SUCCESS = True
except Exception as e:
    st.error(f"Gagal memuat model: {e}. Prediksi akan disimulasikan.")
    MODEL_LOAD_SUCCESS = False

    class DummyYOLO:
        def __call__(self, img, conf=0.25):
            class DummyBoxes:
                cls = []
            class DummyResults:
                boxes = DummyBoxes()
                def plot(self):
                    return np.array(img.convert('RGB')) 
            return [DummyResults()]
        names = ["smoking", "notsmoking"]

    yolo_model = DummyYOLO()
    classifier = None


# ==========================
# Sidebar
# ==========================
st.sidebar.title("📊 ML Dashboard")
page = st.sidebar.radio("Navigasi:", ["🏠 Beranda", "🧠 Prediksi Model", "📈 Analitik", "ℹ️ Tentang"])


# ==========================
# PAGE 1: BERANDA
# ==========================
if page == "🏠 Beranda":
    st.markdown('<div class="title-text">Dashboard: Classification and Detection by <span class="highlight">S.</span></div>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Platform untuk pengujian Model Machine Learning.</p>', unsafe_allow_html=True)

    st.markdown('<div class="info-card">Pilih prediksi model pada bagian navigasi untuk memulai deteksi atau klasifikasi gambar.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="info-card"><h4>Model Tersedia:</h4><p><span class="highlight">TensorFlow (Keras)</span> dan <span class="highlight">PyTorch</span> untuk klasifikasi</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="info-card"><h4>Fitur Utama:</h4><p>Klasifikasi & Deteksi Objek pada Gambar</p></div>', unsafe_allow_html=True)


# ==========================
# PAGE 2: PREDIKSI MODEL
# ==========================
elif page == "🧠 Prediksi Model":
    st.title("🧠 Prediksi Model Deteksi & Klasifikasi")

    menu = st.selectbox("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])
    uploaded_file = st.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="🖼 Gambar yang Diupload", use_container_width=True)

        if menu == "Deteksi Objek (YOLO)":
            st.subheader("🔍 Hasil Deteksi Objek")
            try:
                results = yolo_model(img, conf=0.25)
                class_names = yolo_model.names
                detected = any(cls in [0, 1] for r in results for cls in getattr(r.boxes, 'cls', []))

                if detected:
                    st.image(results[0].plot(), caption="📦 Hasil Deteksi", use_container_width=True)
                    st.success("✅ Objek 'smoking' atau 'notsmoking' terdeteksi!")
                else:
                    st.warning("⚠️ Tidak ada objek target terdeteksi.")
            except Exception as e:
                st.error(f"Terjadi kesalahan saat deteksi: {e}")

        elif menu == "Klasifikasi Gambar":
            st.subheader("🧩 Hasil Klasifikasi Gambar")

            CLASSIFICATION_LABELS = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]
            if not MODEL_LOAD_SUCCESS or classifier is None:
                st.error("Model Klasifikasi tidak tersedia.")
            else:
                img_resized = img.resize((128, 128))
                img_array = np.expand_dims(image.img_to_array(img_resized) / 255.0, axis=0)
                prediction = classifier.predict(img_array, verbose=0)
                class_index = np.argmax(prediction)
                confidence = np.max(prediction)
                label = CLASSIFICATION_LABELS[class_index]

                if confidence >= 0.7:
                    st.success(f"🔖 Kelas Prediksi: {label}")
                    st.metric("🎯 Probabilitas", f"{confidence:.2%}")
                else:
                    st.warning(f"Model tidak yakin (prediksi tertinggi: {label} - {confidence:.2%})")
    else:
        st.info("⬆ Silakan unggah gambar terlebih dahulu untuk melakukan prediksi.")


# ==========================
# PAGE 3: TENTANG
# ==========================
elif page == "ℹ️ Tentang":
    st.title("💡 Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini dikembangkan oleh **Shafa** untuk mendeteksi dan mengklasifikasikan gambar menggunakan dua model utama:

    - 🔍 **YOLOv8 (Deteksi):** Mendeteksi objek *smoking* dan *notsmoking*.  
    - 🧠 **DenseNet201 (Klasifikasi):** Mengklasifikasikan 5 jenis beras.

    ### Cara Menggunakan
    1. Buka halaman *Prediksi Model*.
    2. Unggah gambar dalam format `.jpg`, `.jpeg`, atau `.png`.
    3. Pilih mode deteksi atau klasifikasi.
    """)
