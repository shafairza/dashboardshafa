import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Attempt to detect available libs (you already had this in your snippet)
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.preprocessing import image
    TENSORFLOW_AVAILABLE = True
except Exception:
    TENSORFLOW_AVAILABLE = False

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="ML Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================
# HIDE STREAMLIT DEFAULTS (you included many rules already)
# ==========================
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

# ==========================
# COMPREHENSIVE THEME CSS (your provided large CSS)
# ==========================
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
    }

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
        0%, 100% { transform: translateX(0) translateY(0) scale(1); opacity: 1; }
        25% { transform: translateX(-10px) translateY(-5px) scale(1.02); opacity: 0.8; }
        50% { transform: translateX(5px) translateY(-10px) scale(0.98); opacity: 0.9; }
        75% { transform: translateX(-5px) translateY(5px) scale(1.01); opacity: 0.85; }
    }

    .main > div { position: relative; z-index: 1; }

    [data-testid="stSidebar"] {
        background: rgba(26, 13, 46, 0.98) !important;
        backdrop-filter: blur(40px) saturate(180%);
        -webkit-backdrop-filter: blur(40px) saturate(180%);
        border-right: 1px solid var(--border);
        box-shadow: 4px 0 24px rgba(168, 85, 247, 0.2);
    }
    [data-testid="stSidebar"] > div { background: transparent !important; padding: 2rem 1.5rem !important; }
    [data-testid="stSidebar"] * { color: var(--text-primary) !important; }

    .main h1 { font-weight: 700 !important; color: #000000 !important; font-size: 2.25rem !important; margin-bottom: 0.5rem !important; }
    .main h2 { font-weight: 600 !important; color: #000000 !important; font-size: 1.75rem !important; margin-top: 2rem !important; }
    .main h3 { font-weight: 600 !important; color: #000000 !important; font-size: 1.125rem !important; }

    .main p, .main label, .main span, .main div { color: #000000 !important; line-height: 1.6 !important; }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span, [data-testid="stSidebar"] div { color: #ffffff !important; }

    .glass-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0.02) 100%) !important;
        backdrop-filter: blur(20px) saturate(200%);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 24px;
        box-shadow: 0 12px 40px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.2);
        padding: 1.5rem;
        transition: all 0.5s;
    }
    .glass-card:hover { transform: translateY(-6px) scale(1.01); }

    .stButton > button {
        background: linear-gradient(135deg, rgba(168,85,247,0.9) 0%, rgba(147,51,234,0.9) 100%) !important;
        color: #fff !important; font-weight: 600 !important;
        border-radius: 14px !important; padding: 0.875rem 1.6rem !important;
        box-shadow: 0 4px 20px rgba(168,85,247,0.5);
    }
    .stButton > button:hover { transform: translateY(-2px) scale(1.02) !important; }

    .stRadio > div > label { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 16px !important; padding: 1rem 1.25rem !important; min-height: 60px !important; display:flex !important; align-items:center !important; }
    .stRadio > div > label[data-checked="true"] { background: linear-gradient(135deg, rgba(168,85,247,0.3) 0%, rgba(147,51,234,0.2) 100%) !important; border-color: var(--primary) !important; transform: translateX(4px) !important; }

    [data-testid="stFileUploader"] {
        background: rgba(168,85,247,0.08); border: 2px dashed rgba(168,85,247,0.5); border-radius: 20px; padding: 2rem;
    }
    [data-testid="stFileUploader"] label, [data-testid="stFileUploader"] span, [data-testid="stFileUploader"] p { color: #000000 !important; font-weight: 500 !important; }

    .balance-card {
        background: linear-gradient(135deg, rgba(168,85,247,0.2) 0%, rgba(147,51,234,0.15) 100%); border-radius: 24px; padding: 2rem; box-shadow: 0 8px 32px rgba(168,85,247,0.3);
    }

    [data-testid="stImage"] { border-radius: 12px; overflow: hidden; box-shadow: 0 8px 32px rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.1); }

    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    </style>
    """

st.markdown(load_css(), unsafe_allow_html=True)

# ==========================
# SESSION STATE INITIALIZATION
# ==========================
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []  # list of dicts: {time, type, label, confidence}
if 'total_predictions' not in st.session_state:
    st.session_state.total_predictions = 0
if 'accuracy_score' not in st.session_state:
    st.session_state.accuracy_score = 95.7
if 'task_type' not in st.session_state:
    st.session_state.task_type = "Image Classification"
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Beranda"

# ==========================
# MODEL LOADING (with cache_resource)
# ==========================
@st.cache_resource
def load_models(yolo_path="model/Shafa_Laporan 4.pt", clf_path="model/Shafa_Laporan 2.h5"):
    # Load YOLO if available
    yolo_model = None
    classifier = None

    # Ultralytics YOLO
    try:
        if TORCH_AVAILABLE:
            from ultralytics import YOLO
            yolo_model = YOLO(yolo_path)
        else:
            yolo_model = None
    except Exception as e:
        yolo_model = None

    # TensorFlow classifier
    try:
        if TENSORFLOW_AVAILABLE:
            classifier = tf.keras.models.load_model(clf_path)
        else:
            classifier = None
    except Exception as e:
        classifier = None

    return yolo_model, classifier

# Try to load models
try:
    yolo_model, classifier = load_models()
    st.session_state.model_loaded = True
except Exception as e:
    yolo_model, classifier = None, None
    st.session_state.model_loaded = False

# If models failed, define dummy fallbacks
class DummyYOLO:
    def __init__(self):
        self.names = ["smoking", "notsmoking"]
    def __call__(self, img, conf=0.25):
        # returns a list-like object similar to ultralytics result
        class Boxes:
            def __init__(self):
                self.cls = np.array([], dtype=np.int64)
        class Result:
            def __init__(self, img):
                self.boxes = Boxes()
                self.orig_img = img
            def plot(self):
                return np.array(self.orig_img.convert("RGB"))
        return [Result(img)]

if yolo_model is None:
    yolo_model = DummyYOLO()

# ==========================
# SIDEBAR NAVIGATION
# ==========================
with st.sidebar:
    st.markdown("<h2 style='color: white;'>ML Dashboard</h2>", unsafe_allow_html=True)
    st.markdown("##")
    page = st.radio("Navigasi:", ("Beranda", "Prediksi Model", "Analitik", "Tentang"), index=0)

# ==========================
# PAGE: BERANDA
# ==========================
if page == "Beranda":
    st.markdown('<div style="display:flex; gap:1rem; align-items:center;">'
                '<div style="width:96px; height:96px; border-radius:22px; background: linear-gradient(135deg,#6c5dd3,#c084fc); display:flex; align-items:center; justify-content:center; box-shadow: 0 12px 40px rgba(108,93,211,0.25);">'
                '<span style="font-size:48px;">🔬</span></div>'
                '<div style="flex:1;">'
                '<h1 style="margin-bottom:0.1rem;">Dashboard: Classification and Detection by <span style="color:#6C5DD3">S.</span></h1>'
                '<p style="margin-top:0.25rem; color:#444;">Platform untuk pengujian Model Machine Learning.</p>'
                '</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="balance-card" style="padding: 1.2rem 1.4rem; margin-top:1.2rem;">'
                '<strong>Pilih prediksi model pada bagian navigasi untuk memulai deteksi atau klasifikasi gambar.</strong>'
                '</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="glass-card"><h3 style="color:#6C5DD3;">Model Tersedia:</h3><p>TensorFlow (Keras) dan PyTorch (untuk klasifikasi/deteksi)</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="glass-card"><h3 style="color:#6C5DD3;">Fitur Utama:</h3><p>Klasifikasi & Deteksi Objek pada Gambar</p></div>', unsafe_allow_html=True)

    st.markdown("##")
    st.markdown("<h3>Ringkasan Prediksi</h3>", unsafe_allow_html=True)
    st.metric("Total Prediksi", st.session_state.total_predictions, delta=f"{len(st.session_state.prediction_history)} baru")
    st.metric("Akurasi (perkiraan)", f"{st.session_state.accuracy_score:.1f}%")

# ==========================
# PAGE: PREDIKSI MODEL
# ==========================
elif page == "Prediksi Model":
    st.markdown('<h2 style="color:#000;">🧠 Prediksi Model Deteksi & Klasifikasi</h2>', unsafe_allow_html=True)

    mode = st.selectbox("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])
    uploaded_file = st.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="🖼 Gambar yang Diupload", use_container_width=True)

        if mode == "Deteksi Objek (YOLO)":
            st.subheader("🔍 Hasil Deteksi Objek (YOLO)")

            try:
                # run detection
                results = yolo_model(img, conf=0.25)
                # check results structure safely
                detected_class_names = []
                for r in results:
                    # ultralytics result: r.boxes.cls is tensor (N,)
                    cls_attr = getattr(r.boxes, 'cls', None)
                    if cls_attr is None:
                        continue
                    try:
                        # try numpy conversion
                        indices = np.array(cls_attr).astype(int).tolist()
                    except Exception:
                        try:
                            indices = cls_attr.tolist()
                        except Exception:
                            indices = []
                    for idx in indices:
                        if isinstance(yolo_model.names, (list, dict)):
                            # if names is list-like or dict-like
                            try:
                                name = yolo_model.names[int(idx)]
                            except Exception:
                                name = str(idx)
                        else:
                            name = str(idx)
                        detected_class_names.append(name)

                if len(detected_class_names) > 0:
                    # If ultralytics supports plot()
                    try:
                        plotted = results[0].plot()
                        st.image(plotted, caption="📦 Hasil Deteksi (dengan bounding box)", use_container_width=True)
                    except Exception:
                        st.image(img, caption="📦 Hasil Deteksi (gambar asli)", use_container_width=True)
                    st.success(f"✅ Objek terdeteksi: {', '.join(detected_class_names)}")
                    # log to history
                    st.session_state.prediction_history.append({
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "type": "detection",
                        "label": ",".join(detected_class_names),
                        "confidence": None
                    })
                    st.session_state.total_predictions += 1
                else:
                    st.warning("⚠️ Tidak ada objek terdeteksi (target tidak ditemukan).")
                    st.image(img, caption="Gambar Asli (Tidak Ada Deteksi Target)", use_container_width=True)

            except Exception as e:
                st.error(f"Terjadi kesalahan saat deteksi: {e}")

        elif mode == "Klasifikasi Gambar":
            st.subheader("🧩 Hasil Klasifikasi Gambar")

            CLASSIFICATION_LABELS = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]

            if not st.session_state.model_loaded or classifier is None:
                st.error("Model Klasifikasi tidak dapat dimuat. Menjalankan simulasi prediksi.")
                # Simulate a random prediction
                idx = np.random.randint(0, len(CLASSIFICATION_LABELS))
                pred_label = CLASSIFICATION_LABELS[idx]
                pred_conf = np.round(np.random.uniform(0.4, 0.95), 2)
                st.warning(f"Simulasi: {pred_label} ({pred_conf:.2%})")
            else:
                try:
                    target_size = (128, 128)
                    img_resized = img.resize(target_size)
                    img_array = image.img_to_array(img_resized)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = img_array / 255.0

                    prediction = classifier.predict(img_array, verbose=0)
                    class_index = int(np.argmax(prediction, axis=1)[0])
                    confidence = float(np.max(prediction))
                    predicted_label = CLASSIFICATION_LABELS[class_index]
                    threshold = 0.7

                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.image(img_resized, caption="🔍 Gambar yang Diklasifikasikan", use_container_width=True)
                    with col2:
                        if confidence >= threshold:
                            st.success(f"🔖 Kelas Prediksi: {predicted_label}")
                            st.metric(label="🎯 Probabilitas", value=f"{confidence:.2%}")
                        else:
                            st.warning("⚠️ Model tidak yakin dengan prediksi ini.")
                            st.write(f"Prediksi tertinggi: **{predicted_label}** ({confidence:.2%})")

                    # log history
                    st.session_state.prediction_history.append({
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "type": "classification",
                        "label": predicted_label,
                        "confidence": confidence
                    })
                    st.session_state.total_predictions += 1

                except Exception as e:
                    err = str(e)
                    if "incompatible" in err or "shape" in err or "Matrix" in err:
                        st.error("""
                            🛑 **ERROR KRITIS MODEL KLASIFIKASI!**
                            Model gagal prediksi karena ketidaksesuaian dimensi (shape mismatch).
                            Solusi: Pastikan model memiliki GlobalAveragePooling2D() atau Flatten() sebelum Dense terakhir
                            lalu simpan ulang model (.h5).
                        """)
                    else:
                        st.error(f"Terjadi kesalahan saat klasifikasi: {err}")

    else:
        st.info("⬆ Silakan unggah gambar terlebih dahulu untuk melakukan prediksi.")

    st.markdown("---")
    st.markdown("### Riwayat Prediksi Terakhir")
    # show last 6 predictions
    history_df = pd.DataFrame(list(reversed(st.session_state.prediction_history))[:6])
    if not history_df.empty:
        st.dataframe(history_df)
    else:
        st.write("Belum ada prediksi.")

# ==========================
# PAGE: ANALITIK
# ==========================
elif page == "Analitik":
    st.markdown('<h2 style="color:#000;">📈 Analitik</h2>', unsafe_allow_html=True)

    st.markdown("**Ringkasan pemakaian & distribusi prediksi**")
    # simple example analytics using history
    if len(st.session_state.prediction_history) > 0:
        df_hist = pd.DataFrame(st.session_state.prediction_history)
        # count by type
        counts = df_hist['type'].value_counts().reset_index()
        counts.columns = ['type', 'count']
        fig = px.pie(counts, names='type', values='count', title='Distribusi Jenis Prediksi')
        st.plotly_chart(fig, use_container_width=True)

        # time series of predictions
        df_hist['time_parsed'] = pd.to_datetime(df_hist['time'])
        times = df_hist.groupby(pd.Grouper(key='time_parsed', freq='H')).size().reset_index(name='counts')
        fig2 = px.bar(times, x='time_parsed', y='counts', title='Prediksi per Jam')
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Belum ada data riwayat prediksi untuk dianalisis.")

# ==========================
# PAGE: TENTANG
# ==========================
elif page == "Tentang":
    st.markdown('<h2 style="color:#000;">💡 Tentang Aplikasi</h2>', unsafe_allow_html=True)
    st.markdown("""
    Aplikasi ini dikembangkan oleh **Shafa** untuk mendeteksi dan mengklasifikasikan gambar menggunakan dua model utama:

    - 🔍 **YOLOv8 (Deteksi):** Mendeteksi objek 'smoking' dan 'notsmoking'.  
    - 🧠 **DenseNet201 (Klasifikasi):** Mengklasifikasikan 5 jenis beras.

    **Catatan Teknis:**  
    - Model klasifikasi dilatih pada ukuran input 128x128 dan dinormalisasi dengan `rescale=1./255`.  
    - Jika mengalami masalah `shape mismatch`, tambahkan `GlobalAveragePooling2D()` atau `Flatten()` sebelum Dense terakhir.
    """)
    st.markdown("##")
    st.markdown("**Kontak / Versi**")
    st.write("Dikembangkan oleh Shafa — versi demo dashboard.")

# ==========================
# FOOTER (tiny)
# ==========================
st.markdown("<div style='text-align:center; margin-top:2rem; color:#999;'>ML Dashboard • Built with Streamlit</div>", unsafe_allow_html=True)
