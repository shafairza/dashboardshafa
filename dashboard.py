import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="ML Dashboard by Shafa",
    page_icon="🧠",
    layout="wide"
)

# ==========================
# CUSTOM CSS
# ==========================
st.markdown("""
    <style>
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #1C1433;
        color: white;
        padding-top: 2rem;
    }

    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p {
        color: white !important;
    }

    /* Sidebar Radio Buttons */
    div[role="radiogroup"] label {
        background-color: #2B2050;
        color: white;
        border-radius: 12px;
        padding: 0.6rem 1rem;
        margin-bottom: 0.4rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        transition: all 0.2s ease-in-out;
    }

    div[role="radiogroup"] label:hover {
        background-color: #4B3D80;
    }

    div[role="radiogroup"] input:checked + div {
        background-color: #6C5DD3 !important;
        color: white !important;
    }

    /* Cards */
    .info-card {
        background-color: white;
        padding: 1.5rem 2rem;
        border-radius: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        margin-top: 1.2rem;
        text-align: center;
    }

    .title-text {
        font-size: 2rem;
        font-weight: 700;
        color: #000000;
    }

    .subtitle {
        font-size: 1.1rem;
        color: #444;
    }

    .highlight {
        color: #6C5DD3;
        font-weight: bold;
    }

    </style>
""", unsafe_allow_html=True)


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
