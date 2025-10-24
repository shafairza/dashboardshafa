import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/Shafa_Laporan 4.pt")  # Model deteksi objek
    classifier = tf.keras.models.load_model("model/Shafa_Laporan 2.h5")  # Model klasifikasi
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# Sidebar Navigation
# ==========================
st.sidebar.title("📊 Navigasi Dashboard")
page = st.sidebar.radio("Pilih Halaman:", ["Tentang", "Prediksi Model"])

# ==========================
# PAGE 1: Tentang
# ==========================
if page == "Tentang":
    st.title("💡 Tentang Aplikasi Deteksi & Klasifikasi Gambar")

    st.markdown("""
    Aplikasi ini dikembangkan oleh Shafa untuk mendeteksi dan mengklasifikasikan gambar menggunakan dua model utama:
    
    - 🔍 **YOLOv8**: Model deteksi objek yang mengenali objek dalam gambar.  
    - 🧠 **CNN (DenseNet201)**: Model klasifikasi gambar untuk mengidentifikasi kategori spesifik dari citra.
    
    ### 🎯 Tujuan
    - Menyediakan alat bantu interaktif untuk mengenali dan mengklasifikasikan objek secara otomatis.  
    - Meningkatkan efisiensi dalam pengolahan citra berbasis AI.  
    
    ### 📘 Cara Menggunakan
    1. Buka halaman **Prediksi Model** di sidebar.
    2. Unggah gambar (.jpg/.jpeg/.png).
    3. Pilih mode **Deteksi Objek (YOLO)** atau **Klasifikasi Gambar**.
    4. Lihat hasil yang ditampilkan beserta probabilitasnya.
    """)

# ==========================
# PAGE 2: Prediksi Model
# ==========================
elif page == "Prediksi Model":
    st.title("🧠 Prediksi Model Deteksi & Klasifikasi")

    menu = st.sidebar.selectbox("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])

    uploaded_file = st.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="🖼 Gambar yang Diupload", use_container_width=True)

        # ==========================
        # DETEKSI OBJEK
        # ==========================
        if menu == "Deteksi Objek (YOLO)":
            st.subheader("🔍 Hasil Deteksi Objek (YOLO)")
            results = yolo_model(img)
            result_img = results[0].plot()
            st.image(result_img, caption="📦 Hasil Deteksi", use_container_width=True)

        # ==========================
        # KLASIFIKASI GAMBAR
        # ==========================
        elif menu == "Klasifikasi Gambar":
            st.subheader("🧩 Hasil Klasifikasi Gambar")

            try:
                # Gunakan dimensi 250x250 (sesuai model kamu)
                target_size = (250, 250)
                img_resized = img.resize(target_size)
                img_array = image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = img_array / 255.0

                # Prediksi
                prediction = classifier.predict(img_array)
                class_index = np.argmax(prediction)
                confidence = np.max(prediction)

                # Label kelas sesuai urutan model
                class_labels = ["Kelas Basmati", "Kelas Ipsala", "Kelas Arborio", "Kelas Karacadag", "Kelas Jasmine"]
                predicted_label = class_labels[class_index]

                # Batas ambang untuk validasi gambar
                confidence_threshold = 0.7

                if confidence < confidence_threshold:
                    st.warning("⚠️ Gambar tidak dikenali oleh model. Pastikan gambar sesuai dengan data pelatihan.")
                else:
                    st.success(f"### 🔖 Kelas Prediksi: {predicted_label}")
                    st.write(f"🎯 Probabilitas: {confidence:.2%}")

            except Exception as e:
                st.error(f"Terjadi kesalahan saat klasifikasi: {e}")

    else:
        st.info("⬆ Silakan unggah gambar terlebih dahulu untuk melakukan prediksi.")
