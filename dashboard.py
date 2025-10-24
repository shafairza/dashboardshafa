import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from tensorflow.keras.applications.densenet import preprocess_input

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
    Aplikasi ini dikembangkan oleh **Shafa** untuk mendeteksi dan mengklasifikasikan gambar menggunakan dua model utama:
    
    - 🔍 **YOLOv8**: Model deteksi objek.
    - 🧠 **CNN (DenseNet201)**: Model klasifikasi gambar.
    
    ### 📘 Cara Menggunakan
    1. Buka halaman **Prediksi Model**.
    2. Unggah gambar (.jpg/.jpeg/.png).
    3. Pilih mode deteksi atau klasifikasi.
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
        # DETEKSI OBJEK (YOLO)
        # ==========================
        if menu == "Deteksi Objek (YOLO)":
            st.subheader("🔍 Hasil Deteksi Objek (YOLO)")
            try:
                results = yolo_model(img)
                result_img = results[0].plot()
                st.image(result_img, caption="📦 Hasil Deteksi", use_container_width=True)
            except Exception as e:
                st.error(f"Terjadi kesalahan saat deteksi: {str(e)}")

        # ==========================
        # KLASIFIKASI GAMBAR (DenseNet201)
        # ==========================
        elif menu == "Klasifikasi Gambar":
            st.subheader("🧩 Hasil Klasifikasi Gambar")

            try:
                # Ambil ukuran input model
                input_shape = classifier.input_shape
                if len(input_shape) == 4:
                    target_size = (input_shape[1], input_shape[2])
                else:
                    target_size = (224, 224)  # default DenseNet201

                # Pastikan tidak ada None
                if None in target_size:
                    target_size = (224, 224)

                # Preprocessing sesuai DenseNet201
                img_resized = img.resize(target_size)
                img_array = image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)

                # Prediksi
                prediction = classifier.predict(img_array)
                class_index = np.argmax(prediction)
                confidence = np.max(prediction)

                # Label kelas (⚠️ pastikan urutannya sesuai training)
                class_labels = ["Basmati", "Ipsala", "Arborio", "Karacadag", "Jasmine"]
                predicted_label = class_labels[class_index]

                # Ambang batas
                confidence_threshold = 0.7

                if confidence < confidence_threshold:
                    st.warning("⚠️ Gambar tidak dikenali oleh model. Pastikan gambar sesuai dengan data pelatihan.")
                else:
                    st.success(f"### 🔖 Kelas Prediksi: {predicted_label}")
                    st.write(f"🎯 Probabilitas: {confidence:.2%}")

            except Exception as e:
                st.error(f"Terjadi kesalahan saat klasifikasi: {str(e)}")

    else:
        st.info("⬆ Silakan unggah gambar terlebih dahulu untuk melakukan prediksi.")
