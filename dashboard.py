import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

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
# SIDEBAR NAVIGATION
# ==========================
st.sidebar.title("📊 Navigasi Dashboard")
page = st.sidebar.radio("Pilih Halaman:", ["Tentang", "Prediksi Model"])

st.sidebar.markdown("---")
menu = st.sidebar.selectbox("🧠 Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])

st.sidebar.info("Unggah gambar di halaman *Prediksi Model* untuk melihat hasil deteksi atau klasifikasi.")

# ==========================
# PAGE 1: TENTANG
# ==========================
if page == "Tentang":
    st.title("💡 Informasi Objek Deteksi & Klasifikasi Gambar")
    st.markdown("""
    Aplikasi ini dibuat oleh **Shafa** untuk mendeteksi dan mengklasifikasikan gambar menggunakan dua model:
    
    - 🔍 **YOLOv8** untuk deteksi objek (misalnya mendeteksi apakah gambar mengandung aktivitas merokok atau tidak).
    - 🧠 **DenseNet201 / CNN Classifier** untuk klasifikasi gambar (misalnya membedakan antara api, asap, aman, atau keduanya).
    
    ### 🎯 Tujuan
    - Memberikan sistem cerdas yang mampu mengenali objek penting pada gambar secara otomatis.  
    - Menampilkan hasil prediksi secara **interaktif dan real-time**.
    
    ### 📘 Petunjuk
    1. Buka halaman **Prediksi Model** di sidebar.
    2. Unggah gambar (format `.jpg`, `.jpeg`, atau `.png`).
    3. Pilih mode deteksi (YOLO) atau klasifikasi gambar.
    4. Lihat hasil prediksi dan probabilitasnya.
    """)

# ==========================
# PAGE 2: PREDIKSI MODEL
# ==========================
elif page == "Prediksi Model":
    st.title("🧠 Prediksi Model Deteksi & Klasifikasi")

    uploaded_file = st.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="🖼️ Gambar yang Diupload", use_container_width=True)

        if menu == "Deteksi Objek (YOLO)":
            # ==========================
            # YOLO DETECTION
            # ==========================
            st.subheader("🔍 Hasil Deteksi Objek (YOLO)")
            results = yolo_model(img)
            result_img = results[0].plot()
            st.image(result_img, caption="📦 Deteksi Objek", use_container_width=True)

            detected_labels = [r.names[int(cls)] for r in results for cls in r.boxes.cls]

            # Jika tidak ada objek smoking / not smoking
            valid_labels = ["smoking", "not smoking"]
            if not any(label in valid_labels for label in detected_labels):
                st.warning("🚫 Ini bukan objek deteksi (bukan gambar 'smoking' atau 'not smoking').")
            else:
                st.success(f"✅ Objek terdeteksi: {', '.join(detected_labels)}")

        elif menu == "Klasifikasi Gambar":
            # ==========================
            # IMAGE CLASSIFICATION
            # ==========================
            st.subheader("🧩 Hasil Klasifikasi Gambar")

            # Preprocessing
            img_resized = img.resize((224, 224))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            # Prediksi
            prediction = classifier.predict(img_array)
            class_index = np.argmax(prediction)
            confidence = np.max(prediction)

            # Mapping label kelas
            class_labels = ["Aman", "Api", "Asap", "Asap dan Api"]
            predicted_label = class_labels[class_index]

            st.write(f"### 🔖 Kelas Prediksi: **{predicted_label}**")
            st.write(f"🎯 Probabilitas: **{confidence:.2%}**")

    else:
        st.info("⬆️ Silakan unggah gambar terlebih dahulu untuk melakukan prediksi.")
