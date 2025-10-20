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
    yolo_model = YOLO("model/Shafa_Laporan 4.pt")  # Model deteksi objek (Smoking / Not Smoking)
    classifier = tf.keras.models.load_model("model/Shafa_Laporan 2.h5")  # Model klasifikasi gambar
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
    
    - 🔍 **YOLOv8**: Model deteksi objek untuk mengidentifikasi perilaku **Smoking** dan **Not Smoking**.
    - 🧠 **CNN / DenseNet201**: Model klasifikasi gambar dengan kategori:  
      **Kelas Aman, Kelas Api, Kelas Asap, dan Kelas Asap & Api.**
    
    ### 🎯 Tujuan
    - Mengintegrasikan dua pendekatan AI: deteksi perilaku dan klasifikasi kondisi visual.
    - Menyaring input agar sistem tetap akurat sesuai konteks modelnya.
    """)

# ==========================
# PAGE 2: Prediksi Model
# ==========================
elif page == "Prediksi Model":
    st.title("🧠 Prediksi Model Deteksi & Klasifikasi")

    # Pilihan mode
    menu = st.sidebar.selectbox("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])

    uploaded_file = st.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="🖼️ Gambar yang Diupload", use_container_width=True)

        # ==========================
        # DETEKSI OBJEK (YOLO)
        # ==========================
        if menu == "Deteksi Objek (YOLO)":
            st.subheader("🔍 Hasil Deteksi Objek (YOLO)")

            # Jalankan deteksi
            results = yolo_model(img)
            detected_names = results[0].names  # daftar nama kelas dari model
            boxes = results[0].boxes  # hasil deteksi
            detected_classes = [detected_names[int(cls)] for cls in boxes.cls]

            # Daftar kelas yang diizinkan
            allowed_classes = ["Smoking", "Not Smoking"]

            # Filter hasil hanya untuk kelas relevan
            relevant_detections = [c for c in detected_classes if c in allowed_classes]

            if len(relevant_detections) > 0:
                st.success(f"✅ Ditemukan objek relevan: {', '.join(set(relevant_detections))}")
                result_img = results[0].plot()
                st.image(result_img, caption="📦 Hasil Deteksi", use_container_width=True)
            else:
                st.warning("⚠️ Tidak ditemukan objek relevan (Smoking / Not Smoking). "
                           "Kemungkinan gambar tidak sesuai konteks model deteksi.")

        # ==========================
        # KLASIFIKASI GAMBAR (CNN)
        # ==========================
        elif menu == "Klasifikasi Gambar":
            st.subheader("🧩 Hasil Klasifikasi Gambar")

            # Preprocessing
            img_resized = img.resize((224, 224))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            # Prediksi
            prediction = classifier.predict(img_array)
            class_index = np.argmax(prediction)
            confidence = np.max(prediction)

            class_labels = ["Kelas Aman", "Kelas Api", "Kelas Asap", "Kelas Asap dan Api"]
            predicted_label = class_labels[class_index]

            st.write(f"### 🔖 Kelas Prediksi: **{predicted_label}**")
            st.write(f"🎯 Probabilitas: **{confidence:.2%}**")

    else:
        st.info("⬆️ Silakan unggah gambar terlebih dahulu untuk melakukan prediksi.")
