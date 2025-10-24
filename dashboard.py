import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input
import numpy as np
from PIL import Image

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    try:
        yolo_model = YOLO("model/Shafa_Laporan 4.pt")  # Model deteksi objek
    except Exception as e:
        st.error(f"Gagal memuat model YOLO: {e}")
        yolo_model = None

    try:
        classifier = tf.keras.models.load_model("model/Shafa_Laporan 2.h5")  # Model klasifikasi
    except Exception as e:
        st.error(f"Gagal memuat model klasifikasi: {e}")
        classifier = None

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
    
    - 🔍 **YOLOv8**: Model deteksi objek yang mengenali objek tertentu di dalam gambar.  
    - 🧠 **DenseNet201 (CNN)**: Model klasifikasi gambar yang mengidentifikasi kategori beras dari gambar yang diunggah.
    
    ### 🎯 Tujuan Aplikasi
    - Menyediakan alat bantu interaktif untuk mendeteksi dan mengklasifikasikan objek otomatis.  
    - Meningkatkan efisiensi dalam pengolahan citra berbasis AI.  
    
    ### 📘 Cara Menggunakan
    1. Masuk ke halaman **Prediksi Model** di sidebar.
    2. Unggah gambar berformat `.jpg`, `.jpeg`, atau `.png`.
    3. Pilih mode **Deteksi Objek (YOLO)** atau **Klasifikasi Gambar** di sidebar.
    4. Lihat hasil deteksi atau klasifikasi beserta probabilitasnya.
    """)

# ==========================
# PAGE 2: Prediksi Model
# ==========================
elif page == "Prediksi Model":
    st.title("🧠 Prediksi Model Deteksi & Klasifikasi")

    # Pilihan mode di sidebar
    menu = st.sidebar.selectbox("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])

    uploaded_file = st.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="🖼 Gambar yang Diupload", use_container_width=True)
        except Exception:
            st.error("❌ Gagal memuat gambar. Pastikan file valid.")
            st.stop()

        # ==========================
        # DETEKSI OBJEK
        # ==========================
        if menu == "Deteksi Objek (YOLO)":
            if yolo_model is None:
                st.error("Model YOLO belum dimuat.")
            else:
                st.subheader("🔍 Hasil Deteksi Objek (YOLO)")
                results = yolo_model(img)
                result_img = results[0].plot()
                st.image(result_img, caption="📦 Hasil Deteksi", use_container_width=True)

        # ==========================
        # KLASIFIKASI GAMBAR
        # ==========================
        elif menu == "Klasifikasi Gambar":
            if classifier is None:
                st.error("Model klasifikasi belum dimuat.")
            else:
                st.subheader("🧩 Hasil Klasifikasi Gambar")

                # --- Preprocessing sesuai DenseNet201 ---
                img_resized = img.resize((224, 224))
                img_array = image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)

                # --- Prediksi ---
                try:
                    prediction = classifier.predict(img_array)
                except Exception as e:
                    st.error(f"❌ Gagal melakukan prediksi: {e}")
                    st.stop()

                # --- Tentukan kelas ---
                if prediction.shape[-1] > 1:  # multi-class
                    class_index = np.argmax(prediction)
                    confidence = np.max(prediction)
                else:  # binary
                    class_index = int(prediction[0][0] > 0.5)
                    confidence = float(prediction[0][0]) if class_index == 1 else 1 - float(prediction[0][0])

                # --- Label sesuai urutan model ---
                class_labels = ["Kelas Arborio", "Kelas Basmati", "Kelas Ipsala", "Kelas Jasmine", "Kelas Karacadag"]

                # --- Pastikan index tidak melebihi jumlah kelas ---
                if class_index < len(class_labels):
                    predicted_label = class_labels[class_index]
                else:
                    predicted_label = "Tidak Dikenal"

                # --- Tampilkan hasil ---
                st.success("✅ Hasil Klasifikasi Gambar")
                st.write(f"### 🔖 Kelas Prediksi: **{predicted_label}**")
                st.write(f"🎯 Probabilitas: **{confidence:.2%}**")

    else:
        st.info("⬆ Silakan unggah gambar terlebih dahulu untuk melakukan prediksi.")
