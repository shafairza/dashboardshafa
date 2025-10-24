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
    
    - 🔍 YOLOv8: Model deteksi objek yang dapat mengenali objek tertentu di dalam gambar.  
    - 🧠 CNN / DenseNet201: Model klasifikasi gambar yang mengidentifikasi kategori dari gambar yang diunggah.
    
    ### 🎯 Tujuan Aplikasi
    - Menyediakan alat bantu interaktif untuk mengenali dan mengklasifikasikan objek secara otomatis.  
    - Meningkatkan efisiensi dalam pengolahan citra berbasis AI.  
    
    ### 📘 Cara Menggunakan
    1. Masuk ke halaman Prediksi Model di sidebar.
    2. Unggah gambar berformat .jpg, .jpeg, atau .png.
    3. Pilih mode Deteksi Objek (YOLO) atau Klasifikasi Gambar di sidebar.
    4. Lihat hasil deteksi atau klasifikasi yang ditampilkan secara visual dan probabilitasnya.
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
        img = Image.open(uploaded_file)
        st.image(img, caption="🖼 Gambar yang Diupload", use_container_width=True)

        # ==========================
        # DETEKSI OBJEK (DENGAN MODIFIKASI FOKUS)
        # ==========================
        if menu == "Deteksi Objek (YOLO)":
            st.subheader("🔍 Hasil Deteksi Objek (YOLO)")
            
            # --- MODIFIKASI DIMULAI DI SINI ---
            
            # Mendapatkan hasil prediksi (conf=0.5 adalah ambang batas minimal kepercayaan)
            results = yolo_model(img, conf=0.5) 
            
            # Mendapatkan daftar nama kelas dari model
            class_names = yolo_model.names
            
            # Kelas yang ingin difokuskan (HARUS sesuai dengan label di model Anda!)
            target_classes = ["smoking", "notsmoking"]
            
            # Cek apakah ada deteksi dari kelas target
            target_detections_found = False
            for r in results:
                # Mengambil indeks kelas yang terdeteksi
                detected_indices = r.boxes.cls.tolist()
                
                # Mengubah indeks kelas menjadi nama kelas yang terdeteksi
                detected_class_names = [class_names[int(i)] for i in detected_indices]
                
                # Memeriksa apakah ada kelas target yang terdeteksi
                if any(name in target_classes for name in detected_class_names):
                    target_detections_found = True
                    break

            if target_detections_found:
                # Jika objek target terdeteksi, tampilkan hasil plot dari YOLO
                result_img = results[0].plot()
                st.image(result_img, caption="📦 Hasil Deteksi", use_container_width=True)
                st.success(f"✅ Objek 'smoking' atau 'notsmoking' terdeteksi.")
            else:
                # Jika objek target tidak terdeteksi, tampilkan gambar asli dan peringatan
                st.warning(f"⚠️ **Tidak Terdeteksi:** Objek 'smoking' atau 'notsmoking' tidak ditemukan dalam gambar ini.")
                st.image(img, caption="Gambar Asli (Tidak Ada Deteksi Target)", use_container_width=True)

        # ==========================
        # KLASIFIKASI GAMBAR
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

            # Label kelas sesuai model
            class_labels = ["Kelas Aman", "Kelas Api", "Kelas Asap", "Kelas Asap dan Api"]
            predicted_label = class_labels[class_index]

            st.write(f"### 🔖 Kelas Prediksi: {predicted_label}")
            st.write(f"🎯 Probabilitas: {confidence:.2%}")

    else:
        st.info("⬆ Silakan unggah gambar terlebih dahulu untuk melakukan prediksi.")
