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
    # Asumsi: Shafa_Laporan 4.pt adalah model YOLO yang mencakup kelas 'smoking'/'notsmoking'
    yolo_model = YOLO("model/Shafa_Laporan 4.pt")  # Model deteksi objek
    # Asumsi: Shafa_Laporan 2.h5 adalah model Klasifikasi Beras
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
    Aplikasi ini dikembangkan oleh *Shafa* untuk mendeteksi dan mengklasifikasikan gambar menggunakan dua model utama:
    
    - 🔍 *YOLOv8*: Model deteksi objek (fokus: smoking/notsmoking).
    - 🧠 *CNN (DenseNet201)*: Model klasifikasi gambar (fokus: jenis beras).
    
    ### 📘 Cara Menggunakan
    1. Buka halaman *Prediksi Model*.
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
        # DETEKSI OBJEK (YOLO) - MODIFIKASI
        # ==========================
        if menu == "Deteksi Objek (YOLO)":
            st.subheader("🔍 Hasil Deteksi Objek (YOLO)")
            
            # Kelas yang ingin difokuskan (HARUS sesuai dengan label di model YOLO Anda!)
            # Asumsi nama kelasnya adalah 'smoking' dan 'notsmoking'
            TARGET_DETECTION_CLASSES = ["smoking", "notsmoking"] 
            
            try:
                # Lakukan deteksi
                # Menambahkan argumen conf=0.25 (default YOLO) atau sesuai kebutuhan
                results = yolo_model(img, conf=0.25) 
                
                # Mendapatkan daftar nama kelas dari model YOLO
                class_names = yolo_model.names
                target_detections_found = False
                
                # Cek hasil deteksi
                for r in results:
                    detected_indices = r.boxes.cls.tolist()
                    detected_class_names = [class_names[int(i)] for i in detected_indices]
                    
                    # Periksa apakah ada kelas target yang terdeteksi
                    if any(name in TARGET_DETECTION_CLASSES for name in detected_class_names):
                        target_detections_found = True
                        break # Cukup satu deteksi yang relevan

                if target_detections_found:
                    # Tampilkan gambar hasil deteksi
                    result_img = results[0].plot()
                    st.image(result_img, caption="📦 Hasil Deteksi", use_container_width=True)
                    st.success("✅ Objek 'smoking' atau 'notsmoking' terdeteksi!")
                else:
                    # Tampilkan pesan jika objek target tidak terdeteksi
                    st.warning("Ini bukan gambar objek deteksi 'smoking' atau 'notsmoking'.")
                    st.image(img, caption="Gambar Asli (Tidak Ada Deteksi Target)", use_container_width=True)

            except Exception as e:
                st.error(f"Terjadi kesalahan saat deteksi: {str(e)}")

        # ==========================
        # KLASIFIKASI GAMBAR (DenseNet201) - MODIFIKASI
        # ==========================
        elif menu == "Klasifikasi Gambar":
            st.subheader("🧩 Hasil Klasifikasi Gambar")

            # Label kelas (⚠ pastikan urutannya sesuai training model 'Shafa_Laporan 2.h5')
            CLASSIFICATION_LABELS = ["Basmati", "Ipsala", "Arborio", "Karacadag", "Jasmine"] 
            
            try:
                # Ambil ukuran input model
                input_shape = classifier.input_shape
                target_size = (input_shape[1], input_shape[2]) if len(input_shape) == 4 and None not in input_shape else (224, 224)

                # Preprocessing sesuai DenseNet201
                img_resized = img.resize(target_size)
                img_array = image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array) # Menggunakan preprocess_input DenseNet

                # Prediksi
                prediction = classifier.predict(img_array, verbose=0)
                class_index = np.argmax(prediction)
                confidence = np.max(prediction)
                
                # Ambang batas kepercayaan umum
                confidence_threshold = 0.7 

                # Cek apakah confidence mencapai ambang batas dan apakah kelasnya termasuk kategori beras
                if confidence >= confidence_threshold and 0 <= class_index < len(CLASSIFICATION_LABELS):
                    predicted_label = CLASSIFICATION_LABELS[class_index]
                    st.success(f"### 🔖 Kelas Prediksi: {predicted_label}")
                    st.write(f"🎯 Probabilitas: {confidence:.2%}")
                else:
                    # Output jika kepercayaan rendah ATAU kelas tidak teridentifikasi dengan jelas sebagai beras
                    st.warning("Ini bukan klasifikasi gambar butir beras atau model tidak yakin dengan prediksi.")
                    st.write(f"Probabilitas Tertinggi ({CLASSIFICATION_LABELS[class_index]}): {confidence:.2%}")

            except Exception as e:
                st.error(f"Terjadi kesalahan saat klasifikasi: {str(e)}")

    else:
        st.info("⬆ Silakan unggah gambar terlebih dahulu untuk melakukan prediksi.")
