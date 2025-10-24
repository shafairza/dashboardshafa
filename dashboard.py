import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
# Hapus import preprocess_input jika hanya menggunakan rescale=1./255
# from tensorflow.keras.applications.densenet import preprocess_input 

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
    Aplikasi ini dikembangkan oleh *Shafa* untuk mendeteksi dan mengklasifikasikan gambar menggunakan dua model utama:
    
    - 🔍 *YOLOv8*: Model deteksi objek (fokus: smoking/notsmoking).
    - 🧠 *CNN (DenseNet201)*: Model klasifikasi gambar (fokus: jenis beras).
    
    **Parameter Pelatihan Klasifikasi:**
    Gambar dinormalisasi dengan **`rescale=1./255`**.
    
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
        # DETEKSI OBJEK (YOLO) - Disesuaikan
        # ==========================
        if menu == "Deteksi Objek (YOLO)":
            st.subheader("🔍 Hasil Deteksi Objek (YOLO)")
            
            TARGET_DETECTION_CLASSES = ["smoking", "notsmoking"] 
            
            try:
                results = yolo_model(img, conf=0.25) 
                class_names = yolo_model.names
                target_detections_found = False
                
                for r in results:
                    detected_indices = r.boxes.cls.tolist()
                    detected_class_names = [class_names[int(i)] for i in detected_indices]
                    
                    if any(name in TARGET_DETECTION_CLASSES for name in detected_class_names):
                        target_detections_found = True
                        break

                if target_detections_found:
                    result_img = results[0].plot()
                    st.image(result_img, caption="📦 Hasil Deteksi", use_container_width=True)
                    st.success("✅ Objek 'smoking' atau 'notsmoking' terdeteksi!")
                else:
                    st.warning("Ini bukan gambar objek deteksi 'smoking' atau 'notsmoking'.")
                    st.image(img, caption="Gambar Asli (Tidak Ada Deteksi Target)", use_container_width=True)

            except Exception as e:
                st.error(f"Terjadi kesalahan saat deteksi: {str(e)}")

        # ==========================
        # KLASIFIKASI GAMBAR (DenseNet201) - MODIFIKASI RESCALE
        # ==========================
        elif menu == "Klasifikasi Gambar":
            st.subheader("🧩 Hasil Klasifikasi Gambar")

            CLASSIFICATION_LABELS = ["Basmati", "Ipsala", "Arborio", "Karacadag", "Jasmine"] 
            
            try:
                # 1. Tentukan target size
                input_shape = classifier.input_shape
                # DenseNet201 default input (224, 224, 3)
                target_size = (input_shape[1], input_shape[2]) if len(input_shape) == 4 and None not in input_shape else (224, 224)

                # 2. Preprocessing (Resize dan Konversi ke Array)
                img_resized = img.resize(target_size)
                img_array = image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0)
                
                # 3. NORMALISASI: Sesuai parameter training generator (rescale=1./255)
                # Langkah ini menggantikan preprocess_input jika training HANYA menggunakan rescale
                img_array = img_array / 255.0 
                
                # Prediksi
                prediction = classifier.predict(img_array, verbose=0)
                class_index = np.argmax(prediction)
                confidence = np.max(prediction)
                confidence_threshold = 0.7 

                if confidence >= confidence_threshold and 0 <= class_index < len(CLASSIFICATION_LABELS):
                    predicted_label = CLASSIFICATION_LABELS[class_index]
                    st.success(f"### 🔖 Kelas Prediksi: {predicted_label}")
                    st.write(f"🎯 Probabilitas: {confidence:.2%}")
                else:
                    st.warning("Ini bukan klasifikasi gambar butir beras atau model tidak yakin dengan prediksi.")
                    st.write(f"Probabilitas Tertinggi ({CLASSIFICATION_LABELS[class_index]}): {confidence:.2%}")

            except Exception as e:
                # Menangani error shape mismatch dengan pesan yang lebih membantu
                if "expected axis -1 of input shape to have value" in str(e):
                    st.error("""
                        🛑 **ERROR MODEL ARSITEKTUR (Klasifikasi)!**
                        Model `Shafa_Laporan 2.h5` memiliki masalah *shape mismatch* pada lapisan Dense pertamanya.
                        
                        Penyebab: Model yang disimpan **tidak memiliki lapisan Flatten atau Global Pooling** setelah lapisan DenseNet201.
                        
                        Solusi: **Latih ulang/simpan ulang model Anda** di script Keras *training* Anda dan pastikan `tf.keras.layers.Flatten()` atau `tf.keras.layers.GlobalAveragePooling2D()` disertakan sebelum lapisan Dense terakhir, lalu simpan kembali sebagai `Shafa_Laporan 2.h5`.
                    """)
                else:
                    st.error(f"Terjadi kesalahan saat klasifikasi: {str(e)}")

    else:
        st.info("⬆ Silakan unggah gambar terlebih dahulu untuk melakukan prediksi.")
