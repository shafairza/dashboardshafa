import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
# Hapus import preprocess_input karena kita menggunakan rescale=1./255
# from tensorflow.keras.applications.densenet import preprocess_input 

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    # Model deteksi objek (asumsi: smoking/notsmoking)
    yolo_model = YOLO("model/Shafa_Laporan 4.pt")  
    # Model klasifikasi (asumsi: jenis beras)
    classifier = tf.keras.models.load_model("model/Shafa_Laporan 2.h5") 
    return yolo_model, classifier

# Menangani kegagalan memuat model saat inisialisasi
try:
    yolo_model, classifier = load_models()
    MODEL_LOAD_SUCCESS = True
except Exception as e:
    st.error(f"Gagal memuat model: {e}. Prediksi akan disimulasikan.")
    MODEL_LOAD_SUCCESS = False
    
    # Definisi placeholder jika model gagal dimuat
    class DummyYOLO:
        def __call__(self, img, conf=0.25):
            class DummyBoxes:
                cls = []
            class DummyResults:
                boxes = DummyBoxes()
                def plot(self):
                    # Mengkonversi PIL Image ke numpy array untuk simulasi plot
                    return np.array(img.convert('RGB')) 
            return [DummyResults()]
        names = ["smoking", "notsmoking"]
    
    yolo_model = DummyYOLO()
    classifier = None

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
    
    - 🔍 **YOLOv8 (Deteksi):** Model deteksi objek yang **difokuskan pada objek 'smoking' atau 'notsmoking'**.
    - 🧠 **CNN (Klasifikasi):** Model klasifikasi gambar yang **difokuskan pada 5 jenis butir beras**.
    
    **PENTING (Klasifikasi):** Model dilatih menggunakan pra-pemrosesan **`rescale=1./255`**.
    
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
        # DETEKSI OBJEK (YOLO)
        # ==========================
        if menu == "Deteksi Objek (YOLO)":
            st.subheader("🔍 Hasil Deteksi Objek (YOLO)")
            
            TARGET_DETECTION_CLASSES = ["smoking", "notsmoking"] 
            
            try:
                # Lakukan deteksi
                results = yolo_model(img, conf=0.25) 
                class_names = yolo_model.names
                target_detections_found = False
                
                # Cek hasil deteksi untuk kelas target
                for r in results:
                    if hasattr(r, 'boxes') and hasattr(r.boxes, 'cls'):
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
        # KLASIFIKASI GAMBAR (DenseNet201)
        # ==========================
        elif menu == "Klasifikasi Gambar":
            st.subheader("🧩 Hasil Klasifikasi Gambar")

            CLASSIFICATION_LABELS = ["Basmati", "Ipsala", "Arborio", "Karacadag", "Jasmine"] 
            
            # Cek status pemuatan model sebelum mencoba prediksi
            if not MODEL_LOAD_SUCCESS or classifier is None:
                 st.error("Model Klasifikasi (`Shafa_Laporan 2.h5`) tidak dapat dimuat atau gagal diinisialisasi. Tidak bisa memproses prediksi.")
                 # Tidak perlu 'return' karena ini bukan fungsi

            try:
                # 1. Tentukan target size (Asumsi 224x224, atau ambil dari model)
                if MODEL_LOAD_SUCCESS and classifier:
                    input_shape = classifier.input_shape
                    target_size = (input_shape[1], input_shape[2]) if len(input_shape) == 4 and None not in input_shape else (224, 224)
                else:
                    target_size = (224, 224)
                    
                # 2. Preprocessing (Resize dan Konversi ke Array)
                img_resized = img.resize(target_size)
                img_array = image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0)
                
                # 3. NORMALISASI: Sesuai parameter training generator (rescale=1./255)
                img_array = img_array / 255.0 
                
                # Prediksi
                prediction = classifier.predict(img_array, verbose=0)
                class_index = np.argmax(prediction)
                confidence = np.max(prediction)
                confidence_threshold = 0.7 

                # Cek hasil dan confidence
                if confidence >= confidence_threshold and 0 <= class_index < len(CLASSIFICATION_LABELS):
                    predicted_label = CLASSIFICATION_LABELS[class_index]
                    st.success(f"### 🔖 Kelas Prediksi: {predicted_label}")
                    st.write(f"🎯 Probabilitas: {confidence:.2%}")
                else:
                    st.warning("Ini bukan klasifikasi gambar butir beras atau model tidak yakin dengan prediksi.")
                    st.write(f"Probabilitas Tertinggi ({CLASSIFICATION_LABELS[class_index]}): {confidence:.2%}")

            except Exception as e:
                # Penanganan error Shape Mismatch yang Konsisten (Masalah model)
                error_message = str(e)
                if "Matrix size-incompatible" in error_message or "incompatible with the layer: expected axis -1" in error_message:
                    st.error("""
                        🛑 **ERROR KRITIS MODEL KLASIFIKASI!**
                        Model `Shafa_Laporan 2.h5` gagal prediksi karena **ketidaksesuaian dimensi fitur** (shape mismatch).
                        
                        **Tindakan Perbaikan:** Anda harus **memperbaiki arsitektur model DenseNet di script training asli** Anda. Pastikan Anda menyertakan lapisan **`tf.keras.layers.GlobalAveragePooling2D()`** atau **`tf.keras.layers.Flatten()`** antara *Base Model DenseNet201* dan lapisan *Dense* terakhir. Setelah diperbaiki, **simpan ulang model** ke `model/Shafa_Laporan 2.h5`.
                    """)
                else:
                    st.error(f"Terjadi kesalahan saat klasifikasi: {error_message}")

    else:
        st.info("⬆ Silakan unggah gambar terlebih dahulu untuk melakukan prediksi.")
