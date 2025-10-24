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
    # Model deteksi objek (smoking/notsmoking)
    yolo_model = YOLO("model/Shafa_Laporan 4.pt")  
    # Model klasifikasi (jenis beras)
    classifier = tf.keras.models.load_model("model/Shafa_Laporan 2.h5") 
    return yolo_model, classifier

# Menangani kegagalan memuat model saat inisialisasi
try:
    yolo_model, classifier = load_models()
    MODEL_LOAD_SUCCESS = True
except Exception as e:
    st.error(f"Gagal memuat model: {e}. Prediksi akan disimulasikan.")
    MODEL_LOAD_SUCCESS = False

    # Placeholder jika gagal load
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
    
    - 🔍 **YOLOv8 (Deteksi):** Model deteksi objek yang difokuskan pada objek **'smoking' atau 'notsmoking'**.
    - 🧠 **DenseNet201 (Klasifikasi):** Model klasifikasi gambar yang difokuskan pada **5 jenis butir beras**.

    **Keterangan:**  
    Model klasifikasi dilatih dengan ukuran gambar **250×250 piksel** dan **normalisasi `rescale=1./255`**.

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
                results = yolo_model(img, conf=0.25) 
                class_names = yolo_model.names
                target_detections_found = False
                
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
                    st.warning("⚠️ Tidak ada objek 'smoking' atau 'notsmoking' terdeteksi.")
                    st.image(img, caption="Gambar Asli (Tidak Ada Deteksi Target)", use_container_width=True)

            except Exception as e:
                st.error(f"Terjadi kesalahan saat deteksi: {str(e)}")

        # ==========================
        # KLASIFIKASI GAMBAR
        # ==========================
        elif menu == "Klasifikasi Gambar":
            st.subheader("🧩 Hasil Klasifikasi Gambar")

            CLASSIFICATION_LABELS = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]
            
            if not MODEL_LOAD_SUCCESS or classifier is None:
                st.error("Model Klasifikasi (`Shafa_Laporan 2.h5`) tidak dapat dimuat atau gagal diinisialisasi.")
            else:
                try:
                    # Gunakan ukuran input tetap sesuai pelatihan
                    target_size = (128, 128)

                    # Preprocessing
                    img_resized = img.resize(target_size)
                    img_array = image.img_to_array(img_resized)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = img_array / 255.0  

                    # Prediksi
                    prediction = classifier.predict(img_array, verbose=0)
                    class_index = np.argmax(prediction)
                    confidence = np.max(prediction)
                    confidence_threshold = 0.7  

                    predicted_label = CLASSIFICATION_LABELS[class_index]

                    # Tampilkan hasil
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.image(img_resized, caption="🔍 Gambar yang Diklasifikasikan", use_container_width=True)
                    with col2:
                        if confidence >= confidence_threshold:
                            st.success(f"### 🔖 Kelas Prediksi: {predicted_label}")
                            st.metric(label="🎯 Probabilitas", value=f"{confidence:.2%}")
                        else:
                            st.warning("⚠️ Model tidak yakin dengan prediksi ini.")
                            st.write(f"Prediksi tertinggi: **{predicted_label}** ({confidence:.2%})")

                except Exception as e:
                    error_message = str(e)
                    if "Matrix size-incompatible" in error_message or "incompatible with the layer" in error_message:
                        st.error("""
                            🛑 **ERROR KRITIS MODEL KLASIFIKASI!**
                            Model `Shafa_Laporan 2.h5` gagal prediksi karena **ketidaksesuaian dimensi fitur (shape mismatch)**.
                            
                            **Solusi:** Tambahkan `GlobalAveragePooling2D()` atau `Flatten()` di akhir base model sebelum Dense terakhir, lalu simpan ulang model.
                        """)
                    else:
                        st.error(f"Terjadi kesalahan saat klasifikasi: {error_message}")

    else:
        st.info("⬆ Silakan unggah gambar terlebih dahulu untuk melakukan prediksi.")
