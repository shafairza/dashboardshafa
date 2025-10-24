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
    yolo_model = YOLO("model/Shafa_Laporan 4.pt")  # Model deteksi objek (smoking / not smoking)
    classifier = tf.keras.models.load_model("model/Shafa_Laporan 2.h5")  # Model klasifikasi (beras)
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
    Aplikasi ini dikembangkan oleh *Shafa* untuk dua fungsi utama:

    - 🔍 **Deteksi Objek (YOLOv8)** — mendeteksi aktivitas seperti *Smoking* atau *Not Smoking*.
    - 🧠 **Klasifikasi Gambar (DenseNet201)** — mengenali jenis butir beras seperti *Basmati*, *Ipsala*, *Arborio*, *Karacadag*, dan *Jasmine*.

    ### 📘 Cara Menggunakan
    1. Buka halaman *Prediksi Model*.
    2. Unggah gambar (.jpg/.jpeg/.png).
    3. Pilih mode deteksi atau klasifikasi sesuai kebutuhan.
    """)

# ==========================
# PAGE 2: Prediksi Model
# ==========================
elif page == "Prediksi Model":
    st.title("🧠 Prediksi Model Deteksi & Klasifikasi")

    menu = st.sidebar.selectbox("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])
    uploaded_file = st.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="🖼 Gambar yang Diupload", use_container_width=True)
        except Exception as e:
            st.error(f"Gagal membaca gambar: {str(e)}")
            st.stop()

        # ==========================
        # DETEKSI OBJEK (YOLO)
        # ==========================
        if menu == "Deteksi Objek (YOLO)":
            st.subheader("🔍 Hasil Deteksi Objek (YOLO)")
            try:
                results = yolo_model(img)
                result_img = results[0].plot()

                # Ambil nama kelas dari hasil YOLO
                names = yolo_model.names
                detected_classes = [names[int(box.cls)] for box in results[0].boxes]

                if any(cls.lower() in ["smoking", "not smoking"] for cls in detected_classes):
                    st.image(result_img, caption="📦 Hasil Deteksi", use_container_width=True)
                    st.success(f"🎯 Objek terdeteksi: {', '.join(detected_classes)}")
                else:
                    st.warning("⚠ Ini bukan gambar objek deteksi (Smoking/Not Smoking).")

            except Exception as e:
                st.error(f"Terjadi kesalahan saat deteksi: {str(e)}")

        # ==========================
        # KLASIFIKASI GAMBAR (DenseNet201)
        # ==========================
        elif menu == "Klasifikasi Gambar":
            st.subheader("🧩 Hasil Klasifikasi Gambar")

            try:
                # Ukuran input model
                input_shape = classifier.input_shape
                if len(input_shape) == 4 and all(v is not None for v in input_shape[1:3]):
                    target_size = (input_shape[1], input_shape[2])
                else:
                    target_size = (224, 224)  # default DenseNet201

                # Preprocessing
                img_resized = img.resize(target_size)
                img_array = image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)

                # Prediksi
                prediction = classifier.predict(img_array)
                class_index = np.argmax(prediction)
                confidence = np.max(prediction)

                # Label kelas
                class_labels = ["Basmati", "Ipsala", "Arborio", "Karacadag", "Jasmine"]
                predicted_label = class_labels[class_index]

                # Ambang batas
                confidence_threshold = 0.7

                if confidence < confidence_threshold:
                    st.warning("⚠ Ini bukan klasifikasi gambar butir beras.")
                else:
                    st.success(f"### 🔖 Kelas Prediksi: {predicted_label}")
                    st.write(f"🎯 Probabilitas: {confidence:.2%}")

            except Exception as e:
                st.error(f"Terjadi kesalahan saat klasifikasi: {str(e)}")

    else:
        st.info("⬆ Silakan unggah gambar terlebih dahulu untuk melakukan prediksi.")
