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
    Aplikasi ini dikembangkan oleh *Shafa* untuk mendeteksi dan mengklasifikasikan gambar menggunakan dua model utama:
    
    - 🔍 *YOLOv8*: Model deteksi objek yang dapat mengenali objek tertentu di dalam gambar.  
    - 🧠 *CNN / DenseNet201*: Model klasifikasi gambar yang mengidentifikasi kategori dari gambar yang diunggah.
    
    ### 🎯 Tujuan Aplikasi
    - Menyediakan alat bantu interaktif untuk mengenali dan mengklasifikasikan objek secara otomatis.  
    - Meningkatkan efisiensi dalam pengolahan citra berbasis AI.  
    
    ### 📘 Cara Menggunakan
    1. Masuk ke halaman *Prediksi Model* di sidebar.
    2. Unggah gambar berformat .jpg, .jpeg, atau .png.
    3. Pilih mode *Deteksi Objek (YOLO)* atau *Klasifikasi Gambar* di sidebar.
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
        # DETEKSI OBJEK (HANYA PEROKOK & NON PEROKOK)
        # ==========================
        if menu == "Deteksi Objek (YOLO)":
            st.subheader("🔍 Hasil Deteksi Objek (YOLO)")

            # Jalankan deteksi
            results = yolo_model(img)
            result = results[0]

            # Ambil nama kelas dari model YOLO
            names = result.names  # Dictionary {id: label}
            boxes = result.boxes

            # Fokus hanya pada dua kelas berikut
            target_classes = ["perokok", "non_perokok"]

            filtered_boxes = []
            for box in boxes:
                cls_id = int(box.cls[0])
                label = names[cls_id]
                if label in target_classes:
                    filtered_boxes.append(box)

            # Jika ditemukan deteksi perokok/non perokok
            if filtered_boxes:
                result_img = result.plot()
                st.image(result_img, caption="📦 Hasil Deteksi: Orang Perokok & Non-Perokok", use_container_width=True)

                # Menampilkan hasil deteksi dalam teks
                for box in filtered_boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = names[cls_id]
                    st.write(f"- **Kelas:** {label} | **Kepercayaan:** {conf:.2%}")

            else:
                st.warning("🚭 Tidak ditemukan objek orang perokok atau non-perokok pada gambar ini.")

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

            st.write(f"### 🔖 Kelas Prediksi: *{predicted_label}*")
            st.write(f"🎯 Probabilitas: *{confidence:.2%}*")

    else:
        st.info("⬆ Silakan unggah gambar terlebih dahulu untuk melakukan prediksi.")
