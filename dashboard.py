import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2
import pandas as pd # Tambahkan untuk tabel hasil

# ==========================
# Konfigurasi Halaman (Mode Gelap disarankan)
# ==========================
st.set_page_config(
    page_title="Dashboard Rice and Smoker: Image Classification and Object Detection",
    layout="wide", # Gunakan layout lebar
    initial_sidebar_state="expanded"
)

# ==========================
# Load Models (Tetap menggunakan @st.cache_resource)
# ==========================
@st.cache_resource
def load_models():
    # Asumsi: Anda memiliki daftar nama kelas untuk klasifikasi
    # Ganti dengan nama kelas Anda yang sebenarnya!
    CLASS_NAMES = ['Kelas A', 'Kelas B', 'Kelas C', 'Kelas D', 'Kelas E']
    
    try:
        yolo_model = YOLO("model/Shafa_Laporan 4.pt")  # Model deteksi objek
        classifier = tf.keras.models.load_model("model/Shafa_Laporan 2.h5")  # Model klasifikasi
        return yolo_model, classifier, CLASS_NAMES
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None, None, None

yolo_model, classifier, CLASS_NAMES = load_models()

# Keluar jika model gagal dimuat
if yolo_model is None or classifier is None:
    st.stop()


# ==========================
# Fungsi untuk Tampilan Deteksi Objek
# ==========================
def display_object_detection(img, yolo_model):
    st.subheader("🚀 Hasil Deteksi Objek (YOLO)")
    
    # Bagian Kontrol (Sidebar)
    with st.sidebar:
        st.header("Pengaturan Deteksi")
        # Slider untuk Confidence Threshold
        conf_threshold = st.slider(
            "Confidence Threshold", 0.0, 1.0, 0.5, 0.05, 
            help="Hanya tampilkan objek dengan skor keyakinan di atas nilai ini."
        )

    # Deteksi objek dengan threshold
    results = yolo_model(img, conf=conf_threshold)
    result_img = results[0].plot()  # hasil deteksi (gambar dengan box)
    
    # 1. Tampilkan Gambar Hasil Deteksi
    st.image(result_img, caption="Hasil Deteksi dengan Bounding Box", use_container_width=True)

    # 2. Tampilkan Detail dalam Bentuk Tabel
    st.subheader("📊 Detail Objek Terdeteksi")
    boxes = results[0].boxes
    
    if len(boxes) > 0:
        data = []
        for box in boxes:
            conf = box.conf.item()
            cls_index = int(box.cls.item())
            
            # Asumsi: Nama kelas YOLO terperoleh dari names
            class_name = yolo_model.names.get(cls_index, f"ID: {cls_index}")
            
            # Koordinat Bounding Box (XYXY)
            x1, y1, x2, y2 = [int(i) for i in box.xyxy[0].tolist()]
            
            data.append({
                "Objek Terdeteksi": class_name,
                "Skor Keyakinan": f"{conf:.2f}",
                "Koordinat (x1, y1, x2, y2)": f"({x1}, {y1}, {x2}, {y2})"
            })
            
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Tidak ada objek terdeteksi dengan threshold ini.")

# ==========================
# Fungsi untuk Tampilan Klasifikasi Gambar
# ==========================
def display_image_classification(img, classifier, class_names):
    st.subheader("🔬 Hasil Klasifikasi Gambar")
    
    # Bagian Kontrol (Sidebar)
    with st.sidebar:
        st.header("Pengaturan Klasifikasi")
        # Anda bisa menambahkan pengaturan pre-processing di sini jika perlu
        st.info("Model ini mengharapkan gambar berukuran (224, 224).")

    # === LOGIKA PRE-PROCESSING & PREDIKSI (Sesuai kode asli Anda) ===
    # Preprocessing
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Prediksi
    with st.spinner('Menganalisis gambar...'):
        prediction = classifier.predict(img_array)
    
    # Menemukan hasil terbaik
    class_index = np.argmax(prediction)
    max_prob = np.max(prediction)
    
    # === TAMPILAN HASIL UTAMA ===
    predicted_class_name = class_names[class_index]
    
    # Gunakan metrik untuk menonjolkan hasil
    st.metric(
        label="Kelas Prediksi Paling Mungkin", 
        value=predicted_class_name, 
        delta=f"Probabilitas: {max_prob * 100:.2f}%"
    )
    
    st.markdown("---")
    
    # === TAMPILAN DETAIL PROBABILITAS ===
    st.subheader("📈 Distribusi Probabilitas Kelas")
    
    # Konversi hasil prediksi ke DataFrame
    prob_data = {
        "Kelas": class_names,
        "Probabilitas": prediction[0]
    }
    prob_df = pd.DataFrame(prob_data).sort_values(by="Probabilitas", ascending=False)
    
    # Tampilkan Bar Chart
    st.bar_chart(prob_df, x="Kelas", y="Probabilitas")
    
    # Tampilkan Tabel
    with st.expander("Lihat Detail Probabilitas dalam Tabel"):
        prob_df['Probabilitas'] = (prob_df['Probabilitas'] * 100).map('{:.2f}%'.format)
        st.dataframe(prob_df, use_container_width=True, hide_index=True)


# ==========================
# MAIN UI
# ==========================
st.title("🤖 Dashboard Prediksi Gambar Shafa Imut")
st.markdown("---")

# Gunakan kolom untuk menata Sidebar lebih rapi (Navigasi tetap di sidebar bawaan)
# st.sidebar.title("Kontrol & Mode") # Judul untuk sidebar

# Navigasi utama
menu = st.sidebar.selectbox("Pilih Mode Analisis:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])

uploaded_file = st.file_uploader("🖼️ Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    
    # Gunakan kolom untuk menampilkan gambar yang diupload dan hasil secara berdampingan (hanya ide)
    # Untuk kasus deteksi, kita biarkan gambar hasil memakan lebar penuh
    
    st.markdown("#### Gambar yang Diupload")
    st.image(img, caption="Gambar Sumber", use_container_width=True)
    st.markdown("---")

    if menu == "Deteksi Objek (YOLO)":
        display_object_detection(img, yolo_model)

    elif menu == "Klasifikasi Gambar":
        display_image_classification(img, classifier, CLASS_NAMES)

else:
    st.info("Silakan unggah file gambar untuk memulai prediksi. Pilih mode analisis di sidebar kiri.")

# Tambahkan footer sederhana
st.sidebar.markdown("---")
st.sidebar.caption("Dibuat dengan Streamlit, YOLOv8, & TensorFlow Keras")
