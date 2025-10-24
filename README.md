# 🧠 ML Image Prediction Dashboard

A modern, interactive dashboard for machine learning image classification and object detection built with Streamlit.

---

## 🚀 Features
- 🖼️ **Image Classification** — Classify images into categories (Fire, Smoke, Safe, Fire & Smoke)  
- 🎯 **Object Detection** — Detect and locate objects in images  
- 🤖 **Multiple Model Support** — TensorFlow (`.h5`) and PyTorch (`.pt`) models  
- 📊 **Interactive Visualizations** — Real-time charts and analytics  
- 🎨 **Modern UI** — Glassmorphism design with dark/light theme  
- 📱 **Responsive Design** — Works seamlessly on desktop and mobile  
- 📈 **Analytics Dashboard** — Performance metrics and prediction history  
- 💾 **Data Export** — Download prediction results in CSV format  

---

## ⚙️ Installation

### 🧩 Quick Start (Demo Mode)
For a quick demo without ML frameworks:

```bash
pip install streamlit pandas numpy pillow plotly
streamlit run app.py

# Install core dependencies
pip install streamlit pandas numpy pillow plotly

# Install ML frameworks (optional)
pip install tensorflow torch torchvision

# Run the application
streamlit run app.py

# Install all dependencies
pip install -r requirements.txt

# Or install only core dependencies
pip install streamlit pandas numpy pillow plotly scikit-learn opencv-python-headless matplotlib seaborn

streamlit run app.py

├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── models/                # ML model files
│   ├── Shafa_Laporan2.h5
│   └── Shafa_Laporan4.pt
├── uploads/               # Uploaded images (auto-created)
└── data/                  # Data files (if any)
