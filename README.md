# Dashboard Image Prediction by S.

A modern, interactive dashboard for machine learning / Deep Learning image classification and object detection built with Streamlit.

## Features

- 🖼️ **Image Classification** - Classify images into categories (Rice types)
- 🎯 **Object Detection** - Detect and locate objects in images specific to Smoking or NotSmoking
- 🤖 **Multiple Model Support** - TensorFlow (.h5) and PyTorch (.pt) models
- 🎨 **Modern UI** - Glassmorphism design with dark/light theme
- 📱 **Responsive Design** - Works on desktop but still not going well on mobile
- 📈 **Analytics Dashboard** - Interactive visualization chart and prediction history
- 💾 **Data Export** - Download results in CSV format

## Installation

### Quick Start (Demo Mode)

For a quick demo with frameworks:

```bash
pip install streamlit pandas numpy pillow plotly
streamlit run app.py
```

### Full Installation (With ML/DL Support)

For full functionality with actual model predictions:

```bash
# Install core dependencies
pip install streamlit pandas numpy pillow plotly

# Install ML frameworks
pip install tensorflow torch torchvision

# Run the application
streamlit run app.py
```

### Using requirements.txt

```bash
# Install all dependencies
pip install -r requirements.txt

```

## Usage

1. **Start the application:**
   ```bash
   streamlit run app.py
   ```

2. **Choose Task Type:**
   - Image Classification: Classify images into categories
   - Object Detection: Detect objects in images

3. **Select Model:**
   - TensorFlow Model: Uses `models/Shafa_Laporan 2.h5` Classify images into categories
   - PyTorch Model: Uses `models/Shafa_Laporan 4.pt` Detect objects in images

4. **Upload Image:**
   - Supported formats: PNG, JPG, JPEG
   - Maximum file size: 200MB

5. **Run Prediction:**
   - Click "Browse File" button for uploaded image
   - Wait for a while
   - View results and confidence scores
   - Analyze visualizations and history on analytics page

## Model Integration

### TensorFlow Models
Place your `.h5` model files in the `models/` directory:
```
models/
├── Shafa_Laporan 2.h5
```

### PyTorch Models
Place your `.pt` model files in the `models/` directory:
```
models/
├── Shafa_Laporan 4.pt
```

## Configuration

### Model Categories
The application is configured for rice classification with these categories based on rice varieties commonly cultivated in Turkey:
- Arborio
- Basmati
- Ipsala
- Karacadag
- Jasmine

Object Detection:
- Not Smoking
- Smoking

### Confidence Threshold
Adjust the confidence threshold in the sidebar (0-100%) to filter predictions based on model confidence.

## File Structure

```
├── Assets/                  # Logo file
│   └── Logo Dashboard.png
├── models/                  # DL model files
│   ├── Shafa_Laporan 2.h5
│   └── Shafa_Laporan 4.pt
├── sample image/            # Image used
│   └── Arborio.png
│   .....
│   └── Smoking.png
├── README.md               # This file
├── dashboard.py            # Main application file
├── requirements.txt        # Python dependencies

```

## License

This project is open source by Shafa Statistics Student Universitas Syiah Kuala and available under laboratory Big Data Programming.

---

**Note:** This dashboard works in production mode (with actual ML and DL models). The production mode provides real model predictions by developer is Shafa.
