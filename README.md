# Dashboard Image Prediction by S.

A modern, interactive dashboard for machine learning image classification and object detection built with Streamlit.

## Features

- 🖼️ **Image Classification** - Classify images into categories (Rice types)
- 🎯 **Object Detection** - Detect and locate objects in images specific to Smoking or NotSmoking
- 🤖 **Multiple Model Support** - TensorFlow (.h5) and PyTorch (.pt) models
- 📊 **Interactive Visualizations** - Real-time charts and analytics
- 🎨 **Modern UI** - Glassmorphism design with dark/light theme
- 📱 **Responsive Design** - Works on desktop and mobile
- 📈 **Analytics Dashboard** - Performance metrics and prediction history
- 💾 **Data Export** - Download results in CSV format

## Installation

### Quick Start (Demo Mode)

For a quick demo without ML frameworks:

```bash
pip install streamlit pandas numpy pillow plotly
streamlit run app.py
```

### Full Installation (With ML Support)

For full functionality with actual model predictions:

```bash
# Install core dependencies
pip install streamlit pandas numpy pillow plotly

# Install ML frameworks (optional)
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
   - TensorFlow Model: Uses `models/Shafa_Laporan 2.h5`
   - PyTorch Model: Uses `models/Shafa_Laporan 4.pt`
   - Simulation Model: Demo mode without ML frameworks

4. **Upload Image:**
   - Supported formats: PNG, JPG, JPEG
   - Maximum file size: 200MB

5. **Run Prediction:**
   - Click "Start Prediction" button
   - View results and confidence scores
   - Analyze visualizations and history

## Model Integration

### TensorFlow Models
Place your `.h5` model files in the `models/` directory:
```
models/
├── Shafa_Laporan 2.h5
└── your_model.h5
```

### PyTorch Models
Place your `.pt` model files in the `models/` directory:
```
models/
├── Shafa_Laporan 4.pt
└── your_model.pt
```

## Configuration

### Model Categories
The application is configured for rice classification with these categories:
- Arborio
- Basmati
- Ipsala
- Jasmine

Object Detection:
- Not Smoking
- Smoking

To modify categories, edit the `categories` list in the `predict_classification()` function.

### Confidence Threshold
Adjust the confidence threshold in the sidebar (0-100%) to filter predictions based on model confidence.

### Framework Status

The dashboard automatically detects available ML frameworks and shows status in the sidebar:
- ✅ Available: Framework is installed and ready
- ❌ Not Available: Framework is not installed (demo mode)

## File Structure

```
├── dashboard.py                 # Main application file
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── models/               # ML model files
│   ├── Shafa_Laporan 2.h5
│   └── Shafa_Laporan 4.pt
├── sample image/         # Uploaded images (auto-created)

```

## Development

### Adding New Models

1. Place model files in `models/` directory
2. Update model loading functions in `app.py`
3. Modify prediction functions for your specific model
4. Update categories and labels as needed

### Customizing UI

The application uses inline CSS for styling. Modify the `load_css()` function to customize:
- Colors and themes
- Layout and spacing
- Animations and effects
- Responsive design

## License

This project is open source and available under the MIT License.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review console error messages
3. Ensure all dependencies are installed correctly
4. Verify model file compatibility

---

**Note:** This dashboard works in both demo mode (simulation) and production mode (with actual ML models). The demo mode is perfect for testing and development, while production mode provides real model predictions.
