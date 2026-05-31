# 🛡️ Media Forensics & Deepfake Detection Suite

Advanced digital image forensics and deep learning classification application for detecting synthetic media and deepfakes using the MesoNet neural network architecture.

## Features

- **Executive Diagnostics**: Real-time forensic verdict with confidence scoring
- **Microscopic Vision Suite**: Advanced forensic preprocessing pipelines including:
  - Error Level Analysis (ELA)
  - Fourier Transform & Frequency Fingerprinting
  - Noise Floor & Structural Variance Analysis
  - Chrominance Channel Forensics
- **Statistical Fingerprints**: Benford's Law compliance and PRNU detection
- **MesoNet & XAI Hub**: Deep learning model visualization with Grad-CAM attention maps

## Technology Stack

- **Deep Learning**: PyTorch (MesoNet-4 CNN architecture)
- **Image Processing**: OpenCV, Pillow, SciPy
- **Data Science**: NumPy, Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Frontend**: Streamlit with custom glassmorphism styling

## Installation

```bash
pip install -r requirements.txt
```

## Running Locally

```bash
streamlit run app.py
```

The application will open at `http://localhost:8501`

## Deployment on Streamlit Cloud

1. Push your code to GitHub
2. Sign up at [share.streamlit.io](https://share.streamlit.io)
3. Click "New app" and select your repository
4. Set main file path to `app.py`
5. Deploy!

## Project Structure

```
├── app.py                                   # Main Streamlit application
├── requirements.txt                         # Python dependencies
├── deepfake_forensics_research_Project.ipynb  # Research notebook
├── implementation_plan.md                   # Technical documentation
└── README.md                               # This file
```

## Model Architecture

### MesoNet-4 CNN

- **Input**: 256×256×3 RGB images
- **Architecture**: 4 Convolutional blocks with BatchNorm, ReLU, and MaxPooling
- **Output**: Binary classification (Authentic/Deepfake) with confidence score
- **Performance**: ~97.6% accuracy on Celeb-DF dataset

## Forensic Analysis Methods

### Error Level Analysis (ELA)
Detects spliced regions through JPEG re-compression artifacts.

### FFT Spectral Analysis
Reveals periodic grid patterns from generative upsampling layers.

### Benford's Law Verification
Statistical deviation analysis on DCT coefficients.

### PRNU Detection
Simulates camera sensor fingerprint extraction.

### Grad-CAM Attribution
Visual explanation of model decision boundaries.

## Requirements

- Python 3.8+
- 2GB RAM (minimum)
- GPU recommended for faster processing

## License

This project is for research and educational purposes.

## Author

Deepfake Forensics Research Project
