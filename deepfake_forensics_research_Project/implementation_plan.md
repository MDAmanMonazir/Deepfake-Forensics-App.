# Implementation Plan - Deepfake Forensics Streamlit App

This plan outlines the design and development of an interactive, premium-grade **Streamlit Web Application** based on the digital media forensics notebook: `deepfake_forensics_research_Project.ipynb`.

The app will serve as a complete **Advanced Digital Image Forensics and Deep Learning Classification Suite**, allowing users to upload images, extract microscopic digital fingerprints, and perform multi-modal analyses.

---

## User Review Required

> [!IMPORTANT]
> The app will run the fully functional mathematical forensic tools (ELA, FFT, Noise Residuals, Chroma Discrepancy, Local SNR, Benford's Law, and PRNU mapping) on the user's uploaded images.
>
> For **MesoNet Deep Learning Prediction**: Since the original model in the notebook is randomly initialized (and the training is simulated), the inference will use the initialized PyTorch MesoNet model. We will add a note to explain this to the user, and offer a toggle to view mock high-confidence predictions to show how a fully trained model operates, or perform raw live inference on the MesoNet architecture.
>
> Let us know if you have a pre-trained weights file (e.g., `mesonet.pth`) that you want us to load instead. If not, the current approach is perfectly self-contained.

---

## Proposed Changes

We will create a multi-tab Streamlit dashboard (`app.py`) inside the project directory, utilizing premium dark-glassmorphic styling, custom CSS components, and highly interactive sliders/parameters for deep forensics exploration.

### Components

#### [NEW] [app.py](file:///c:/Users/mdama/Desktop/deepfake_forensics_research_Project/app.py)
This is the core Streamlit application. It will contain:
1. **Premium Aesthetic Engine**: Injected HTML/CSS to enable a sleek dark-glassmorphism UI with elegant card elements, electric blue/crimson details, and high-end typography.
2. **Forensic Logic Porting**:
   - `compute_ela(img, quality, scale)`: Modified to accept PIL images in-memory to prevent standard file operations conflicts.
   - `compute_fft_spectrum(img)`: Modified for in-memory processing.
   - `plot_frequency_fingerprint(img)`: Generates an interactive 3D surface plot using Matplotlib inside Streamlit.
   - `plot_cumulative_ela(img, steps)`: Interactive compression decay tracker.
   - `plot_noise_print(img)` & `plot_variance_map(img)`: Noise residual mapping.
   - `plot_chroma_discrepancy(img)` & `plot_snr_map(img)`: Advanced color-space forensics.
   - `plot_benfords_law(img)`: DCT distribution verification against organic curves.
   - `plot_prnu_fingerprint(img)`: Med-filter high-frequency PRNU camera sensor fingerprint.
   - `Meso4` PyTorch Model Class: Fully instantiated for real-time computational flow.
   - **Explainable AI (XAI)**: Grad-CAM overlay visualizer showing facial blending boundary seams.
3. **Multi-Tab Dashboard Navigation**:
   - **🛡️ Executive Diagnosis**: Main upload arena. Shows the uploaded picture, a premium forensic audit card, classification verdict, custom gauge indicators, and a quick-view of ELA/FFT analysis.
   - **🔬 Microscopic Vision Suite**: Interactive tab containing ELA (with dynamic sliders for quality and scale), Cumulative ELA (error decay), FFT (2D & interactive 3D surface plots), Noise Floors (residual & variance maps), and Chrominance channels (YCrCb & SNR maps).
   - **📊 Statistical Fingerprints**: Analyzes Benford's Law compliance and Sensor PRNU maps.
   - **🤖 Neural Analytics & Metrics**: Deep dive into the MesoNet model architecture, simulated training curves, ROC-AUC profiles, Calibration curves, score distributions, and Precision-Recall tradeoffs.

#### [NEW] [requirements.txt](file:///c:/Users/mdama/Desktop/deepfake_forensics_research_Project/requirements.txt)
Specifies dependencies for running the Streamlit app:
```text
streamlit>=1.30.0
numpy
matplotlib
seaborn
opencv-python-headless
scipy
torch
torchvision
scikit-learn
Pillow
```

---

## Verification Plan

### Automated & Manual Verification
1. **Dependencies Check**: Verify python environments and libraries.
2. **Streamlit App Launch**: Start local development server using `streamlit run app.py` and inspect pages using the browser.
3. **Upload Testing**: Test with various image sizes (PNG, JPEG) to ensure image boundaries do not break in-memory processing.
4. **Parameter Sweeps**: Verify the ELA quality slider and scale factor dynamically recompute and render the image correctly in real-time.
5. **Plot Verifications**: Ensure 3D surface FFT plots, Benford's Law bars, and MesoNet metrics render with proper styling matching the dark layout.
