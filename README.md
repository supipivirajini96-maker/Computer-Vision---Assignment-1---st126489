# Computer Vision Assignment #1

This project implements a webcam-based interactive computer vision application using **OpenCV** and **NumPy**.  
It demonstrates multiple image processing, camera calibration, and augmented reality (AR) techniques.

---

## ✨ Features

Your app includes the following functions:

- **Color Conversion**  
  Convert image between:
  - RGB ↔ Grayscale  
  - RGB ↔ HSV (Hue, Saturation, Value channels)

- **Contrast and Brightness**  
  Adjust contrast and brightness interactively using trackbars.

- **Image Histogram**  
  Display the histogram of the live video feed.

- **Filters**  
  - Gaussian filter (with adjustable kernel size and sigma)  
  - Bilateral filter (with adjustable parameters)

- **Edge and Line Detection**  
  - Canny edge detection  
  - Line detection using Hough Transform

- **Panorama Creation**  
  Capture multiple frames and stitch them into a panorama. This implementation is **without using built-in OpenCV panorama functions**.

- **Geometric Transformations**  
  Apply translation, rotation, and scaling using trackbars.

- **Camera Calibration**  
  Calibrate the camera using a chessboard pattern and show distortion correction.

- **Augmented Reality (AR)**  
  - Detect an ArUco marker (`A4_ArUco_Marker.png`).  
  - Instead of projecting a simple cube, the app loads the provided **T-Rex 3D model (`trex_model.obj`)** and overlays it in AR mode.  
  - The model is scaled up for better visibility.

---

## 📦 Requirements

- **VSCode** (or any Python IDE)  
- **Python 3.9+**

Install required dependencies:

```bash
pip install opencv-python opencv-contrib-python matplotlib numpy

```
## 🎮 Navigation & Controls

Once the webcam feed is running, you can press keys to navigate between modes.
The respective keys are avaiable in webcam feed in suqare [] brackers

# Home / Normal Mode

i → Image Color Conversion (switch between RGB, Gray, HSV)

c → Contrast & Brightness (adjust using trackbars)

h → Show Histogram

f → Filter Mode (choose Gaussian or Bilateral filter)

e → Edge Detection (Canny / Hough Line)

p → Panorama Mode (capture frames and stitch into panorama)

m → Transform Mode (translation, rotation, scaling via trackbars)

d → Camera Calibration & Distortion Demo

a → Augmented Reality Mode (overlay T-Rex 3D model on ArUco marker)

q → Quit the app

# Inside Each Mode

ESC → Return to Home / Normal mode

Trackbars → Adjust available parameters interactively

Mode-specific keys

In Filter mode: g = Gaussian, b = Bilateral

In Edge mode: c = Canny, l = Hough Lines

In Color mode: r = RGB, g = Gray, h = HSV

In HSV mode: 0 = Hue, 1 = Saturation, 2 = Value

## ▶️ How to Run in VS Code

1. **Clone this repository** into your local machine using the following command:

   ```bash
   git clone https://github.com/supipivirajini96-maker/Computer-Vision-Assignment-1-st126489.git

   ```
2. Check app.py and trex_model.obj avaiable in the same repository
3. In terminal, run the command "python app.py"

