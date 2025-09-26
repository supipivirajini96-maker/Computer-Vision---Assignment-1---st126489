# Computer-Vision---Assignment-1---st126489

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
  Capture multiple frames and stitch them into a panorama.  
  ⚠️ Note: The panorama stitching is implemented **without using built-in OpenCV panorama functions**.

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
