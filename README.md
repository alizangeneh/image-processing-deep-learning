Image Processing with Deep Learning
https://github.com/alizangeneh

This project is a cross-platform desktop application for advanced image processing using deep learning models. The application is built with PyQt5 and works on Windows, Linux, and macOS.

The program includes the following features:

Smart Image Compressor
Automatically reduces image resolution to an optimal size while maintaining good visual quality.

Background Removal (AI-based)
Uses the rembg deep learning model to remove the background of images.

Face Blurring Privacy Tool
Detects faces using a DNN model and applies Gaussian blur to protect privacy.

Quality Booster (Super-Resolution)
Enhances image sharpness and resolution by 4x using the RealESRGAN model.

Drag and Drop Support
Users can drag and drop images directly into the application window.

Project Folder Structure:

project/

main.py

requirements.txt

README.md

models/

deploy.prototxt

res10_300x300_ssd_iter_140000.caffemodel

RealESRGAN_x4plus.pth

Installation Instructions:

Install all dependencies with:
pip install -r requirements.txt

Place the required model files inside the "models" folder:

deploy.prototxt

res10_300x300_ssd_iter_140000.caffemodel

RealESRGAN_x4plus.pth

Running the Application:

python main.py

Optional: Build an Executable File

Windows:
pyinstaller --noconsole --onefile main.py

macOS:
pyinstaller --onefile main.py

Linux:
pyinstaller --onefile main.py

Deep Learning Models Used:

RealESRGAN x4plus: Used for super-resolution.

Rembg (U2Net): Used for background removal.

OpenCV DNN Face Detector (SSD): Used for face detection.

PyTorch CPU: Used for maximum compatibility with older GPUs such as GeForce 210.

Notes:

Torch is installed in CPU mode for full compatibility across systems.

The application is cross-platform.

Due to file size limits, model weights must be downloaded manually.

Author:
Ali Zangeneh
GitHub: https://github.com/alizangeneh