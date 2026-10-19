# 2110433 - Computer Vision (2026/1)
## Assignment 3: Face Application

An advanced Emotion Tracking Monitor & Evaluation Dashboard built with **Streamlit**, **OpenCV**, **MediaPipe**, and **ONNX Runtime (EfficientFace)**. 

### 🎯 Purpose
This application is designed to continuously monitor and evaluate human emotions and feelings during various tasks or work sessions. It captures video feeds, detects faces, predicts emotions in real-time, visualizes probability streams over customizable time windows (1, 5, or 10 minutes), and logs session metrics for analysis.

### Reference
[https://github.com/kaiegg/EfficientFace](https://github.com/kaiegg/EfficientFace)

---

## 📋 Prerequisites & System Requirements

⚠️To ensure full compatibility with **MediaPipe (`0.10.21`)** and other core libraries, you must use a Python version **strictly lower than 3.14**. 

* **Recommended Python Version:** Python **3.9, 3.10, 3.11, or 3.12**
* **Webcam:** A working webcam connected to your device (Camera Index `0`).

---

## ⚙️ Installation & Setup

1. **Clone or Download** this repository to your local machine and navigate to the project directory.

2. **Create and Activate a Virtual Environment** (Recommended):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
3. **Install requirements**
    ```bash
    pip install -r requirements.txt

## 🚀 Running the Application
1. **Run streamlit**
    ```bash
    streamlit run app.py

Open the local URL provided in your terminal (usually http://localhost:8501) in your web browser to start using the application.