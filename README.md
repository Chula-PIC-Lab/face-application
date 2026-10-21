# 2110433 - Computer Vision (2026/1)
## Assignment 3: Face Application & Facial Emotion Recognition

An advanced Emotion Tracking Monitor & Evaluation Dashboard built with **Streamlit**, **OpenCV**, **MediaPipe**, **PyTorch**, and **PyTorch Lightning** (with optional **ONNX Runtime** support).

### 🎯 Purpose
This application is designed to continuously monitor and evaluate human emotions during various tasks or work sessions. It captures video feeds, detects faces, predicts emotions in real-time, visualizes probability streams over customizable time windows (1, 5, or 10 minutes), and logs session metrics for analysis.

Students can contribute to the project by:
1. Understanding the EfficientFace model architecture and attention mechanisms (`models/`).
2. Training, testing, and fine-tuning models using **PyTorch Lightning** (`train.py`).
3. Collecting self-captured facial images to fine-tune the emotion classifier.
4. Experimenting with PyTorch checkpoints and ONNX deployment in the interactive Streamlit app (`app.py`).

### Reference
[EfficientFace: An Efficient Deep Neural Network for Facial Emotion Recognition](https://github.com/kaiegg/EfficientFace)

---

## 📋 Prerequisites & System Requirements

⚠️ To ensure full compatibility with **MediaPipe (`0.10.21`)** and other core libraries, you must use a Python version **strictly lower than 3.14**.

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
   ```

3. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Running the Streamlit Application

```bash
streamlit run app.py
```
Open the local URL (usually `http://localhost:8501`) in your web browser.

### Switching Between PyTorch Checkpoint and ONNX in `app.py`
- **Default (PyTorch Checkpoint):** The app automatically discovers and loads the latest checkpoint (`.pth.tar` or `.ckpt`) from the `./checkpoint/` directory.
- **ONNX Runtime (Optional):** If you wish to use ONNX runtime:
  1. Open `app.py`.
  2. Set `USE_ONNX = True`.
  3. Uncomment the `import onnxruntime as ort` and ONNX loader blocks (look for the `TODO: ONNX` sections).

---

## 🏋️ Training, Testing & Fine-Tuning (`train.py`)

`train.py` is implemented using **PyTorch Lightning**, supporting full training, validation, testing, inference, and fine-tuning on custom student-captured images.

### 1. Fine-Tune on Student Captured Images
To fine-tune a pretrained checkpoint on images of yourself captured in `captured_images/`:
```bash
python train.py --mode fine_tune --custom_data_dir captured_images --checkpoint_path checkpoint/model_best.pth.tar --epochs 10 --lr 1e-4 --freeze_backbone
```

### 2. Full Training on FER2013 Dataset
To train on the standard FER2013 dataset:
```bash
python train.py --mode train --data_dir dataset/FER2013 --epochs 30 --batch_size 32 --lr 1e-3
```

### 3. Evaluate / Test Checkpoint
To test model accuracy on the test set:
```bash
python train.py --mode test --checkpoint_path checkpoint/model_best.pth.tar
```

### 4. Single-Image Inference
To predict the emotion of a single face image (just example, please rewrite to your real image path):
```bash
python train.py --mode infer --image_path captured_images/capture_1792406512183_1_Hapiness.jpg --checkpoint_path checkpoint/model_best.pth.tar
```

### 5. Export Checkpoint to ONNX
To export your fine-tuned model for deployment:
```bash
python train.py --mode export_onnx --checkpoint_path checkpoint/model_best.pth.tar --onnx_output models/efficient_face_student.onnx
```

---

## 💡 Student Contribution Guide (`TODO` Sections)

All contribution checkpoints are marked with:
```python
##################
# TODO: ...
##################
```

Key areas to contribute (if-any):
* **`models/EfficientFace.py`**:
  - Customize `LocalFeatureExtractor` (patch partitioning, facial landmark attention).
  - Add regularization (Dropout) to the classification head.
* **`models/modulator.py`**:
  - Implement alternative attention mechanisms (CBAM, ECA-Net, Coordinate Attention).
  - Experiment with dilation rates and channel reduction ratios.
* **`models/resnet.py`**:
  - Benchmark ResNet vs EfficientFace.
  - Implement Teacher-Student Knowledge Distillation.
* **`train.py`**:
  - Enhance Data Augmentations in `EmotionDataModule`.
  - Implement custom loss functions (Focal Loss, Class-weighted Loss, Label Smoothing).
  - Experiment with Learning Rate Schedulers (CosineAnnealing, OneCycleLR) and layer-wise learning rates.
* **`app.py`**:
  - Enable automatic face snapshot logging to `./captured_images/` during sessions.
  - Toggle between PyTorch and ONNX backends to compare latency and throughput.
