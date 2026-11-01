# 🍎 Apple Disease Detection Using YOLO11

An AI-powered Apple Fruit Disease Detection system built using **YOLO11 (You Only Look Once)** and **Streamlit**. The application detects and classifies apple fruit conditions from uploaded images through an interactive web interface.

## 📌 Project Overview

Apple diseases can affect fruit quality, agricultural productivity, and market value. This project uses deep learning and computer vision to identify apple fruit conditions automatically.

The system uses a custom-trained YOLO11 model to detect four categories:

- **BLOTCH** – Apple blotch
- **HEALTHY** – Healthy apple fruit
- **ROT** – Apple rot
- **SCAB** – Apple scab

Users can upload an apple image and view the detected classes, confidence scores, bounding boxes, and annotated image through a Streamlit dashboard.

## ✨ Features

- YOLO11-based object detection.
- Detection of four apple fruit categories.
- Interactive Streamlit web interface.
- Image upload and real-time inference.
- Bounding boxes around detected regions.
- Confidence scores for predictions.
- Detection statistics and class-wise counts.
- Original and annotated image comparison.
- Downloadable prediction results.
- GPU acceleration when CUDA is available.

## 🧠 Model Information

| Parameter | Details |
|---|---|
| Model | YOLO11n |
| Framework | Ultralytics YOLO |
| Task | Object Detection |
| Image Size | 640 × 640 |
| Training Epochs | 60 (training stopped early) |
| Best Epoch | 33 |
| Number of Classes | 4 |
| Device | NVIDIA GPU / CPU |

### Model Performance

The following are the validation results obtained during model training:

| Metric | Result |
|---|---:|
| Precision | 92.8% |
| Recall | 90.0% |
| mAP@50 | 95.9% |
| mAP@50-95 | 79.3% |

*Note: These are validation metrics, not independent test-set results. Performance on new real-world images may vary.*

## 📂 Dataset

The project uses the Apple Fruit Disease Detection dataset from Roboflow Universe.

**Dataset:** [Apple Fruit Disease Detection – Roboflow](https://universe.roboflow.com/fruit-disease-detection-nrrei/apple-fruit-disease-detection-uqrc4/dataset/1)

Dataset distribution:

| Dataset Split | Images |
|---|---:|
| Training | 912 |
| Validation | 87 |
| Testing | 43 |
| **Total** | **1,042** |

The dataset contains 1,122 annotated bounding boxes across four classes.

## 🛠️ Technologies Used

- Python
- YOLO11
- Ultralytics
- PyTorch
- OpenCV
- Streamlit
- NumPy
- Pillow
- CUDA (optional, for NVIDIA GPU acceleration)


## 📁 Project Structure

```text
Apple-Disease-Detection/
│
├── app.py
├── train.py
├── evaluate.py
├── predict.py
├── validate_dataset.py
├── requirements.txt
├── README.md
├── yolo11n.pt
│
├── dataset/
│   ├── data.yaml
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
│
└── runs/
    └── detect/
        └── runs/
            └── apple_disease/
                └── weights/
                    └── best.pt
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/its-Kunal-fr/Apple-Disease-Detection.git
```

### 2. Open the project directory

```bash
cd Apple-Disease-Detection
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment (Windows PowerShell)

```powershell
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal, usually:

```text
http://localhost:8501
```

## 🚀 Model Training

To train the model again using the provided dataset:

```bash
python train.py
```

The trained model is saved under the training output directory.

## 🔍 Model Evaluation

Run the evaluation script:

```bash
python evaluate.py
```

## 🖼️ Making Predictions

Run the prediction script:

```bash
python predict.py
```

## 💻 Hardware

The project was developed and tested on a Windows laptop with:

- Intel Core i5 13th Generation processor.
- NVIDIA GeForce RTX 4050 GPU.
- Python virtual environment.

A compatible CPU can also be used, although inference and training may be slower.

## ⚠️ Limitations

- Detection accuracy depends on image quality and dataset diversity.
- The model is trained for four specific apple fruit categories.
- Predictions may be less reliable under lighting conditions or backgrounds not represented in the training data.
- This project is an academic prototype and should not replace professional agricultural diagnosis.

## 👨‍💻 Author

**GitHub:** [its-Kunal-fr](https://github.com/its-Kunal-fr)

**Project:** Apple Disease Detection Using YOLO11

---

⭐ If you find this project useful, consider giving the repository a star.

