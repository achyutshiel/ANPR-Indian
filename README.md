# 🚘 Indian Vehicle ANPR System

![Build](https://img.shields.io/badge/build-passing-brightgreen)
![Status](https://img.shields.io/badge/status-active-success)
![Model](https://img.shields.io/badge/model-YOLOv8-blue)
![Framework](https://img.shields.io/badge/framework-Ultralytics-red)
![Dataset](https://img.shields.io/badge/dataset-Indian%20Vehicles-orange)
![mAP50](https://img.shields.io/badge/mAP50-0.8%2B-brightgreen)
![Deployment](https://img.shields.io/badge/deployment-OpenCV%20Inference-blueviolet)
![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

---

## 📌 Project Overview

---

### 🔥 Task 1: Real-World License Plate Detection Pipeline

This project presents a **robust and scalable computer vision framework** for real-time automatic number plate recognition (ANPR), specifically optimized for **Indian vehicle license plates in unconstrained environments**.

Unlike standard dataset tutorials, this approach is engineered to handle the domain gap between high-quality static images and real-world, high-speed CCTV or dashcam footage. It leverages **YOLOv8** to reliably detect:

- Car License Plates
- Commercial Truck License Plates
- Two-Wheeler Plates

The system is engineered under **practical, real-world constraints**, addressing:

- Small object detection (plates at a distance appearing <30px)
- High-speed motion blur and H.264 video compression artifacts
- Complex XML (Pascal VOC) to YOLO bounding box format conversions

This makes it highly suitable for **logistics, fleet management, and Transport Management Systems (TMS)**.

---

### 🧠 Task 2: Advanced Data Transformation & Inference Optimization

A key component of this project is the transition from raw, disorganized data to a structured, production-ready inference pipeline.

The pipeline is optimized through:

- Dynamic ETL scripting (Pascal VOC `.xml` to YOLO `.txt` normalized coordinates)
- High-resolution inference scaling (`imgsz=1280`) for distant object detection
- Aggressive confidence threshold tuning for high-recall tracking

This enables:

- High-accuracy detection on heavy commercial vehicles (e.g., trucks)
- Automated dataset preparation and train/validation splitting
- Video frame-by-frame processing via OpenCV

---

## 🧠 Technologies Used

- Python
- YOLOv8 (Ultralytics)
- OpenCV (Video Processing & Annotation)
- Kaggle API (Automated Data Ingestion)
- PyYAML (Dynamic Config Generation)
- Google Colab (Training Environment)

---

## 📁 Repository Structure

```bash
Indian-ANPR-YOLOv8/
│
├── models/
│   └── best.pt                  # Best trained YOLOv8 weights (if < 100MB)
│
├── notebooks/
│   └── ANPR_Training.ipynb      # End-to-end Colab training workflow
│
├── scripts/
│   └── inference_video.py       # Production video detection script
│
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```
---
## 📊 Results & Evaluation
📌 Detection Output (High-Speed Truck Inference)
The model demonstrates:
- Stable bounding box localization on distant commercial vehicles
- Robust detection despite varying lighting and camera angles
- Successful tracking of regional Indian plates (e.g., RJ 14)
<img width="1024" height="768" alt="Screenshot (3)" src="https://github.com/user-attachments/assets/94e81ef6-95cd-4476-987c-1f6cb9274d4b" />

---

## 📈 Performance Metrics
<img width="792" height="269" alt="Screenshot (4)" src="https://github.com/user-attachments/assets/66d927b7-c47c-4b8c-8788-fab3f7e81764" />

---

## ⚙️ Usage
**1. Environment Setup & Dependencies**
```sh
git clone [https://github.com/AchyutKumarPandey/Indian-ANPR-YOLOv8.git](https://github.com/AchyutKumarPandey/Indian-ANPR-YOLOv8.git)
cd Indian-ANPR-YOLOv8
pip install -r requirements.txt
```
**2. Automated Data Prep & Training (via Script/Notebook)**
```sh
from ultralytics import YOLO

# Initialize the pre-trained Nano model
model = YOLO("yolov8n.pt")

# Train the model on the generated YAML
model.train(
    data="yolo_format/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    project="runs",
    name="lymkit_anpr_model"
)
```
**3. Video Inference (High-Resolution Tracking)**
```sh
import cv2
from ultralytics import YOLO

model = YOLO("models/best.pt")
cap = cv2.VideoCapture("input_video.mp4")
out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (1920, 1080))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
        
    # Using imgsz=1280 to catch small plates at a distance
    results = model.predict(frame, conf=0.15, imgsz=1280, verbose=False)
    annotated_frame = results[0].plot()
    out.write(annotated_frame)
    
cap.release()
out.release()
```

---

## 🚧 Challenges Addressed
- ***Small Object Detection:*** Mitigated by scaling up inference resolution for highway/CCTV footage.
- ***Format Incompatibilities:*** Engineered a custom XML-to-YOLO conversion script to handle missing bounding box tags.
- ***Domain Gap:*** Addressed the disparity between static, close-up Kaggle dataset images and real-world, motion-blurred video inputs.

---

## 🚀 Applications
- Supply Chain & Fleet Management (TMS)
- Automated Toll Collection Systems
- Smart City Traffic Monitoring
- E-proof of Deliveries at Logistics Hubs

---

## 🔮 Future Work
- Integration of ***EasyOCR*** or ***Tesseract*** within the bounding box crop to extract text.
- Regex validation strictly for Indian RTO formats (e.g., MH 12 AB 1234).
- Deployment of the inference pipeline as a REST API using FastAPI.

---

## 📌 Conclusion
This project demonstrates a practical and deployment-oriented deep learning pipeline for Automatic Number Plate Recognition on Indian vehicles. It bridges the gap between raw, unstructured dataset experimentation and real-world video inference, emphasizing robustness, adaptability, and operational feasibility for logistics and supply chain tech.

---

## 📄 License
This project is licensed under the MIT License.
You are free to use, modify, and distribute this project with proper attribution.

---

## 👤 Author
#### Achyut Kumar Pandey
This work reflects a practical approach to ***Computer Vision engineering***, focusing on building robust, real-time object detection systems capable of operating in challenging real-world logistics and transportation environments.

---
