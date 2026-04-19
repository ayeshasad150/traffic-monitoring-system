# 🚦 AI Traffic Monitoring System

AI-based Traffic Monitoring System using YOLOv8 for vehicle detection, 
helmet detection, number plate recognition and signal violation detection.

## 🎯 Features
- 🚗 Vehicle Detection (Car, Bus, Truck, Motorbike)
- ⛑️ Helmet Detection (Helmet / No-Helmet)
- 🔤 Number Plate Detection & OCR
- 🚦 Signal Violation Detection
- 📊 Speed Estimation (km/h)
- 📥 Excel Violation Report

## 📊 Model Performance
| Model | mAP50 | Epochs |
|-------|-------|--------|
| Vehicle Detection | 95.6% | 300 |
| Helmet Detection | 59% | 300 |
| License Plate | Pretrained | - |

## 🛠️ Tech Stack
- YOLOv8 (Ultralytics)
- EasyOCR
- Streamlit
- OpenCV
- Python 3.11

## 🚀 How to Run
```bash
pip install ultralytics streamlit easyocr opencv-python openpyxl
streamlit run app.py
```

## 📁 Dataset Sources
- Vehicle: Roboflow (1000 images)
- Helmet: Roboflow (3835 images)  
- License Plate Pakistan: Roboflow (6160 images)

## 👩‍💻 Developed By
Ayesha | AI Traffic Monitoring Assignment