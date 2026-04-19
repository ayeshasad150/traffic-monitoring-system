# 🚦 AI Traffic Monitoring System

An AI-based Traffic Monitoring System using YOLOv8 for vehicle detection, 
helmet detection, number plate recognition, and signal violation detection.

## 🎯 Features
- 🚗 Vehicle Detection & Classification (Car, Bus, Truck, Motorbike)
- ⛑️ Helmet Detection (Helmet / No-Helmet)
- 🔤 Number Plate Detection & OCR
- 🚦 Signal Violation Detection
- 📊 Speed Estimation
- 📥 Excel Violation Report Export

## 🛠️ Technologies
- YOLOv8 (Ultralytics)
- EasyOCR
- Streamlit
- OpenCV
- Python 3.11

## 📊 Model Performance
| Model | mAP50 | Classes |
|-------|-------|---------|
| Vehicle | 95.6% | car, bus, truck, motorbike |
| Helmet | 59% | Helmet, No-Helmet |
| Plate | Pretrained | license_plate |

## 🚀 How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Dataset
- Vehicle Detection: Roboflow (1000 images)
- Helmet Detection: Roboflow (3835 images)
- License Plate: Roboflow Pakistan (6160 images)