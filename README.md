# 🌾 AgriVision AI – Backend

AgriVision AI is an AI-powered agriculture system for detecting and analyzing weeds in crop-field images using computer vision and deep learning.

This repository contains the **backend API** of the AgriVision AI system. It provides image upload, AI inference, weed analysis, detection history, and result management.

---

## 🚀 Project Overview

The backend receives crop-field images from the AgriVision AI frontend and processes them using a trained YOLO model.

The system provides:

- 🌱 Crop detection
- 🌿 Weed detection
- 🎯 Detection confidence
- 🔢 Crop and weed count
- 📊 Weed density
- ⚠️ Weed severity classification
- 🔥 Weed hotspot information
- 🕘 Detection history
- 🖼️ Annotated detection images

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Backend development |
| Flask | REST API |
| YOLO | Object detection |
| Ultralytics | YOLO implementation |
| OpenCV | Image processing |
| NumPy | Numerical processing |
| Pillow | Image handling |
| SQLite | Detection history database |
| Gunicorn | Production server |

---

## 🤖 AI Model

The backend uses a trained **YOLO11n** model for crop and weed detection.

### Classes

```text
0 → Crop
1 → Weed
