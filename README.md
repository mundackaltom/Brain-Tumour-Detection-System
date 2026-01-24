# 🧠 Brain Tumour Detection System (YOLOv8)

A deep learning-based medical imaging project that detects **brain tumours from MRI scans** using a **custom-trained YOLOv8 model**.  
Includes an interactive **Streamlit web app** for real-time image upload and tumour detection.

---

## 📌 Project Overview

Brain tumour detection is an important healthcare application where early diagnosis can improve treatment planning and patient outcomes.  
This project uses **YOLOv8 (You Only Look Once)** for object detection to identify tumour regions in MRI images and visualize predictions using bounding boxes.

✅ Detects tumour presence from MRI images  
✅ Highlights tumour region using bounding boxes  
✅ Streamlit-based web interface for easy testing  
✅ Custom-trained `.pt` model included in the repository

---

## 🚀 Features

- 📷 Upload MRI scans and get tumour detection results instantly  
- 🎯 YOLOv8-based object detection (fast + accurate)  
- 🧠 Deep learning model trained on brain MRI tumour dataset  
- 📦 Easy to run locally with Python + Streamlit  
- 🖼️ Output image preview with predicted tumour region

---

## 🛠️ Tech Stack

- **Python**
- **YOLOv8 (Ultralytics)**
- **OpenCV**
- **NumPy**
- **Streamlit**
- **Matplotlib (optional for visualization)**

---

## 📂 Repository Structure

```bash
Brain-Tumour-Detection-System/
│
├── README.md               # Project documentation
├── brain.py                # Core code / inference logic
├── newapp.py               # Streamlit app file
├── Tumour.ipynb            # Notebook (training/testing experiments)
├── best.pt                 # Custom trained YOLOv8 weights
├── yolov8s.pt              # Base YOLOv8 weights (if included)
├── result.jpg              # Sample output image
├── pics.docx               # Supporting images/screenshots (optional)
└── .DS_Store               # System file (can be ignored)


