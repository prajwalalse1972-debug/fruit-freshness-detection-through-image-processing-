# fruit-freshness-detection-through-image-processing-

# 🍎 FreshScan — Smartphone-Based Fruit Freshness Detection

> A lightweight, GPU-free fruit freshness detection system using **KNN classification** and **handcrafted image features**, deployable as a desktop GUI or smartphone-accessible web app.

![Python](https://img.shields.io/badge/Python-3.14-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)
![Flask](https://img.shields.io/badge/Flask-Web%20App-lightgrey?logo=flask)
![Accuracy](https://img.shields.io/badge/Accuracy-90.81%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Results](#results)
- [Team](#team)
- [References](#references)

---

## 📖 Overview

**FreshScan** is an interdisciplinary project developed for **1BPRJ208 – Interdisciplinary Project Based Learning** at **Dayananda Sagar College of Engineering (DSCE), VTU**.

It detects whether a fruit is **FRESH** or **ROTTEN** by analysing a photo taken from any smartphone or camera, using a 575-dimensional feature vector and a K-Nearest Neighbours (KNN) classifier — no GPU or deep learning framework required.

---

## ✨ Features

- 🔍 **575-D Feature Vector** — Color (512 HSV histogram) + Texture (59 LBP) + Shape (4 contour features)
- 🤖 **KNN Classifier** — k=5, trained on 4,789 labeled images across 5 fruit types
- 🖥️ **Tkinter Desktop GUI** — Threaded inference, result display, CSV logging, PDF export
- 📱 **Flask Web App** — Mobile-responsive; accessible from any smartphone on the same Wi-Fi
- 📊 **CSV Logging** — Auto-saves every prediction with timestamp and confidence score
- 📄 **PDF Report Export** — One-click PDF generation via fpdf2
- 🍌 **5 Fruit Types** — Banana, Apple, Mango, Orange, Strawberry

---

## 📁 Project Structure

```
FreshScan/
│
├── app/
│   ├── gui.py               # Tkinter desktop application
│   └── web_app.py           # Flask web application
│
├── models/
│   └── fruit_model.pkl      # Trained KNN model (joblib)
│
├── utils/
│   ├── feature_extractor.py # HSV histogram + LBP + Shape features
│   ├── knn_classifier.py    # Pure NumPy KNN implementation
│   ├── preprocess.py        # Image resize, masking, HSV conversion
│   └── report_generator.py  # PDF + CSV export utilities
│
├── static/
│   ├── css/                 # Web app stylesheets
│   ├── js/                  # Frontend scripts
│   └── uploads/             # Temporary uploaded images
│
├── templates/
│   └── index.html           # Flask HTML template
│
├── dataset/                 # Training images (not pushed to GitHub)
│
├── reports/                 # Generated PDF/CSV reports
│
├── train.py                 # Model training script
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/FreshScan.git
cd FreshScan
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Desktop GUI (Tkinter)
```bash
python app/gui.py
```

### Web App (Flask — accessible from smartphone)
```bash
python app/web_app.py
```
Then open `http://<your-PC-IP>:5000` on any smartphone connected to the same Wi-Fi.

### Train the model (optional — model already included)
```bash
python train.py
```

---

## 🔬 How It Works

```
Input Image
    │
    ▼
Preprocessing  ──►  Resize + HSV Conversion + Fruit Mask
    │
    ▼
Feature Extraction
    ├── Color Histogram  →  512 values  (8×8×8 HSV bins)
    ├── LBP Texture      →   59 values  (Local Binary Pattern)
    └── Shape Features   →    4 values  (Circularity, Extent, Area, Perimeter)
    │
    ▼
575-D Feature Vector
    │
    ▼
KNN Classifier (k=5)  ──►  Majority Vote among 5 Nearest Neighbours
    │
    ▼
Output: FRESH ✅ or ROTTEN ❌ + Confidence Score
```

---

## 📊 Results

| Metric        | Value  |
|---------------|--------|
| Accuracy      | 90.81% |
| Precision     | 0.93   |
| Recall        | 0.88   |
| Training Data | 4,789 images |
| Feature Dims  | 575    |
| KNN k value   | 5      |

**Fruits supported:** Banana 🍌 · Apple 🍎 · Mango 🥭 · Orange 🍊 · Strawberry 🍓

---

## 👥 Team

| Name | USN | Department |
|------|-----|------------|
| Prajwal Alse | 1DS25BT038 | BT |
| Pavan Kumar| 1DS25BT037 | BT |


**Guide:** Dr. Priya S, Assistant Professor, Dept. of CSE, DSCE

---

## 📚 References

1. Muxiddinov et al., "Fruits and Vegetables Freshness Categorization Using Deep Learning," *CMC*, 2022
2. Amin et al., "Improved Classification for Fruit/Vegetable Freshness," *Sensors*, 2022
3. Zhang et al., "Fruit Freshness Detection Based on Multi-Task CNN," *CRFS*, 2024
4. Yuan & Chen, "Vegetable and Fruit Freshness Detection via Deep Features and PCA," *CRFS*, 2024
5. Baglat et al., "Non-Destructive Banana Ripeness Detection," *Sensors*, vol. 23, 2023

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">Made with ❤️ at Dayananda Sagar College of Engineering, Bengaluru</p>
