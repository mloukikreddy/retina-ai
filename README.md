# 👁 RetinaAI — Multimodal Diabetic Retinopathy Detection

> AI-powered DR stage classification using dual retinal imaging modalities and a gradient-boosted classifier — deployed as a full-stack Flask web application.

[![Live Demo](https://img.shields.io/badge/🤗%20Live%20Demo-HuggingFace%20Spaces-yellow)](https://loukikreddy22-retina-ai.hf.space)
[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange?logo=tensorflow)](https://tensorflow.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-Classifier-brightgreen)](https://lightgbm.readthedocs.io)
[![Flask](https://img.shields.io/badge/Flask-3.0-black?logo=flask)](https://flask.palletsprojects.com)

---

## 🚀 Live Demo

**👉 [loukikreddy22-retina-ai.hf.space](https://loukikreddy22-retina-ai.hf.space)**

Upload a fundus photograph + OCT scan → get DR stage + confidence score in ~1 second.

---

## 🧠 Model Architecture

```
Fundus Image ──► DenseNet121 (ImageNet) ──► GAP ──► 1024-dim ──┐
                                                                  ├──► Concat (2048-dim) ──► StandardScaler ──► LightGBM
OCT Scan     ──► DenseNet121 (ImageNet) ──► GAP ──► 1024-dim ──┘
                                                                                                                    │
                                                                                               No DR  /  NPDR  /  PDR
```

| Component | Model | Output |
|-----------|-------|--------|
| Fundus Stream | DenseNet121 (frozen, ImageNet) | 1024-dim features |
| OCT Stream | DenseNet121 (frozen, ImageNet) | 1024-dim features |
| Fusion | Concatenation + StandardScaler | 2048-dim scaled vector |
| Classifier | LightGBM (500 estimators) | 3-class + confidence % |

---

## 🔬 DR Classification

| Stage | Label | Description |
|-------|-------|-------------|
| 0 | No DR | No signs of diabetic retinopathy |
| 1 | NPDR | Non-Proliferative DR — microaneurysms, haemorrhages present |
| 2 | PDR | Proliferative DR — neovascularisation detected. Urgent referral needed |

---

## 🖥️ Web Application

Built with **Flask** — 5 pages:

- **Home** — Architecture overview + stats
- **Predict** — Drag & drop fundus + OCT upload → instant prediction
- **Result** — DR stage badge, confidence bar, clinical recommendation
- **About** — Pipeline breakdown + model table
- **Sign In** — Auth page (demo credentials)

Features: dark/light theme toggle, drag & drop upload, mobile responsive, toast notifications.

---

## 🗂️ Project Structure

```
retina-ai/
├── app.py              # Flask routes
├── predict.py          # Inference pipeline
├── config.py           # IMG_SIZE, CLASS_MAP
├── train_model.py      # Training script (augmentation + LightGBM)
├── requirements.txt
├── Dockerfile          # HuggingFace Spaces deployment
├── models/
│   ├── fundus_model.h5 # DenseNet121 fundus feature extractor
│   ├── oct_model.h5    # DenseNet121 OCT feature extractor
│   ├── lgbm_model.pkl  # Trained LightGBM classifier
│   └── scaler.pkl      # Fitted StandardScaler
├── static/
│   ├── style.css
│   └── theme.js
└── templates/
    ├── index.html
    ├── predict.html
    ├── result.html
    ├── about.html
    └── signin.html
```

---

## ⚙️ Local Setup

```bash
# Clone
git clone https://github.com/mloukikreddy/retina-ai.git
cd retina-ai

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run
python app.py
```

Open **http://localhost:7860** in browser.

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| Backend | Flask 3.0, Python 3.10 |
| Deep Learning | TensorFlow 2.19, Keras, DenseNet121 |
| ML Classifier | LightGBM, Scikit-learn |
| Image Processing | OpenCV, NumPy |
| Frontend | HTML5, CSS3, Vanilla JS |
| Deployment | Docker, HuggingFace Spaces |

---

## 📊 Training Details

- **Dataset:** ~410 paired Fundus + OCT samples
- **Augmentation:** x8 per sample → ~3,690 total (flip, rotate, brightness, zoom, gaussian noise)
- **Feature extraction:** DenseNet121 GlobalAveragePooling2D (frozen ImageNet weights)
- **Classifier:** LightGBM · 500 estimators · lr=0.05 · `class_weight=balanced`
- **Split:** 80/20 stratified train-test

---

## 🔗 Related

> **Colab Notebook Version** (with SHAP explainability + patient-wise split):  
> 🔗 [Multi-Modal DR Classification Notebook](https://colab.research.google.com/drive/1cicYcsZg32RakChF-KBpIsVJRpRTVKcM?usp=sharing)

> **Mini Project (VGG16 single-modal):**  
> 🔗 [github.com/mloukikreddy/Diabetic-Retinopathy](https://github.com/mloukikreddy/Diabetic-Retinopathy)

---

## 👤 Author

**Mekala Loukik Reddy**  
Final Year B.Tech AI · Anurag University · Hyderabad  
🔗 [LinkedIn](https://linkedin.com/in/mekala-loukik-reddy-a717b4287) · [GitHub](https://github.com/mloukikreddy)

---

## 📄 License

Academic and research use only.