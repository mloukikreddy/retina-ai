---
title: RetinaAI
emoji: 👁
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# RetinaAI — Multimodal Diabetic Retinopathy Detection

A dual-stream deep learning system for automated DR staging using fundus photographs and OCT scans.

## Architecture
- **Fundus Stream:** DenseNet121 → 1024-dim features
- **OCT Stream:** DenseNet121 → 1024-dim features
- **Fusion:** Concatenated 2048-dim → StandardScaler → LightGBM
- **Output:** 3-class DR stage (No DR / NPDR / PDR) + confidence %

## Usage
1. Upload a fundus image
2. Upload an OCT scan
3. Click **Analyze Images**
4. Get DR stage + clinical recommendation

## Tech Stack
Python · Flask · TensorFlow · LightGBM · OpenCV

## Author
[Mekala Loukik Reddy](https://github.com/mloukikreddy) — Final Year B.Tech AI, Anurag University