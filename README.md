# 🦠 Social Distancing Analyser System
- Real-time Social Distancing Analyser System using YOLOv8, OpenCV, and Gradio. Supports video upload, live webcam monitoring, and analytics dashboard with risk detection.
- It supports both **video upload** and **live webcam monitoring**, along with a live analytics dashboard and session history tracking.

## 🌐 Live Demo
👉 https://codeboy710-social-distancing-detector.hf.space

> ⚠️ **Note:** The live demo runs on a CPU-based environment, so performance may be significantly slower than real-time.  
> For best results and smooth real-time processing, please run the project locally on a GPU-enabled system.


## 🚀 Features

- 🎯 Real-time person detection using YOLOv8
- 📏 Distance-based social distancing violation detection
- 📊 Live analytics dashboard (safety score, density, risk level)
- 📹 Video upload processing
- 📷 Live webcam monitoring
- 🧠 Risk classification (LOW / HIGH / CRITICAL)
- 🕘 History tracking of analyzed sessions
- ⚡ Lightweight and fast inference

---
### Detection Screens

<p float="left">
  <img src="AppSS/Screenshot (307).png" width="250"/>
  <img src="AppSS/Screenshot (308).png" width="250"/>
  <img src="AppSS/Screenshot (309).png" width="250"/>
</p>

<p float="left">
  <img src="AppSS/Screenshot (310).png" width="250"/>
  <img src="AppSS/Screenshot (311).png" width="250"/>
  <img src="AppSS/Screenshot (312).png" width="250"/>
</p>
---

## 🧠 Tech Stack

- Python 🐍
- YOLOv8 (Ultralytics)
- OpenCV
- NumPy
- SciPy (distance calculations)
- Gradio (UI framework)

## 📂 Project Structure


Social Distancing web/  
│  
├── app.py (Main application)  
├── requirements.txt (Dependencies)  
├── .gitignore  
├── README.md  
└── assets/  (optional images/screenshots)


## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/social-distancing-web.git
cd social-distancing-web
```
## 2. Install dependencies
```bash
pip install -r requirements.txt
```

## 3. Run the application
```bash
python app.py

```
⚠️ Note:
YOLOv8 model (yolov8n.pt) is automatically downloaded on first run.
Do NOT upload .pt or .weights files to GitHub (handled via .gitignore).
Ensure webcam permissions are enabled for live mode.

```
👨‍💻 Author

Gourav Adhikary

Full Stack Developer (MERN)
AI/ML Enthusiast
Focus: GenAI + Computer Vision
