# 🛡️ ZaBir — Crowd Behavior & Risk Detection System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-orange" />
  <img src="https://img.shields.io/badge/GUI-CustomTkinter-green" />
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-red?logo=opencv" />
  <img src="https://img.shields.io/badge/ByteTrack-Object%20Tracking-purple" />
  <img src="https://img.shields.io/badge/Status-Active-success" />
</p>

<p align="center">
  <b>Intelligent Crowd Monitoring, Behavior Analysis & Anomaly Detection</b>
  <br/>
  <i>Monitor Crowds. Analyze Movement. Detect Unusual Activities.</i>
</p>

---

## 📸 Project Overview

**Crowd Behavior & Risk Detection** is an AI-powered crowd monitoring and behavior analysis system developed using Python, YOLOv8, OpenCV, ByteTrack, and CustomTkinter.

The system analyzes video footage to detect and track individuals, estimate crowd density, monitor movement patterns, and flag potentially abnormal activities such as fighting, weapon detection, sudden movement, and suspicious gatherings.

By combining deep learning-based object detection with computer vision techniques, background subtraction, optical flow, and rule-based anomaly analysis, Crowd Sentinel provides an intelligent approach to crowd surveillance.

The application features a modern dark-themed desktop interface with live analytics, automated evidence collection, heatmap generation, incident logging, and audio alerts.

## ✨ Key Features

- 🎯 **Dual-Model Architecture** — Combines standard YOLOv8 person detection with a custom-trained activity detection model.
- 👥 **Human Detection & Tracking** — Detect and track individuals using YOLOv8 and ByteTrack.
- 🚨 **Crime & Activity Detection** — Identify custom-trained classes such as fighting, violence, and weapons.
- 🧠 **Smart Abnormality Engine** — Combines multiple detection signals to flag unusual activities.
- 📊 **Crowd Density Analysis** — Estimate crowd density and identify overcrowded areas.
- 🏃 **Sudden Movement Detection** — Identify unusual movement patterns using optical flow and displacement.
- 👥 **Group Gathering Detection** — Flag groups of three or more people standing close together.
- 🔥 **Movement Heatmaps** — Generate visual heatmaps representing crowd movement.
- 📸 **Automated Evidence Collection** — Save annotated frames when abnormal activities are detected.
- 🔊 **Audio Alarm System** — Trigger background alarms based on incident severity.
- 📜 **Incident Logging** — Maintain a history of abnormal activities in text files.
- 📈 **Live Analytics HUD** — Display human count, crowd density, FPS, and system status.
- 🌙 **Modern Dark GUI** — Responsive desktop interface built with CustomTkinter.
- ⚡ **Multithreaded Processing** — Keep video processing and interface operations responsive.

---

## 🧠 Intelligent Anomaly Detection

Crowd Behavior & Risk Detection uses multiple detection signals to identify potentially abnormal crowd behavior.

### 1️⃣ Fighting & Weapon Detection 🚨

Uses a custom-trained YOLO model to detect configured activity classes.

**Capabilities:**
- Fighting detection.
- Weapon detection.
- Violence-related activity detection.
- Custom activity classification.

> Detection accuracy depends on the quality, diversity, and training of your custom dataset.

### 2️⃣ Crowd Density Detection 👥

Analyzes the number and distribution of detected people to estimate crowd density.

**Capabilities:**
- Real-time person counting.
- Crowd density estimation.
- Overcrowding alerts.
- Crowd monitoring.

### 3️⃣ Sudden Movement Detection 🏃

Uses optical flow and movement displacement to identify unusual motion patterns.

**Capabilities:**
- Movement trajectory analysis.
- Sudden displacement detection.
- Motion intensity estimation.
- Unusual movement alerts.

---

## 🏗️ System Architecture

Crowd Behavior & Risk Detection combines multiple computer vision components into a single processing pipeline.

| Component | Responsibility |
|---|---|
| YOLOv8n | Person detection |
| Custom YOLO Model | Custom activity and object detection |
| ByteTrack | Individual object tracking |
| OpenCV MOG2 | Background subtraction |
| Optical Flow | Movement analysis |
| Rule Engine | Abnormal activity classification |
| CustomTkinter | Desktop dashboard |
| Logging System | Incident record management |
| Heatmap Generator | Crowd movement visualization |

---

## 📁 Project Structure

```text
Crowd Behavior & Risk Detection/
│
├── 📄 main.py                       # Main application and GUI
├── 📄 detector.py                   # AI detection and anomaly engine
├── 📄 utils.py                      # Logging, audio, and file utilities
├── 📄 requirements.txt              # Project dependencies
├── 📄 README.md                     # Project documentation
│
├── 🤖 best.pt                       # Custom-trained activity model
├── 🤖 yolov8n.pt                    # Standard person detection model
│
├── 📂 abnormal_frames/              # Saved abnormal activity images
│
├── 📂 heatmaps/                     # Generated crowd movement heatmaps
│
└── 📂 logs/
    └── 📄 abnormal_activity_log.txt  # Incident history
```

> 💡 The `abnormal_frames/`, `heatmaps/`, and `logs/` directories may be generated automatically when the application runs.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| 🐍 Python | Core programming language |
| 🤖 Ultralytics YOLOv8 | Object and activity detection |
| 🎯 ByteTrack | Object tracking |
| 👁️ OpenCV | Video processing and computer vision |
| 🌊 Optical Flow | Movement and trajectory analysis |
| 🧠 MOG2 | Background subtraction |
| 🖥️ CustomTkinter | Modern desktop GUI |
| 🔊 Pygame / Winsound | Audio alarms |
| 📊 NumPy | Numerical calculations |

---

## ⚙️ Requirements

Before installing Crowd Behavior & Risk Detection System, make sure you have:

- Python 3.10 or Python 3.11 (recommended).
- Windows, Linux, or macOS.
- Visual Studio Code (recommended).
- At least 4 GB RAM (8 GB recommended).
- CPU support for basic video processing.
- NVIDIA GPU (optional, for accelerated inference).
- Internet connection for downloading model weights and dependencies.

---

## ▶️ Running the Application

After installing the dependencies and configuring the models, launch the application:

```bash
python main.py
```

The Crowd Behavior & Risk Detection System desktop dashboard will open.

---

## 🎮 How to Use

Follow these steps to start crowd monitoring:

1. 🚀 Launch the application using `python main.py`.
2. 📂 Click **Load YOLO Model** to select your custom `.pt` model.
3. 🎬 Click **Select Video** to choose surveillance footage.
4. ▶️ Click **Start Detection** to begin video analysis.
5. 👥 Monitor detected people and their tracking IDs.
6. 📊 Observe the live analytics HUD.
7. 🚨 Review abnormal activity alerts.
8. 🔊 Listen for audio alarms when configured detection rules are triggered.
9. 📸 Open the saved abnormal activity screenshots.
10. 🔥 Review generated movement heatmaps.
11. 📜 Check the incident history in the logs folder.

---

## 📊 Live Analytics Dashboard

The dashboard provides real-time monitoring information.

| Metric | Description |
|---|---|
| 👥 Humans Detected | Number of people detected in the current frame |
| 📊 Crowd Density | Estimated crowd density |
| ⚡ FPS | Processing speed in frames per second |
| 🟢 Normal Status | No configured abnormality rule currently triggered |
| 🔴 Abnormal Status | One or more abnormality rules triggered |

---

## 🔥 Evidence Collection & Logging

Crowd Behavior & Risk Detection automatically saves evidence when configured abnormality conditions are detected.

### 📸 Abnormal Activity Frames

Saved inside:

```text
abnormal_frames/
```

These images contain annotated video frames for later review.

### 🔥 Crowd Movement Heatmaps

Saved inside:

```text
heatmaps/
```

Heatmaps provide a visual representation of movement intensity across the video.

### 📜 Incident Logs

Stored inside:

```text
logs/abnormal_activity_log.txt
```

The log file maintains records of detected abnormal activities.

---

## ⚙️ How the Detection Pipeline Works

The application follows a multi-stage computer vision pipeline.

**Step 1 — Frame Extraction**

OpenCV reads the input video and resizes frames for processing.

**Step 2 — Person Detection & Tracking**

YOLOv8n detects people, while ByteTrack assigns tracking IDs and follows their movement.

**Step 3 — Custom Activity Detection**

The custom YOLO model analyzes the frame for configured activity and object classes.

**Step 4 — Background Subtraction**

OpenCV MOG2 separates moving foreground objects from the background to estimate motion intensity.

**Step 5 — Optical Flow Analysis**

Movement patterns and displacement are analyzed to identify unusual motion.

**Step 6 — Smart Rule Engine**

The system combines:
- Fighting and weapon detections.
- Crowd density.
- Sudden movement.
- Group proximity.

Configured conditions determine whether the system flags the current activity as abnormal.

**Step 7 — Evidence & Logging**

The application saves relevant annotated frames, generates heatmaps, records incidents, and triggers configured audio alarms.

---

## ⚠️ Detection Notes & Limitations

Crowd Behavior & Risk Detection is an experimental AI-based monitoring prototype. Its results should be treated as alerts for human review, not definitive conclusions.

- **Fighting Detection:** Depends on the custom model and its training data. Object detection alone cannot reliably establish that a fight is occurring.
- **Weapon Detection:** Depends on the weapon classes included in the custom dataset and can produce false positives or missed detections.
- **Crowd Density:** Estimates crowd conditions using detected people and configured thresholds.
- **Sudden Movement:** Optical flow and displacement can be affected by camera movement, lighting, perspective, and scene cuts.
- **Group Gathering:** Close proximity does not necessarily indicate suspicious behavior.
- **Tracking:** ByteTrack IDs are temporary tracking identifiers, not verified identities.
- **Background Subtraction:** MOG2 may respond to lighting changes, shadows, and background motion.
- **Audio Alerts:** Depend on the operating system and available audio libraries.

Always review flagged frames manually before taking any action.

---

## 🔧 Troubleshooting

### ❌ 1. The System Detects Zero Humans

**Possible causes:**
- `yolov8n.pt` is missing.
- The video is not loading correctly.
- Detection confidence is too high.
- The model is not configured correctly.

**Solution:**

Ensure the person detection model is available:

```text
yolov8n.pt
```

Also verify that your input video is valid.

### ❌ 2. FileNotFoundError: best.pt

**Solution:**

Place your custom-trained model in the project root directory or select it through the model-loading interface.

### ❌ 3. Audio Alarm Is Not Playing

On Windows, the application may use `winsound`.

On Linux or macOS, it may use Pygame.

Install Pygame if required:

```bash
pip install pygame
```

### ❌ 4. ModuleNotFoundError

Install the required dependencies:

```bash
pip install ultralytics opencv-python customtkinter numpy supervision pygame
```

### ❌ 5. Low FPS or Slow Processing

Try:
- Reducing video resolution.
- Using shorter video clips.
- Closing unnecessary background applications.
- Using a supported NVIDIA GPU.
- Reducing unnecessary processing operations.

---

## 🔮 Future Improvements

- 🌐 Web-based monitoring dashboard.
- 📹 Multi-camera video analysis.
- 📡 IP camera and RTSP integration.
- 📧 Email and SMS notifications.
- 📊 Advanced crowd analytics.
- 🧠 Deep learning-based behavior recognition.
- ☁️ Cloud-based incident storage.
- 🔔 Configurable alarm severity levels.
- 📈 Historical crowd movement reports.

---

## 🔐 Privacy & Responsible Use

- Analyze only video footage you are authorized to use.
- Respect individual privacy and applicable surveillance laws.
- Protect recorded evidence and incident logs.
- Do not interpret automated alerts as proof of criminal activity.
- Ensure human verification before making important decisions.

---

## 👨‍💻 Developer

**ZaBir Hasan**

🌐 Website: [www.zabirhasan.com](https://www.zabirhasan.com)

---

<p align="center">
  <b>🛡️ Crowd Behavior & Risk Detection System</b>
  <br/>
  <i>Smarter Crowd Monitoring. Intelligent Detection. Safer Environments.</i>
  <br/><br/>
  Made with ❤️ using Python, YOLOv8, and Computer Vision.
</p>
