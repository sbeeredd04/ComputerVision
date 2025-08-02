# Computer Vision Learning Repository

A comprehensive repository for learning computer vision concepts with practical examples using YOLO object detection and MediaPipe hand tracking.

## 🎯 Project Overview

This repository contains educational materials and practical implementations for:
- **Object Detection** using YOLO (You Only Look Once)
- **Hand Tracking** using MediaPipe
- **Real-time Computer Vision** applications

## 📁 Repository Structure

```
ComputerVision/
├── data/           # Data files (images, models, outputs)
├── src/            # Source code
│   ├── models/     # Model implementations
│   ├── utils/      # Utility functions
│   └── scripts/    # Demo scripts
├── notebooks/      # Jupyter notebooks for tutorials
├── docs/          # Documentation
├── tests/         # Unit tests
└── examples/      # Example outputs and images
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenCV
- Ultralytics (YOLO)
- MediaPipe

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ComputerVision
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run a demo:
```bash
# Object detection demo
python src/scripts/object_detection_demo.py

# Hand tracking demo
python src/scripts/hand_tracking_demo.py
```

## 📚 Learning Path

### 1. Object Detection with YOLO
- **Tutorial**: `notebooks/tutorials/yolo_tutorial.ipynb`
- **Demo**: `src/scripts/object_detection_demo.py`
- **Model**: `src/models/yolo_detector.py`

### 2. Hand Tracking with MediaPipe
- **Demo**: `src/scripts/hand_tracking_demo.py`
- **Model**: `src/models/hand_tracker.py`

## 🛠️ Key Features

- **Real-time Processing**: Live webcam feeds for object detection and hand tracking
- **Educational Focus**: Well-documented code with explanations
- **Modular Design**: Reusable components for different applications
- **Visualization Tools**: Custom drawing utilities for bounding boxes and landmarks

## 📖 Documentation

- [Setup Guide](docs/setup.md)
- [Tutorials](docs/tutorials.md)

## 🤝 Contributing

This is an educational repository. Feel free to:
- Add new computer vision examples
- Improve documentation
- Fix bugs or add features
- Share your learning experiences

## 📄 License

This project is for educational purposes.

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLO implementation
- [MediaPipe](https://mediapipe.dev/) for hand tracking
- [OpenCV](https://opencv.org/) for computer vision utilities 