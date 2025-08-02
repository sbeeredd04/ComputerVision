# Setup Guide

This guide will help you set up the Computer Vision Learning Repository on your system.

## Prerequisites

### System Requirements
- **Operating System**: Windows 10/11, macOS, or Linux
- **Python**: 3.8 or higher
- **RAM**: At least 4GB (8GB recommended)
- **Storage**: At least 2GB free space
- **Camera**: Webcam for real-time demos (optional)

### Python Installation

1. **Download Python**: Visit [python.org](https://www.python.org/downloads/) and download Python 3.8 or higher
2. **Install Python**: Follow the installation instructions for your operating system
3. **Verify Installation**: Open a terminal/command prompt and run:
   ```bash
   python --version
   ```

## Repository Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd ComputerVision
```

### 2. Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download YOLO Model (Optional)

The YOLO model will be downloaded automatically on first use, but you can download it manually:

```python
from ultralytics import YOLO
model = YOLO('yolov8s.pt')
```

## Verification

### Test Installation

1. **Test OpenCV**:
   ```python
   import cv2
   print(f"OpenCV version: {cv2.__version__}")
   ```

2. **Test YOLO**:
   ```python
   from ultralytics import YOLO
   print("YOLO imported successfully")
   ```

3. **Test MediaPipe**:
   ```python
   import mediapipe as mp
   print("MediaPipe imported successfully")
   ```

### Run a Quick Demo

```bash
# Object detection demo
python src/scripts/object_detection_demo.py

# Hand tracking demo
python src/scripts/hand_tracking_demo.py
```

## Troubleshooting

### Common Issues

#### 1. Camera Not Working
- **Windows**: Check camera permissions in Settings > Privacy > Camera
- **macOS**: Check camera permissions in System Preferences > Security & Privacy
- **Linux**: Install `v4l-utils` and check with `v4l2-ctl --list-devices`

#### 2. YOLO Model Download Issues
- Check internet connection
- Try downloading manually:
  ```python
  from ultralytics import YOLO
  model = YOLO('yolov8s.pt')
  ```

#### 3. Import Errors
- Ensure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`

#### 4. Performance Issues
- Close other applications using the camera
- Reduce camera resolution in the demo scripts
- Use a smaller YOLO model (e.g., `yolov8n.pt`)

### Getting Help

If you encounter issues:

1. Check the [Troubleshooting](#troubleshooting) section above
2. Search existing [GitHub Issues](link-to-issues)
3. Create a new issue with:
   - Operating system and version
   - Python version
   - Error message
   - Steps to reproduce

## Next Steps

After successful setup:

1. Read the [Tutorials](tutorials.md) guide
2. Explore the [notebooks/](notebooks/) directory
3. Try the example scripts in [src/scripts/](src/scripts/)
4. Experiment with your own computer vision projects!

## Development Setup

For contributors:

1. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install pytest black flake8
   ```

2. Set up pre-commit hooks (optional):
   ```bash
   pip install pre-commit
   pre-commit install
   ```

3. Run tests:
   ```bash
   pytest tests/
   ``` 