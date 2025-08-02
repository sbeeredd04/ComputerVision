# Computer Vision Tutorials

This guide provides step-by-step tutorials for learning computer vision concepts using this repository.

## Learning Path

### 1. Getting Started with Object Detection

#### What You'll Learn
- Understanding YOLO (You Only Look Once) architecture
- Real-time object detection
- Bounding box visualization
- Confidence scoring

#### Tutorial Steps

1. **Basic Object Detection**
   ```bash
   # Run the object detection demo
   python src/scripts/object_detection_demo.py
   ```

2. **Process a Single Image**
   ```bash
   # Process an image file
   python src/scripts/object_detection_demo.py --image data/images/your_image.jpg --output results.jpg
   ```

3. **Explore the Code**
   - Open `src/models/yolo_detector.py`
   - Study the `YOLODetector` class
   - Understand the detection pipeline

#### Key Concepts
- **Bounding Boxes**: Rectangular regions around detected objects
- **Confidence Scores**: How certain the model is about its prediction
- **Class Labels**: Names of detected objects (person, car, etc.)

### 2. Hand Tracking with MediaPipe

#### What You'll Learn
- Hand landmark detection
- Real-time hand tracking
- Gesture recognition basics
- MediaPipe integration

#### Tutorial Steps

1. **Basic Hand Tracking**
   ```bash
   # Run the hand tracking demo
   python src/scripts/hand_tracking_demo.py
   ```

2. **Enable Gesture Recognition**
   ```bash
   # Run with gesture recognition
   python src/scripts/hand_tracking_demo.py --gesture
   ```

3. **Adjust Detection Confidence**
   ```bash
   # Use different confidence threshold
   python src/scripts/hand_tracking_demo.py --confidence 0.8
   ```

#### Key Concepts
- **Landmarks**: 21 key points on each hand
- **Connections**: Lines between landmarks showing hand structure
- **Gestures**: Basic hand pose recognition

### 3. Advanced Topics

#### Custom Model Training
1. **Prepare Your Dataset**
   - Collect images of objects you want to detect
   - Label them using tools like LabelImg or Roboflow
   - Organize in YOLO format

2. **Train Custom YOLO Model**
   ```python
   from ultralytics import YOLO
   
   # Load a base model
   model = YOLO('yolov8s.pt')
   
   # Train on your dataset
   model.train(data='path/to/data.yaml', epochs=100)
   ```

#### Performance Optimization
1. **Model Selection**
   - `yolov8n.pt`: Fastest, least accurate
   - `yolov8s.pt`: Balanced speed/accuracy
   - `yolov8m.pt`: More accurate, slower
   - `yolov8l.pt`: Very accurate, slower
   - `yolov8x.pt`: Most accurate, slowest

2. **Camera Settings**
   ```python
   # Lower resolution for better performance
   camera = CameraCapture(width=320, height=240)
   ```

## Interactive Learning

### Jupyter Notebooks

Explore the notebooks in the `notebooks/` directory:

1. **YOLO Tutorial** (`notebooks/tutorials/yolo_tutorial.ipynb`)
   - Step-by-step YOLO implementation
   - Image processing examples
   - Custom visualization techniques

### Code Examples

#### Basic Object Detection
```python
from src.models.yolo_detector import YOLODetector

# Initialize detector
detector = YOLODetector('yolov8s.pt')

# Process image
detections = detector.detect_objects(image)
for detection in detections:
    print(f"Found {detection['class_name']} with confidence {detection['score']:.2f}")
```

#### Hand Tracking
```python
from src.models.hand_tracker import HandTracker

# Initialize tracker
tracker = HandTracker()

# Process image
hands = tracker.detect_hands(image)
for hand in hands:
    landmarks = hand['landmarks']
    gesture = tracker.get_hand_gesture(landmarks)
    print(f"Detected gesture: {gesture}")
```

## Project Ideas

### Beginner Projects
1. **Object Counter**: Count specific objects in a video stream
2. **Hand Gesture Controller**: Control applications with hand gestures
3. **Simple Security System**: Detect people in restricted areas

### Intermediate Projects
1. **Smart Mirror**: Display information based on detected objects
2. **Gesture-Based Game**: Create games controlled by hand gestures
3. **Object Tracking**: Track objects across video frames

### Advanced Projects
1. **Multi-Object Tracking**: Track multiple objects simultaneously
2. **Gesture Recognition System**: Advanced hand pose classification
3. **Real-time Analytics**: Analyze video streams for insights

## Best Practices

### Code Organization
- Keep models in `src/models/`
- Put utilities in `src/utils/`
- Use demo scripts in `src/scripts/`
- Store data in `data/` directory

### Performance Tips
- Use appropriate model size for your use case
- Optimize camera settings
- Process frames at lower resolution if needed
- Use GPU acceleration when available

### Debugging
- Add print statements for debugging
- Use visualization tools to inspect results
- Test with simple images first
- Check camera permissions and settings

## Resources

### Documentation
- [OpenCV Documentation](https://docs.opencv.org/)
- [Ultralytics YOLO](https://docs.ultralytics.com/)
- [MediaPipe Documentation](https://mediapipe.dev/)

### Learning Resources
- [Computer Vision Course](https://opencv.org/courses/)
- [YOLO Paper](https://arxiv.org/abs/1506.02640)
- [MediaPipe Papers](https://mediapipe.dev/docs/concepts/)

### Community
- [OpenCV Forum](https://forum.opencv.org/)
- [Ultralytics Community](https://github.com/ultralytics/ultralytics/discussions)
- [MediaPipe Community](https://github.com/google/mediapipe)

## Next Steps

After completing these tutorials:

1. **Experiment**: Try different models and parameters
2. **Extend**: Add new features to existing demos
3. **Contribute**: Share your improvements with the community
4. **Learn More**: Explore advanced computer vision topics

Remember: The best way to learn is by doing! Start with simple projects and gradually increase complexity. 