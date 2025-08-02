"""
YOLO Object Detector Module
Provides a clean interface for YOLO-based object detection.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict, Any


class YOLODetector:
    """
    A wrapper class for YOLO object detection with additional utilities.
    
    This class provides a clean interface for performing object detection
    using YOLO models, with support for both image and video processing.
    """
    
    def __init__(self, model_path: str = 'yolov8s.pt'):
        """
        Initialize the YOLO detector.
        
        Args:
            model_path (str): Path to the YOLO model file
        """
        self.model = YOLO(model_path)
        self.model_path = model_path
    
    def detect_objects(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in an image.
        
        Args:
            image (np.ndarray): Input image (BGR format)
            
        Returns:
            List[Dict]: List of detected objects with bounding boxes and class info
        """
        results = self.model(image)[0]
        detections = []
        
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            
            detection = {
                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                'score': float(score),
                'class_id': int(class_id),
                'class_name': results.names[int(class_id)]
            }
            detections.append(detection)
        
        return detections
    
    def process_video_stream(self, camera_index: int = 0, 
                           show_fps: bool = True) -> None:
        """
        Process live video stream with object detection.
        
        Args:
            camera_index (int): Camera device index
            show_fps (bool): Whether to display FPS on video
        """
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Press 'q' to quit the video stream")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Detect objects
            detections = self.detect_objects(frame)
            
            # Draw detections
            for detection in detections:
                x1, y1, x2, y2 = detection['bbox']
                score = detection['score']
                class_name = detection['class_name']
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f'{class_name} {score:.2f}'
                cv2.putText(frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.imshow('YOLO Object Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def process_image(self, image_path: str, output_path: str = None) -> np.ndarray:
        """
        Process a single image and optionally save the result.
        
        Args:
            image_path (str): Path to input image
            output_path (str, optional): Path to save output image
            
        Returns:
            np.ndarray: Processed image with detections drawn
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        detections = self.detect_objects(image)
        
        # Draw detections
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            score = detection['score']
            class_name = detection['class_name']
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f'{class_name} {score:.2f}'
            cv2.putText(image, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if output_path:
            cv2.imwrite(output_path, image)
        
        return image 