#!/usr/bin/env python3
"""
Object Detection Demo Script
Demonstrates real-time object detection using YOLO.
"""

import sys
import os
import argparse
import cv2

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.yolo_detector import YOLODetector
from src.utils.visualization import draw_fps
from src.utils.camera_utils import CameraCapture


def main():
    """Main function for object detection demo."""
    parser = argparse.ArgumentParser(description="YOLO Object Detection Demo")
    parser.add_argument("--camera", type=int, default=0, 
                       help="Camera device index (default: 0)")
    parser.add_argument("--model", type=str, default="yolov8s.pt",
                       help="YOLO model path (default: yolov8s.pt)")
    parser.add_argument("--image", type=str, default=None,
                       help="Process single image instead of video stream")
    parser.add_argument("--output", type=str, default=None,
                       help="Output path for processed image")
    
    args = parser.parse_args()
    
    print("🚀 Starting YOLO Object Detection Demo")
    print(f"📷 Camera: {args.camera}")
    print(f"🤖 Model: {args.model}")
    
    try:
        # Initialize YOLO detector
        detector = YOLODetector(args.model)
        print("✅ YOLO model loaded successfully")
        
        if args.image:
            # Process single image
            print(f"🖼️  Processing image: {args.image}")
            result = detector.process_image(args.image, args.output)
            print(f"✅ Image processed successfully")
            if args.output:
                print(f"💾 Result saved to: {args.output}")
        else:
            # Process video stream
            print("🎥 Starting video stream...")
            print("Press 'q' to quit")
            
            with CameraCapture(args.camera) as camera:
                while True:
                    ret, frame = camera.read()
                    if not ret:
                        print("❌ Failed to grab frame")
                        break
                    
                    # Detect objects
                    detections = detector.detect_objects(frame)
                    
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
                    
                    # Draw FPS
                    fps = camera.get_fps()
                    draw_fps(frame, fps)
                    
                    cv2.imshow('YOLO Object Detection', frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
    
    except KeyboardInterrupt:
        print("\n⏹️  Demo stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        cv2.destroyAllWindows()
        print("👋 Demo finished")


if __name__ == "__main__":
    main() 