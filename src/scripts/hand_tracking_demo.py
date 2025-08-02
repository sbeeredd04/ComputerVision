#!/usr/bin/env python3
"""
Hand Tracking Demo Script
Demonstrates real-time hand tracking using MediaPipe.
"""

import sys
import os
import argparse
import cv2

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.hand_tracker import HandTracker
from src.utils.visualization import draw_fps
from src.utils.camera_utils import CameraCapture


def main():
    """Main function for hand tracking demo."""
    parser = argparse.ArgumentParser(description="MediaPipe Hand Tracking Demo")
    parser.add_argument("--camera", type=int, default=0, 
                       help="Camera device index (default: 0)")
    parser.add_argument("--confidence", type=float, default=0.7,
                       help="Minimum detection confidence (default: 0.7)")
    parser.add_argument("--image", type=str, default=None,
                       help="Process single image instead of video stream")
    parser.add_argument("--output", type=str, default=None,
                       help="Output path for processed image")
    parser.add_argument("--gesture", action="store_true",
                       help="Enable gesture recognition")
    
    args = parser.parse_args()
    
    print("🚀 Starting Hand Tracking Demo")
    print(f"📷 Camera: {args.camera}")
    print(f"🎯 Confidence: {args.confidence}")
    print(f"✋ Gesture Recognition: {args.gesture}")
    
    try:
        # Initialize hand tracker
        tracker = HandTracker(min_detection_confidence=args.confidence)
        print("✅ Hand tracker initialized successfully")
        
        if args.image:
            # Process single image
            print(f"🖼️  Processing image: {args.image}")
            result = tracker.process_image(args.image, args.output)
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
                    
                    # Detect hands
                    hands_data = tracker.detect_hands(frame)
                    
                    # Draw hand landmarks
                    for hand_data in hands_data:
                        hand_landmarks = hand_data['hand_landmarks']
                        tracker.mp_draw.draw_landmarks(
                            frame, 
                            hand_landmarks, 
                            tracker.mp_hands.HAND_CONNECTIONS
                        )
                        
                        # Gesture recognition
                        if args.gesture:
                            landmarks = hand_data['landmarks']
                            gesture = tracker.get_hand_gesture(landmarks)
                            
                            # Draw gesture text
                            cv2.putText(frame, f"Gesture: {gesture}", (10, 60), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    
                    # Draw FPS
                    fps = camera.get_fps()
                    draw_fps(frame, fps)
                    
                    # Draw hand count
                    hand_count = len(hands_data)
                    cv2.putText(frame, f"Hands: {hand_count}", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                    
                    cv2.imshow("Hand Tracking", frame)
                    
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