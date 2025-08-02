"""
MediaPipe Hand Tracker Module
Provides a clean interface for hand tracking using MediaPipe.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import List, Tuple, Dict, Any


class HandTracker:
    """
    A wrapper class for MediaPipe hand tracking with additional utilities.
    
    This class provides a clean interface for performing hand tracking
    using MediaPipe, with support for both image and video processing.
    """
    
    def __init__(self, min_detection_confidence: float = 0.7, 
                 min_tracking_confidence: float = 0.5):
        """
        Initialize the hand tracker.
        
        Args:
            min_detection_confidence (float): Minimum confidence for hand detection
            min_tracking_confidence (float): Minimum confidence for hand tracking
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        
    def detect_hands(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect hands in an image.
        
        Args:
            image (np.ndarray): Input image (BGR format)
            
        Returns:
            List[Dict]: List of detected hands with landmarks
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        hands_data = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract landmark coordinates
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z
                    })
                
                hand_data = {
                    'landmarks': landmarks,
                    'hand_landmarks': hand_landmarks
                }
                hands_data.append(hand_data)
        
        return hands_data
    
    def process_video_stream(self, camera_index: int = 0) -> None:
        """
        Process live video stream with hand tracking.
        
        Args:
            camera_index (int): Camera device index
        """
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Press 'q' to quit the video stream")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect hands
            hands_data = self.detect_hands(frame)
            
            # Draw hand landmarks
            for hand_data in hands_data:
                hand_landmarks = hand_data['hand_landmarks']
                self.mp_draw.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS
                )
            
            cv2.imshow("Hand Tracking", frame)
            
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
            np.ndarray: Processed image with hand landmarks drawn
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        hands_data = self.detect_hands(image)
        
        # Draw hand landmarks
        for hand_data in hands_data:
            hand_landmarks = hand_data['hand_landmarks']
            self.mp_draw.draw_landmarks(
                image, 
                hand_landmarks, 
                self.mp_hands.HAND_CONNECTIONS
            )
        
        if output_path:
            cv2.imwrite(output_path, image)
        
        return image
    
    def get_hand_gesture(self, landmarks: List[Dict[str, float]]) -> str:
        """
        Analyze hand landmarks to determine gesture.
        
        Args:
            landmarks (List[Dict]): List of hand landmark coordinates
            
        Returns:
            str: Detected gesture (basic implementation)
        """
        if len(landmarks) < 21:  # MediaPipe hands have 21 landmarks
            return "Unknown"
        
        # Basic gesture detection based on finger positions
        # This is a simplified implementation
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        
        # Check if fingers are extended (simplified logic)
        fingers_extended = []
        
        # Thumb (simplified)
        fingers_extended.append(thumb_tip['x'] > landmarks[3]['x'])
        
        # Other fingers (simplified)
        fingers_extended.append(index_tip['y'] < landmarks[6]['y'])
        fingers_extended.append(middle_tip['y'] < landmarks[10]['y'])
        fingers_extended.append(ring_tip['y'] < landmarks[14]['y'])
        fingers_extended.append(pinky_tip['y'] < landmarks[18]['y'])
        
        # Basic gesture classification
        extended_count = sum(fingers_extended)
        
        if extended_count == 0:
            return "Fist"
        elif extended_count == 1:
            return "Point"
        elif extended_count == 2:
            return "Peace"
        elif extended_count == 3:
            return "Three"
        elif extended_count == 4:
            return "Four"
        elif extended_count == 5:
            return "Open Hand"
        else:
            return "Unknown" 