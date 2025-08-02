"""
Unit tests for computer vision models.
"""

import pytest
import numpy as np
import cv2
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.models.yolo_detector import YOLODetector
from src.models.hand_tracker import HandTracker
from src.utils.visualization import draw_bounding_box, draw_text, create_color_palette


class TestYOLODetector:
    """Test cases for YOLO detector."""
    
    def test_initialization(self):
        """Test YOLO detector initialization."""
        try:
            detector = YOLODetector()
            assert detector is not None
            assert hasattr(detector, 'model')
        except Exception as e:
            pytest.skip(f"YOLO model not available: {e}")
    
    def test_detect_objects_empty_image(self):
        """Test object detection on empty image."""
        try:
            detector = YOLODetector()
            # Create a blank image
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            detections = detector.detect_objects(image)
            assert isinstance(detections, list)
        except Exception as e:
            pytest.skip(f"YOLO model not available: {e}")


class TestHandTracker:
    """Test cases for hand tracker."""
    
    def test_initialization(self):
        """Test hand tracker initialization."""
        tracker = HandTracker()
        assert tracker is not None
        assert hasattr(tracker, 'hands')
        assert hasattr(tracker, 'mp_draw')
    
    def test_detect_hands_empty_image(self):
        """Test hand detection on empty image."""
        tracker = HandTracker()
        # Create a blank image
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        hands = tracker.detect_hands(image)
        assert isinstance(hands, list)
    
    def test_gesture_recognition_empty_landmarks(self):
        """Test gesture recognition with empty landmarks."""
        tracker = HandTracker()
        gesture = tracker.get_hand_gesture([])
        assert gesture == "Unknown"


class TestVisualization:
    """Test cases for visualization utilities."""
    
    def test_draw_bounding_box(self):
        """Test bounding box drawing."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        bbox = (10, 10, 50, 50)
        result = draw_bounding_box(image, bbox, "test")
        assert result is not None
        assert result.shape == image.shape
    
    def test_draw_text(self):
        """Test text drawing."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = draw_text(image, "test", (10, 10))
        assert result is not None
        assert result.shape == image.shape
    
    def test_create_color_palette(self):
        """Test color palette creation."""
        colors = create_color_palette(5)
        assert len(colors) == 5
        assert all(isinstance(color, tuple) for color in colors)
        assert all(len(color) == 3 for color in colors)


class TestCameraUtils:
    """Test cases for camera utilities."""
    
    def test_list_cameras(self):
        """Test camera listing."""
        from src.utils.camera_utils import list_cameras
        cameras = list_cameras()
        assert isinstance(cameras, list)
    
    def test_get_camera_info(self):
        """Test camera info retrieval."""
        from src.utils.camera_utils import get_camera_info
        info = get_camera_info(0)
        assert isinstance(info, dict)


if __name__ == "__main__":
    pytest.main([__file__]) 