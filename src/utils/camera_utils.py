"""
Camera Utilities for Computer Vision
Provides classes for video capture and recording.
"""

import cv2
import numpy as np
import time
from typing import Optional, Tuple


class CameraCapture:
    """
    A wrapper class for camera capture with additional utilities.
    """
    
    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480):
        """
        Initialize camera capture.
        
        Args:
            camera_index (int): Camera device index
            width (int): Frame width
            height (int): Frame height
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = None
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        
    def start(self) -> bool:
        """
        Start camera capture.
        
        Returns:
            bool: True if camera started successfully
        """
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_index}")
            return False
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        return True
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the camera.
        
        Returns:
            Tuple[bool, Optional[np.ndarray]]: (success, frame)
        """
        if self.cap is None:
            return False, None
        
        ret, frame = self.cap.read()
        if ret:
            self._update_fps()
        
        return ret, frame
    
    def _update_fps(self):
        """Update FPS calculation."""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def get_fps(self) -> float:
        """
        Get current FPS.
        
        Returns:
            float: Current FPS
        """
        return self.current_fps
    
    def release(self):
        """Release camera resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()


class VideoWriter:
    """
    A wrapper class for video writing with additional utilities.
    """
    
    def __init__(self, output_path: str, fps: int = 30, 
                 width: int = 640, height: int = 480):
        """
        Initialize video writer.
        
        Args:
            output_path (str): Output video file path
            fps (int): Frames per second
            width (int): Frame width
            height (int): Frame height
        """
        self.output_path = output_path
        self.fps = fps
        self.width = width
        self.height = height
        self.writer = None
        
    def start(self):
        """Start video writer."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            self.output_path, fourcc, self.fps, (self.width, self.height)
        )
        
        if not self.writer.isOpened():
            print(f"Error: Could not open video writer for {self.output_path}")
            return False
        
        return True
    
    def write(self, frame: np.ndarray):
        """
        Write a frame to the video.
        
        Args:
            frame (np.ndarray): Frame to write
        """
        if self.writer is not None:
            # Resize frame if necessary
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))
            
            self.writer.write(frame)
    
    def release(self):
        """Release video writer resources."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()


def list_cameras() -> list:
    """
    List available camera devices.
    
    Returns:
        list: List of available camera indices
    """
    available_cameras = []
    
    for i in range(10):  # Check first 10 camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    
    return available_cameras


def get_camera_info(camera_index: int = 0) -> dict:
    """
    Get camera information.
    
    Args:
        camera_index (int): Camera device index
        
    Returns:
        dict: Camera information
    """
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        return {"error": f"Could not open camera {camera_index}"}
    
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "brightness": cap.get(cv2.CAP_PROP_BRIGHTNESS),
        "contrast": cap.get(cv2.CAP_PROP_CONTRAST),
        "saturation": cap.get(cv2.CAP_PROP_SATURATION),
        "hue": cap.get(cv2.CAP_PROP_HUE),
        "gain": cap.get(cv2.CAP_PROP_GAIN),
        "exposure": cap.get(cv2.CAP_PROP_EXPOSURE)
    }
    
    cap.release()
    return info 