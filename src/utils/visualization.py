"""
Visualization Utilities for Computer Vision
Provides functions for drawing bounding boxes, text, and landmarks on images.
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict, Any


def draw_bounding_box(image: np.ndarray, 
                     bbox: Tuple[int, int, int, int],
                     label: str = "",
                     color: Tuple[int, int, int] = (0, 255, 0),
                     thickness: int = 2) -> np.ndarray:
    """
    Draw a bounding box on an image.
    
    Args:
        image (np.ndarray): Input image
        bbox (Tuple[int, int, int, int]): Bounding box (x1, y1, x2, y2)
        label (str): Label text to display
        color (Tuple[int, int, int]): BGR color tuple
        thickness (int): Line thickness
        
    Returns:
        np.ndarray: Image with bounding box drawn
    """
    x1, y1, x2, y2 = bbox
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    
    if label:
        # Calculate text position
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        text_x = x1
        text_y = y1 - 10 if y1 - 10 > text_size[1] else y1 + text_size[1]
        
        # Draw text background
        cv2.rectangle(image, (text_x, text_y - text_size[1] - 5), 
                     (text_x + text_size[0], text_y + 5), color, -1)
        
        # Draw text
        cv2.putText(image, label, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return image


def draw_text(image: np.ndarray, 
              text: str,
              position: Tuple[int, int],
              font: int = cv2.FONT_HERSHEY_SIMPLEX,
              font_scale: float = 1.0,
              color: Tuple[int, int, int] = (255, 255, 255),
              thickness: int = 2) -> np.ndarray:
    """
    Draw text on an image.
    
    Args:
        image (np.ndarray): Input image
        text (str): Text to draw
        position (Tuple[int, int]): Text position (x, y)
        font: Font type
        font_scale (float): Font scale
        color (Tuple[int, int, int]): BGR color tuple
        thickness (int): Text thickness
        
    Returns:
        np.ndarray: Image with text drawn
    """
    cv2.putText(image, text, position, font, font_scale, color, thickness)
    return image


def draw_landmarks(image: np.ndarray,
                  landmarks: List[Dict[str, float]],
                  connections: List[Tuple[int, int]] = None,
                  landmark_color: Tuple[int, int, int] = (0, 255, 0),
                  connection_color: Tuple[int, int, int] = (255, 0, 0),
                  landmark_radius: int = 3,
                  connection_thickness: int = 2) -> np.ndarray:
    """
    Draw landmarks and connections on an image.
    
    Args:
        image (np.ndarray): Input image
        landmarks (List[Dict]): List of landmark coordinates with 'x', 'y' keys
        connections (List[Tuple]): List of landmark connection pairs
        landmark_color (Tuple[int, int, int]): Color for landmarks
        connection_color (Tuple[int, int, int]): Color for connections
        landmark_radius (int): Radius of landmark circles
        connection_thickness (int): Thickness of connection lines
        
    Returns:
        np.ndarray: Image with landmarks drawn
    """
    height, width = image.shape[:2]
    
    # Draw landmarks
    for landmark in landmarks:
        x = int(landmark['x'] * width)
        y = int(landmark['y'] * height)
        cv2.circle(image, (x, y), landmark_radius, landmark_color, -1)
    
    # Draw connections
    if connections:
        for connection in connections:
            start_idx, end_idx = connection
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_x = int(landmarks[start_idx]['x'] * width)
                start_y = int(landmarks[start_idx]['y'] * height)
                end_x = int(landmarks[end_idx]['x'] * width)
                end_y = int(landmarks[end_idx]['y'] * height)
                
                cv2.line(image, (start_x, start_y), (end_x, end_y), 
                        connection_color, connection_thickness)
    
    return image


def draw_fps(image: np.ndarray, fps: float, position: Tuple[int, int] = (10, 30)) -> np.ndarray:
    """
    Draw FPS counter on an image.
    
    Args:
        image (np.ndarray): Input image
        fps (float): FPS value to display
        position (Tuple[int, int]): Position to draw FPS text
        
    Returns:
        np.ndarray: Image with FPS drawn
    """
    fps_text = f"FPS: {fps:.1f}"
    return draw_text(image, fps_text, position, color=(0, 255, 0))


def create_color_palette(n_colors: int) -> List[Tuple[int, int, int]]:
    """
    Create a color palette for visualization.
    
    Args:
        n_colors (int): Number of colors to generate
        
    Returns:
        List[Tuple[int, int, int]]: List of BGR color tuples
    """
    colors = []
    for i in range(n_colors):
        # Generate distinct colors using HSV space
        hue = (i * 137.508) % 360  # Golden angle approximation
        saturation = 255
        value = 255
        
        # Convert HSV to BGR
        hsv = np.array([[[hue, saturation, value]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        color = tuple(map(int, bgr[0, 0]))
        colors.append(color)
    
    return colors 