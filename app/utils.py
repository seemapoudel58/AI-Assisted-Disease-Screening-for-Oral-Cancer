"""
Utility functions for Oral Cancer Detection Application
"""
import os
import cv2
import numpy as np
from typing import List, Dict

# Import BBOX_COLORS from config, with fallback if config doesn't exist
try:
    from config import BBOX_COLORS
except ImportError:
    # Fallback colors if config is not available
    BBOX_COLORS = [
        (34, 139, 34),   # Forest Green for Healthy
        (255, 165, 0),   # Orange for Benign
        (255, 69, 0),    # Red-Orange for OPMD
        (220, 20, 60),   # Crimson for OCA
    ]


def draw_detections_rgb(image_rgb: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """
    Draw detection bounding boxes and labels on image.
    
    Args:
        image_rgb: Input image in RGB format
        detections: List of detection dictionaries with bbox, confidence, class info
        
    Returns:
        Annotated image with drawn detections
    """
    annotated = image_rgb.copy()
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        conf = det['confidence']
        cid = det['class_id']
        cname = det['class_name']
        color = BBOX_COLORS[cid % len(BBOX_COLORS)]
        
        # Calculate dynamic thickness based on image size
        thickness = max(2, min(image_rgb.shape[:2]) // 300)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        
        # Add label with background
        label = f"{cname}: {conf*100:.1f}%"
        font_scale = max(0.6, min(image_rgb.shape[:2]) / 1200)
        font_thickness = max(1, thickness // 2)
        
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
        
        # Draw semi-transparent background for text
        overlay = annotated.copy()
        cv2.rectangle(
            overlay, 
            (x1, max(y1 - text_height - 8, 0)), 
            (x1 + text_width + 8, y1), 
            color, 
            -1
        )
        annotated = cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0)
        
        # Draw text
        cv2.putText(
            annotated, 
            label, 
            (x1 + 4, max(y1 - 4, text_height)), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            font_scale, 
            (255, 255, 255), 
            font_thickness
        )
    
    return annotated


def get_patient_id_from_filename(filename: str) -> str:
    """
    Extract patient ID from filename.

    Args:
        filename: Image filename
        
    Returns:
        Extracted patient ID
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    parts = base.split('-')
    
    if len(parts) >= 2:
        pid = '-'.join(parts[:-1])
    else:
        pid = base
    
    return pid


def filter_best_detection(detections: List[Dict]) -> List[Dict]:
    """
    Keep only the detection with highest confidence if multiple detections exist.
    
    Args:
        detections: List of detection dictionaries
        
    Returns:
        List with single best detection or empty list
    """
    if len(detections) > 1:
        best_det = max(detections, key=lambda d: d.get('confidence', 0))
        return [best_det]
    return detections


def validate_model_path(model_path: str) -> bool:
    """
    Check if model file exists at specified path.
    
    Args:
        model_path: Path to model file
        
    Returns:
        True if exists, False otherwise
    """
    return os.path.exists(model_path)


def create_temp_directories() -> None:
    """Create necessary temporary directories if they don't exist."""
    import tempfile
    temp_dir = tempfile.gettempdir()
    app_temp = os.path.join(temp_dir, 'oral_cancer_app')
    os.makedirs(app_temp, exist_ok=True)
    return app_temp
