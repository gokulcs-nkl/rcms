"""
Object Detection Module for Classroom Monitoring
Uses YOLO v5 for real-time detection of prohibited objects like mobile phones,
laptops, and other electronic gadgets in the classroom.
"""

import cv2
import torch
import numpy as np
from datetime import datetime

class ObjectDetector:
    """
    Real-time object detector using YOLOv5
    Detects prohibited items like mobile phones, laptops, and electronics
    """
    
    def __init__(self, model_size='s'):
        """
        Initialize the YOLOv5 object detector
        
        Args:
            model_size: 's' for small, 'm' for medium, 'l' for large, 'x' for extra large
        """
        try:
            self.model = torch.hub.load('ultralytics/yolov5', f'yolov5{model_size}')
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            self.model.conf = 0.5  # Confidence threshold
            
            # Classes of interest (gadgets and prohibited items)
            self.target_classes = [
                'cell phone',
                'mobile phone',
                'smartphone',
                'laptop',
                'computer',
                'keyboard',
                'mouse',
                'tablet',
                'ipad',
                'headphones',
                'earbuds',
                'earphones',
                'watch',
                'smartwatch',
                'camera',
                'video camera',
            ]
            
            print("✓ Object Detection Model loaded successfully")
            
        except Exception as e:
            print(f"✗ Error loading object detection model: {e}")
            self.model = None
    
    def detect_objects(self, frame):
        """
        Detect objects in a frame
        
        Args:
            frame: OpenCV frame/image
            
        Returns:
            detections: List of detected objects with details
            annotated_frame: Frame with bounding boxes
        """
        if self.model is None:
            return [], frame
        
        try:
            # Run inference
            results = self.model(frame)
            
            detections = []
            
            # Extract results
            for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
                class_name = results.names[int(cls)]
                
                # Filter for gadgets/electronics only
                if self._is_target_object(class_name):
                    x1, y1, x2, y2 = map(int, xyxy)
                    width = x2 - x1
                    height = y2 - y1
                    
                    detection = {
                        'class': class_name,
                        'confidence': float(conf),
                        'bbox': (x1, y1, x2, y2),
                        'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                        'width': width,
                        'height': height,
                        'timestamp': datetime.now()
                    }
                    detections.append(detection)
            
            # Annotate frame
            annotated_frame = self._annotate_frame(frame, detections)
            
            return detections, annotated_frame
            
        except Exception as e:
            print(f"Error during detection: {e}")
            return [], frame
    
    def _is_target_object(self, class_name):
        """Check if detected class is a target object (gadget/electronics)"""
        class_lower = class_name.lower()
        for target in self.target_classes:
            if target in class_lower or class_lower in target:
                return True
        return False
    
    def _annotate_frame(self, frame, detections):
        """Draw bounding boxes and labels on the frame"""
        annotated = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            conf = detection['confidence']
            class_name = detection['class']
            
            # Draw bounding box (Red for gadgets)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Draw label
            label = f"{class_name} {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            cv2.rectangle(annotated, 
                         (x1, y1 - label_size[1] - 4),
                         (x1 + label_size[0], y1),
                         (0, 0, 255), -1)
            
            cv2.putText(annotated, label, 
                       (x1, y1 - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       (255, 255, 255), 1)
        
        return annotated
    
    def get_object_type_key(self, class_name):
        """
        Map detected class name to model object type key
        
        Returns:
            tuple: (object_type_key, is_prohibited)
        """
        class_lower = class_name.lower()
        
        if 'phone' in class_lower or 'mobile' in class_lower or 'smartphone' in class_lower:
            return 'mobile_phone', True
        elif 'laptop' in class_lower or 'computer' in class_lower or 'keyboard' in class_lower or 'mouse' in class_lower:
            return 'laptop', True
        elif 'tablet' in class_lower or 'ipad' in class_lower:
            return 'tablet', True
        elif 'watch' in class_lower or 'smartwatch' in class_lower:
            return 'smartwatch', True
        elif 'headphone' in class_lower or 'earphone' in class_lower or 'earbuds' in class_lower:
            return 'headphones', True
        elif 'camera' in class_lower or 'video' in class_lower:
            return 'camera', True
        else:
            return 'other_gadget', True
    
    def create_violation_summary(self, detections):
        """Create a summary of detected violations"""
        if not detections:
            return None
        
        summary = {
            'total_detections': len(detections),
            'detection_time': datetime.now(),
            'objects_detected': {}
        }
        
        for detection in detections:
            obj_type = detection['class']
            if obj_type not in summary['objects_detected']:
                summary['objects_detected'][obj_type] = {
                    'count': 0,
                    'confidence_scores': []
                }
            
            summary['objects_detected'][obj_type]['count'] += 1
            summary['objects_detected'][obj_type]['confidence_scores'].append(detection['confidence'])
        
        return summary

# Global object detector instance
_detector = None

def get_object_detector():
    """Get or create the global object detector instance"""
    global _detector
    if _detector is None:
        _detector = ObjectDetector(model_size='s')  # Using small model for faster inference
    return _detector

def detect_prohibited_items(frame):
    """
    Convenience function to detect prohibited items in a frame
    
    Args:
        frame: OpenCV frame
        
    Returns:
        detections, annotated_frame
    """
    detector = get_object_detector()
    return detector.detect_objects(frame)
