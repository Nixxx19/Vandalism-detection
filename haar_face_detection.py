#!/usr/bin/env python3
"""
Haar Cascade Face Detection Module
Uses OpenCV's Haar Cascade classifier for face detection.
"""

import cv2
import numpy as np
from pathlib import Path


class HaarFaceDetector:
    """Class to handle face detection using Haar Cascade."""
    
    def __init__(self, cascade_path=None):
        """
        Initialize Haar Cascade face detector.
        
        Args:
            cascade_path: Path to Haar Cascade XML file. If None, uses default OpenCV frontal face cascade.
        """
        if cascade_path is None:
            # Use OpenCV's built-in frontal face cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        cascade_path = Path(cascade_path)
        if not cascade_path.exists():
            raise FileNotFoundError(f"Haar Cascade file not found: {cascade_path}")
        
        print(f"Loading Haar Cascade from {cascade_path}...")
        self.face_cascade = cv2.CascadeClassifier(str(cascade_path))
        
        if self.face_cascade.empty():
            raise ValueError(f"Failed to load Haar Cascade from {cascade_path}")
        
        print("Haar Cascade face detector loaded successfully!")
    
    def detect_faces(self, image, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
        """
        Detect faces in an image using Haar Cascade.
        
        Args:
            image: Image in BGR or RGB format (OpenCV will handle it)
            scale_factor: Parameter specifying how much the image size is reduced at each image scale
            min_neighbors: Parameter specifying how many neighbors each candidate rectangle should have
            min_size: Minimum possible object size. Objects smaller than this are ignored.
            
        Returns:
            List of face bounding boxes as (x, y, w, h) tuples
        """
        # Convert to grayscale for Haar Cascade
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY if image.shape[2] == 3 else cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size
        )
        
        # Convert from (x, y, w, h) to (x1, y1, x2, y2) format
        face_boxes = []
        for (x, y, w, h) in faces:
            face_boxes.append((int(x), int(y), int(x + w), int(y + h)))
        
        # Filter out overlapping faces - keep only the largest one
        if len(face_boxes) > 1:
            # Calculate areas and sort by size (largest first)
            face_areas = [(box, (box[2] - box[0]) * (box[3] - box[1])) for box in face_boxes]
            face_areas.sort(key=lambda x: x[1], reverse=True)
            
            # Keep the largest face and filter out overlapping ones
            filtered_boxes = [face_areas[0][0]]
            largest_box = face_areas[0][0]
            
            for box, area in face_areas[1:]:
                # Calculate IoU with largest box
                x1, y1, x2, y2 = box
                lx1, ly1, lx2, ly2 = largest_box
                
                # Calculate intersection
                inter_x1 = max(x1, lx1)
                inter_y1 = max(y1, ly1)
                inter_x2 = min(x2, lx2)
                inter_y2 = min(y2, ly2)
                
                if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    box_area = (x2 - x1) * (y2 - y1)
                    iou = inter_area / box_area if box_area > 0 else 0
                    
                    # Only keep if IoU is less than 0.3 (not overlapping much)
                    if iou < 0.3:
                        filtered_boxes.append(box)
                else:
                    # No overlap, keep it
                    filtered_boxes.append(box)
            
            return filtered_boxes
        
        return face_boxes
    
    def get_face_boxes_from_persons(self, image_rgb, human_results):
        """
        Extract face bounding boxes from detected persons using Haar Cascade.
        
        Args:
            image_rgb: Image in RGB format
            human_results: YOLO results from person detection
            
        Returns:
            List of face bounding boxes as (x1, y1, x2, y2) tuples
        """
        face_boxes = []
        h, w = image_rgb.shape[:2]
        
        # Convert RGB to BGR for OpenCV (Haar Cascade works with BGR)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        for result in human_results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Ensure coordinates are within image bounds
                    x1 = max(0, min(x1, w))
                    y1 = max(0, min(y1, h))
                    x2 = max(0, min(x2, w))
                    y2 = max(0, min(y2, h))
                    
                    # Extract person region
                    person_roi = image_bgr[y1:y2, x1:x2]
                    
                    if person_roi.size > 0 and person_roi.shape[0] > 0 and person_roi.shape[1] > 0:
                        # Detect faces in person region with stricter parameters to avoid multiple detections
                        roi_h, roi_w = person_roi.shape[:2]
                        min_face_size = max(30, int(min(roi_w, roi_h) * 0.15))  # At least 15% of ROI size
                        roi_faces = self.detect_faces(person_roi, scale_factor=1.2, min_neighbors=6, min_size=(min_face_size, min_face_size))
                        
                        # Convert face coordinates from ROI to full image
                        for (fx, fy, fx2, fy2) in roi_faces:
                            # Adjust coordinates to full image
                            face_x = fx + x1
                            face_y = fy + y1
                            face_x2 = fx2 + x1
                            face_y2 = fy2 + y1
                            
                            # Ensure face coordinates are within image bounds
                            face_x = max(0, min(face_x, w))
                            face_y = max(0, min(face_y, h))
                            face_x2 = max(face_x+1, min(face_x2, w))
                            face_y2 = max(face_y+1, min(face_y2, h))
                            
                            if face_x2 > face_x and face_y2 > face_y:
                                face_boxes.append((float(face_x), float(face_y), float(face_x2), float(face_y2)))
        
        return face_boxes
    
    def draw_face_boxes(self, image_rgb, human_results):
        """
        Detect and draw face boxes with "ticket" label on image.
        
        Args:
            image_rgb: Image in RGB format (will be modified in place)
            human_results: YOLO results from person detection
            
        Returns:
            Modified image with face boxes and "ticket" labels
        """
        h, w = image_rgb.shape[:2]
        
        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        for result in human_results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Ensure coordinates are within image bounds
                    x1 = max(0, min(x1, w))
                    y1 = max(0, min(y1, h))
                    x2 = max(0, min(x2, w))
                    y2 = max(0, min(y2, h))
                    
                    # Extract person region
                    person_roi = image_bgr[y1:y2, x1:x2]
                    
                    if person_roi.size > 0 and person_roi.shape[0] > 0 and person_roi.shape[1] > 0:
                        # Detect faces in person region with stricter parameters to avoid multiple detections
                        roi_h, roi_w = person_roi.shape[:2]
                        min_face_size = max(30, int(min(roi_w, roi_h) * 0.15))  # At least 15% of ROI size
                        roi_faces = self.detect_faces(person_roi, scale_factor=1.2, min_neighbors=6, min_size=(min_face_size, min_face_size))
                        
                        # Draw face boxes
                        for (fx, fy, fx2, fy2) in roi_faces:
                            # Adjust coordinates to full image
                            face_x = fx + x1
                            face_y = fy + y1
                            face_x2 = fx2 + x1
                            face_y2 = fy2 + y1
                            
                            # Ensure face coordinates are within image bounds
                            face_x = max(0, min(face_x, w-1))
                            face_y = max(0, min(face_y, h-1))
                            face_x2 = max(face_x+1, min(face_x2, w))
                            face_y2 = max(face_y+1, min(face_y2, h))
                            
                            if face_x2 > face_x and face_y2 > face_y:
                                # Draw face bounding box (yellow color)
                                cv2.rectangle(image_bgr, (face_x, face_y), (face_x2, face_y2), (0, 255, 255), 2)  # Yellow in BGR
                                
                                # Draw background for "ticket" label (yellow background)
                                (text_width, text_height), baseline = cv2.getTextSize('ticket', cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                                label_y = max(text_height + 10, face_y)  # Ensure label doesn't go above image
                                cv2.rectangle(image_bgr, (face_x, label_y-text_height-10), (face_x+text_width+5, label_y), (0, 255, 255), -1)  # Yellow in BGR
                                cv2.putText(image_bgr, 'ticket', (face_x+2, label_y-5),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)  # Black text
        
        # Convert back to RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return image_rgb

