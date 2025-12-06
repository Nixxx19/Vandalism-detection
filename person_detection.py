#!/usr/bin/env python3
"""
Person Detection Module
Handles person detection using YOLO and MediaPipe pose/face estimation.
"""

import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from typing import Tuple, Optional


class PersonDetector:
    """Class to handle person detection and MediaPipe processing."""
    
    def __init__(self, human_model_path='yolo11n.pt', enable_pose=True, enable_face=False,
                 keypoint_color=(255, 0, 255), connection_color=(0, 255, 255), enable_face_for_filtering=True):
        """
        Initialize person detector.
        
        Args:
            human_model_path: Path to YOLO model for human detection
            enable_pose: Enable MediaPipe pose estimation
            enable_face: Enable MediaPipe face detection for visualization
            keypoint_color: RGB color for pose keypoints
            connection_color: RGB color for pose connections
            enable_face_for_filtering: Always enable face detection for filtering (default: True)
        """
        # Load human detection model
        print(f"Loading human detection model from {human_model_path}...")
        try:
            self.human_model = YOLO(human_model_path)
            print("Human detection model loaded successfully!")
        except Exception as e:
            print(f"Error loading human detection model: {e}")
            raise
        
        # Initialize MediaPipe
        self.pose = None
        self.face_detection = None
        self.mp_drawing = None
        self.mp_pose = None
        self.keypoint_color = keypoint_color
        self.connection_color = connection_color
        self.draw_face_boxes = enable_face  # Whether to draw face boxes
        
        if enable_pose:
            print("Initializing MediaPipe Pose...")
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.mp_drawing = mp.solutions.drawing_utils
            print("MediaPipe Pose initialized!")
        
        # Always enable face detection for filtering, even if not enabled for visualization
        if enable_face or enable_face_for_filtering:
            if enable_face_for_filtering and not enable_face:
                print("Initializing MediaPipe Face Detection (for filtering)...")
            else:
                print("Initializing MediaPipe Face Detection...")
            mp_face_detection = mp.solutions.face_detection
            self.face_detection = mp_face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=0.3  # Lower threshold for better detection
            )
            print("MediaPipe Face Detection initialized!")
    
    def detect_persons(self, image_rgb, conf=0.25, iou=0.7, imgsz=640, device=None):
        """
        Detect persons in an image.
        
        Args:
            image_rgb: Image in RGB format
            conf: Confidence threshold
            iou: IoU threshold
            imgsz: Image size
            device: Device to run on
            
        Returns:
            YOLO results object
        """
        results = self.human_model.predict(
            source=image_rgb,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            verbose=False,
            classes=[0]  # Class 0 is 'person' in COCO dataset
        )
        return results
    
    def draw_pose_landmarks(self, image, pose_landmarks, x1, y1, x2, y2):
        """Draw pose landmarks adjusted from ROI coordinates to full image coordinates."""
        if pose_landmarks is None or self.mp_pose is None:
            return
        
        h, w = image.shape[:2]
        roi_w = x2 - x1
        roi_h = y2 - y1
        
        # Create a copy of landmarks with adjusted coordinates
        adjusted_landmarks = []
        for landmark in pose_landmarks.landmark:
            # MediaPipe landmarks are normalized (0-1) relative to ROI
            # Convert to absolute coordinates in full image
            x = int(landmark.x * roi_w + x1)
            y = int(landmark.y * roi_h + y1)
            adjusted_landmarks.append((x, y))
        
        # Color scheme for different body parts - Complete mapping
        # Using integer indices from PoseLandmark enum
        # Colors are in RGB format
        connection_colors = {
            # Head/Face connections - Green
            (self.mp_pose.PoseLandmark.NOSE.value, self.mp_pose.PoseLandmark.LEFT_EYE_INNER.value): (0, 255, 0),  # Green in RGB
            (self.mp_pose.PoseLandmark.NOSE.value, self.mp_pose.PoseLandmark.LEFT_EYE.value): (0, 255, 0),
            (self.mp_pose.PoseLandmark.NOSE.value, self.mp_pose.PoseLandmark.LEFT_EYE_OUTER.value): (0, 255, 0),
            (self.mp_pose.PoseLandmark.NOSE.value, self.mp_pose.PoseLandmark.RIGHT_EYE_INNER.value): (0, 255, 0),
            (self.mp_pose.PoseLandmark.NOSE.value, self.mp_pose.PoseLandmark.RIGHT_EYE.value): (0, 255, 0),
            (self.mp_pose.PoseLandmark.NOSE.value, self.mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value): (0, 255, 0),
            (self.mp_pose.PoseLandmark.NOSE.value, self.mp_pose.PoseLandmark.LEFT_EAR.value): (0, 255, 0),
            (self.mp_pose.PoseLandmark.NOSE.value, self.mp_pose.PoseLandmark.RIGHT_EAR.value): (0, 255, 0),
            (self.mp_pose.PoseLandmark.MOUTH_LEFT.value, self.mp_pose.PoseLandmark.MOUTH_RIGHT.value): (0, 255, 0),
            
            # Arm connections - Left arm Blue, Right arm Green
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER.value, self.mp_pose.PoseLandmark.LEFT_ELBOW.value): (0, 0, 255),   # Blue in RGB
            (self.mp_pose.PoseLandmark.LEFT_ELBOW.value, self.mp_pose.PoseLandmark.LEFT_WRIST.value): (0, 0, 255),
            (self.mp_pose.PoseLandmark.LEFT_WRIST.value, self.mp_pose.PoseLandmark.LEFT_PINKY.value): (0, 0, 255),
            (self.mp_pose.PoseLandmark.LEFT_WRIST.value, self.mp_pose.PoseLandmark.LEFT_INDEX.value): (0, 0, 255),
            (self.mp_pose.PoseLandmark.LEFT_WRIST.value, self.mp_pose.PoseLandmark.LEFT_THUMB.value): (0, 0, 255),
            (self.mp_pose.PoseLandmark.LEFT_PINKY.value, self.mp_pose.PoseLandmark.LEFT_INDEX.value): (0, 0, 255),
            
            (self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value, self.mp_pose.PoseLandmark.RIGHT_ELBOW.value): (0, 0, 255),   # Blue in RGB
            (self.mp_pose.PoseLandmark.RIGHT_ELBOW.value, self.mp_pose.PoseLandmark.RIGHT_WRIST.value): (0, 0, 255),
            (self.mp_pose.PoseLandmark.RIGHT_WRIST.value, self.mp_pose.PoseLandmark.RIGHT_PINKY.value): (0, 0, 255),
            (self.mp_pose.PoseLandmark.RIGHT_WRIST.value, self.mp_pose.PoseLandmark.RIGHT_INDEX.value): (0, 0, 255),
            (self.mp_pose.PoseLandmark.RIGHT_WRIST.value, self.mp_pose.PoseLandmark.RIGHT_THUMB.value): (0, 0, 255),
            (self.mp_pose.PoseLandmark.RIGHT_PINKY.value, self.mp_pose.PoseLandmark.RIGHT_INDEX.value): (0, 0, 255),
            
            # Torso connections - Pink
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER.value, self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value): (255, 0, 255),  # Pink/Magenta in RGB
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER.value, self.mp_pose.PoseLandmark.LEFT_HIP.value): (255, 0, 255),
            (self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value, self.mp_pose.PoseLandmark.RIGHT_HIP.value): (255, 0, 255),
            (self.mp_pose.PoseLandmark.LEFT_HIP.value, self.mp_pose.PoseLandmark.RIGHT_HIP.value): (255, 0, 255),
            
            # Leg connections - Orange
            (self.mp_pose.PoseLandmark.LEFT_HIP.value, self.mp_pose.PoseLandmark.LEFT_KNEE.value): (255, 165, 0),  # Orange in RGB
            (self.mp_pose.PoseLandmark.LEFT_KNEE.value, self.mp_pose.PoseLandmark.LEFT_ANKLE.value): (255, 165, 0),
            (self.mp_pose.PoseLandmark.LEFT_ANKLE.value, self.mp_pose.PoseLandmark.LEFT_HEEL.value): (255, 165, 0),
            (self.mp_pose.PoseLandmark.LEFT_ANKLE.value, self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value): (255, 165, 0),
            (self.mp_pose.PoseLandmark.LEFT_HEEL.value, self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value): (255, 165, 0),
            
            (self.mp_pose.PoseLandmark.RIGHT_HIP.value, self.mp_pose.PoseLandmark.RIGHT_KNEE.value): (255, 165, 0),
            (self.mp_pose.PoseLandmark.RIGHT_KNEE.value, self.mp_pose.PoseLandmark.RIGHT_ANKLE.value): (255, 165, 0),
            (self.mp_pose.PoseLandmark.RIGHT_ANKLE.value, self.mp_pose.PoseLandmark.RIGHT_HEEL.value): (255, 165, 0),
            (self.mp_pose.PoseLandmark.RIGHT_ANKLE.value, self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value): (255, 165, 0),
            (self.mp_pose.PoseLandmark.RIGHT_HEEL.value, self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value): (255, 165, 0),
        }
        
        # Draw connections (skeleton lines) with specific colors
        for connection in self.mp_pose.POSE_CONNECTIONS:
            start_idx, end_idx = connection
            if (0 <= adjusted_landmarks[start_idx][0] < w and 
                0 <= adjusted_landmarks[start_idx][1] < h and
                0 <= adjusted_landmarks[end_idx][0] < w and 
                0 <= adjusted_landmarks[end_idx][1] < h):
                # Get color for this specific connection, or use default
                # Try both directions since connections can be in either order
                connection_tuple = (start_idx, end_idx)
                reverse_tuple = (end_idx, start_idx)
                color = connection_colors.get(connection_tuple) or connection_colors.get(reverse_tuple) or self.connection_color
                
                cv2.line(image, 
                        adjusted_landmarks[start_idx], 
                        adjusted_landmarks[end_idx],
                        color, 3)  # Thicker lines for better visibility
        
        # Color mapping for keypoints based on body parts
        keypoint_colors = {
            # Head/Face - Yellow (unique color)
            self.mp_pose.PoseLandmark.NOSE.value: (255, 255, 0),  # Yellow in RGB
            self.mp_pose.PoseLandmark.LEFT_EYE_INNER.value: (255, 255, 0),
            self.mp_pose.PoseLandmark.LEFT_EYE.value: (255, 255, 0),
            self.mp_pose.PoseLandmark.LEFT_EYE_OUTER.value: (255, 255, 0),
            self.mp_pose.PoseLandmark.RIGHT_EYE_INNER.value: (255, 255, 0),
            self.mp_pose.PoseLandmark.RIGHT_EYE.value: (255, 255, 0),
            self.mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value: (255, 255, 0),
            self.mp_pose.PoseLandmark.LEFT_EAR.value: (255, 255, 0),
            self.mp_pose.PoseLandmark.RIGHT_EAR.value: (255, 255, 0),
            self.mp_pose.PoseLandmark.MOUTH_LEFT.value: (255, 255, 0),
            self.mp_pose.PoseLandmark.MOUTH_RIGHT.value: (255, 255, 0),
            # Arms - Blue
            self.mp_pose.PoseLandmark.LEFT_SHOULDER.value: (0, 0, 255),
            self.mp_pose.PoseLandmark.LEFT_ELBOW.value: (0, 0, 255),
            self.mp_pose.PoseLandmark.LEFT_WRIST.value: (0, 0, 255),
            self.mp_pose.PoseLandmark.LEFT_PINKY.value: (0, 0, 255),
            self.mp_pose.PoseLandmark.LEFT_INDEX.value: (0, 0, 255),
            self.mp_pose.PoseLandmark.LEFT_THUMB.value: (0, 0, 255),
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value: (0, 0, 255),
            self.mp_pose.PoseLandmark.RIGHT_ELBOW.value: (0, 0, 255),
            self.mp_pose.PoseLandmark.RIGHT_WRIST.value: (0, 0, 255),
            self.mp_pose.PoseLandmark.RIGHT_PINKY.value: (0, 0, 255),
            self.mp_pose.PoseLandmark.RIGHT_INDEX.value: (0, 0, 255),
            self.mp_pose.PoseLandmark.RIGHT_THUMB.value: (0, 0, 255),
            # Torso - Pink
            self.mp_pose.PoseLandmark.LEFT_HIP.value: (255, 0, 255),
            self.mp_pose.PoseLandmark.RIGHT_HIP.value: (255, 0, 255),
            # Legs - Orange
            self.mp_pose.PoseLandmark.LEFT_KNEE.value: (255, 165, 0),
            self.mp_pose.PoseLandmark.LEFT_ANKLE.value: (255, 165, 0),
            self.mp_pose.PoseLandmark.LEFT_HEEL.value: (255, 165, 0),
            self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value: (255, 165, 0),
            self.mp_pose.PoseLandmark.RIGHT_KNEE.value: (255, 165, 0),
            self.mp_pose.PoseLandmark.RIGHT_ANKLE.value: (255, 165, 0),
            self.mp_pose.PoseLandmark.RIGHT_HEEL.value: (255, 165, 0),
            self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value: (255, 165, 0),
        }
        
        # Draw landmarks (keypoints) with corresponding body part colors
        for idx, (x, y) in enumerate(adjusted_landmarks):
            if 0 <= x < w and 0 <= y < h:
                # Get color for this keypoint, or use default
                keypoint_color = keypoint_colors.get(idx, self.keypoint_color)
                cv2.circle(image, (x, y), 4, keypoint_color, -1)  # Larger, filled circles
                cv2.circle(image, (x, y), 5, (255, 255, 255), 1)  # White outline for contrast
    
    def get_face_boxes(self, image_rgb, human_results):
        """
        Extract face bounding boxes from detected persons.
        
        Args:
            image_rgb: Image in RGB format
            human_results: YOLO results from person detection
            
        Returns:
            List of face bounding boxes as (x1, y1, x2, y2) tuples
        """
        face_boxes = []
        h, w = image_rgb.shape[:2]
        
        if self.face_detection is None:
            return face_boxes
        
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
                    person_roi = image_rgb[y1:y2, x1:x2]
                    
                    if person_roi.size > 0 and person_roi.shape[0] > 0 and person_roi.shape[1] > 0:
                        # Run MediaPipe face detection on person region
                        face_results = self.face_detection.process(person_roi)
                        if face_results.detections:
                            for detection in face_results.detections:
                                bbox = detection.location_data.relative_bounding_box
                                
                                # Get person ROI dimensions
                                roi_w = x2 - x1
                                roi_h = y2 - y1
                                
                                # Convert from relative (0-1) to absolute coordinates in full image
                                face_x = int(bbox.xmin * roi_w + x1)
                                face_y = int(bbox.ymin * roi_h + y1)
                                face_w = int(bbox.width * roi_w)
                                face_h = int(bbox.height * roi_h)
                                face_x2 = face_x + face_w
                                face_y2 = face_y + face_h
                                
                                # Ensure face coordinates are within image bounds
                                face_x = max(0, min(face_x, w-1))
                                face_y = max(0, min(face_y, h-1))
                                face_x2 = max(face_x+1, min(face_x2, w))
                                face_y2 = max(face_y+1, min(face_y2, h))
                                
                                # Only add if box has valid dimensions
                                if face_x2 > face_x and face_y2 > face_y:
                                    face_boxes.append((float(face_x), float(face_y), float(face_x2), float(face_y2)))
        
        return face_boxes
    
    def process_person_detections(self, image_rgb, human_results):
        """
        Process person detections and draw annotations on image.
        
        Args:
            image_rgb: Image in RGB format (will be modified in place)
            human_results: YOLO results from person detection
            
        Returns:
            Modified image with person detections and MediaPipe annotations
        """
        for result in human_results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Ensure coordinates are within image bounds
                    h, w = image_rgb.shape[:2]
                    x1 = max(0, min(x1, w))
                    y1 = max(0, min(y1, h))
                    x2 = max(0, min(x2, w))
                    y2 = max(0, min(y2, h))
                    
                    # Extract person region
                    person_roi = image_rgb[y1:y2, x1:x2]
                    
                    if person_roi.size > 0 and person_roi.shape[0] > 0 and person_roi.shape[1] > 0:
                        # STEP 1: Run MediaPipe pose estimation FIRST (COMPULSORY when persons are detected)
                        # MediaPipe pose should always be initialized, but check to be safe
                        if self.pose is not None:
                            pose_results = self.pose.process(person_roi)
                            
                            # Draw pose landmarks on the full image
                            if pose_results.pose_landmarks:
                                self.draw_pose_landmarks(
                                    image_rgb, pose_results.pose_landmarks,
                                    x1, y1, x2, y2
                                )
                        
                        # STEP 2: Run MediaPipe face detection on person region (after pose)
                        if self.face_detection is not None:
                            # Convert person_roi to RGB if needed (MediaPipe expects RGB)
                            if len(person_roi.shape) == 3 and person_roi.shape[2] == 3:
                                face_results = self.face_detection.process(person_roi)
                            else:
                                face_results = None
                            if face_results and face_results.detections:
                                # Always draw face boxes with "ticket" label
                                for detection in face_results.detections:
                                    bbox = detection.location_data.relative_bounding_box
                                    
                                    # Get person ROI dimensions
                                    roi_w = x2 - x1
                                    roi_h = y2 - y1
                                    
                                    # Convert from relative (0-1) to absolute coordinates in full image
                                    # MediaPipe returns coordinates relative to the person ROI
                                    face_x = int(bbox.xmin * roi_w + x1)
                                    face_y = int(bbox.ymin * roi_h + y1)
                                    face_w = int(bbox.width * roi_w)
                                    face_h = int(bbox.height * roi_h)
                                    face_x2 = face_x + face_w
                                    face_y2 = face_y + face_h
                                    
                                    # Ensure face coordinates are within image bounds
                                    h, w = image_rgb.shape[:2]
                                    face_x = max(0, min(face_x, w-1))
                                    face_y = max(0, min(face_y, h-1))
                                    face_x2 = max(face_x+1, min(face_x2, w))
                                    face_y2 = max(face_y+1, min(face_y2, h))
                                    
                                    # Only draw if box has valid dimensions
                                    if face_x2 > face_x and face_y2 > face_y:
                                        # Draw face bounding box (yellow color)
                                        cv2.rectangle(image_rgb, (face_x, face_y), (face_x2, face_y2), (255, 255, 0), 2)
                                        
                                        # Draw background for "ticket" label (yellow background)
                                        (text_width, text_height), baseline = cv2.getTextSize('ticket', cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                                        label_y = max(text_height + 10, face_y)  # Ensure label doesn't go above image
                                        cv2.rectangle(image_rgb, (face_x, label_y-text_height-10), (face_x+text_width+5, label_y), (255, 255, 0), -1)
                                        cv2.putText(image_rgb, 'ticket', (face_x+2, label_y-5),
                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)  # Black text
                        
                        # STEP 3: Draw human bounding box LAST (on top of everything)
                        cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        # Draw background for text (red background)
                        (text_width, text_height), baseline = cv2.getTextSize('Person', cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                        cv2.rectangle(image_rgb, (x1, y1-text_height-10), (x1+text_width+5, y1), (255, 0, 0), -1)
                        cv2.putText(image_rgb, 'Person', (x1+2, y1-5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)  # White text
        
        return image_rgb

