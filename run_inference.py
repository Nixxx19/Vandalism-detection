#!/usr/bin/env python3
"""
YOLOv11 Inference Script with Human Detection and MediaPipe Pose Estimation
Run inference on images, videos, or webcam using your trained model.
Also detects humans and applies MediaPipe pose estimation.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import sys
import cv2
import numpy as np
from person_detection import PersonDetector
from haar_face_detection import HaarFaceDetector


def main():
    parser = argparse.ArgumentParser(description='Run YOLOv11 inference')
    parser.add_argument(
        '--model',
        type=str,
        default='train/weights/best.pt',
        help='Path to model weights (default: train/weights/best.pt)'
    )
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Source: path to image/video, directory, or "0" for webcam'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.11,
        help='Confidence threshold for graffiti detection (default: 0.11)'
    )
    parser.add_argument(
        '--iou',
        type=float,
        default=0.7,
        help='IoU threshold for NMS (default: 0.7)'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Image size (default: 640)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to run on (cuda, cpu, or None for auto)'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save results'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Show results in window'
    )
    parser.add_argument(
        '--save-txt',
        action='store_true',
        help='Save results as .txt files'
    )
    parser.add_argument(
        '--save-conf',
        action='store_true',
        help='Save confidences in labels'
    )
    parser.add_argument(
        '--project',
        type=str,
        default='runs/detect',
        help='Project directory for saving results'
    )
    parser.add_argument(
        '--name',
        type=str,
        default='inference',
        help='Name for the run'
    )
    parser.add_argument(
        '--human-model',
        type=str,
        default='yolo11n.pt',
        help='YOLO model for human detection (default: yolo11n.pt)'
    )
    # MediaPipe pose estimation is now always enabled when persons are detected
    # (removed --disable-mediapipe flag to make it compulsory)
    parser.add_argument(
        '--human-conf',
        type=float,
        default=0.25,
        help='Confidence threshold for human detection (default: 0.25)'
    )
    parser.add_argument(
        '--enable-face-detection',
        action='store_true',
        default=False,
        help='Enable MediaPipe face detection on detected humans'
    )
    parser.add_argument(
        '--pose-keypoint-color',
        type=str,
        default='255,0,255',
        help='MediaPipe keypoint color in RGB format (default: 255,0,255 for magenta)'
    )
    parser.add_argument(
        '--pose-connection-color',
        type=str,
        default='0,255,255',
        help='MediaPipe connection color in RGB format (default: 0,255,255 for cyan)'
    )
    parser.add_argument(
        '--use-haar-face',
        action='store_true',
        default=False,
        help='Use Haar Cascade for face detection instead of MediaPipe'
    )
    parser.add_argument(
        '--haar-cascade-path',
        type=str,
        default=None,
        help='Path to Haar Cascade XML file (default: uses OpenCV default frontal face cascade)'
    )

    args = parser.parse_args()
    
    # Parse color arguments
    try:
        keypoint_color = tuple(map(int, args.pose_keypoint_color.split(',')))
        connection_color = tuple(map(int, args.pose_connection_color.split(',')))
    except:
        print("Warning: Invalid color format. Using defaults.")
        keypoint_color = (255, 0, 255)
        connection_color = (0, 255, 255)

    # Check if model file exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found at {args.model}")
        print("Available model files:")
        if Path('train/weights/best.pt').exists():
            print("  - train/weights/best.pt (recommended)")
        if Path('train/weights/last.pt').exists():
            print("  - train/weights/last.pt")
        if Path('my_model.pt').exists():
            print("  - my_model.pt")
        sys.exit(1)

    # Load graffiti detection model
    print(f"Loading graffiti model from {args.model}...")
    try:
        graffiti_model = YOLO(args.model)
        print("Graffiti model loaded successfully!")
    except Exception as e:
        print(f"Error loading graffiti model: {e}")
        sys.exit(1)

    # Initialize person detector
    # MediaPipe pose estimation is ALWAYS enabled when persons are detected (compulsory)
    person_detector = PersonDetector(
        human_model_path=args.human_model,
        enable_pose=True,  # Always enabled - compulsory when persons are detected
        enable_face=args.enable_face_detection and not args.use_haar_face,  # Disable MediaPipe face if using Haar
        keypoint_color=keypoint_color,
        connection_color=connection_color
    )
    
    # Initialize Haar face detector if requested
    haar_face_detector = None
    if args.use_haar_face:
        try:
            haar_face_detector = HaarFaceDetector(cascade_path=args.haar_cascade_path)
        except Exception as e:
            print(f"Error initializing Haar Cascade face detector: {e}")
            print("Falling back to MediaPipe face detection...")
            args.use_haar_face = False

    # Determine if source is image or video/webcam
    source = args.source
    is_video = source.isdigit() or Path(source).suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    
    if is_video or source.isdigit():
        # Process video or webcam
        process_video(
            graffiti_model, person_detector,
            source, args.conf, args.human_conf, args.iou, args.imgsz,
            args.device, args.save, args.show, args.project, args.name,
            haar_face_detector
        )
    else:
        # Process image
        process_image(
            graffiti_model, person_detector,
            source, args.conf, args.human_conf, args.iou, args.imgsz,
            args.device, args.save, args.show, args.project, args.name,
            haar_face_detector
        )

    print("\nInference complete!")
    if args.save:
        print(f"Results saved to {args.project}/{args.name}/")


def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Calculate union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def is_box_contained(box1, box2):
    """Check if box1 is contained within or significantly overlaps box2."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Check if center of box1 is inside box2
    center_x = (x1_min + x1_max) / 2
    center_y = (y1_min + y1_max) / 2
    center_inside = (x2_min <= center_x <= x2_max) and (y2_min <= center_y <= y2_max)
    
    # Check if box1 is mostly contained in box2 (at least 50% overlap)
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return False
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    
    # If center is inside or more than 30% of box1 overlaps with box2, consider it contained
    overlap_ratio = inter_area / box1_area if box1_area > 0 else 0
    return center_inside or overlap_ratio > 0.3


def filter_graffiti_overlapping_persons(graffiti_results, human_results, person_detector, image_rgb, haar_face_detector=None, iou_threshold=0.1):
    """Filter out graffiti detections that overlap with face boxes only (not person boxes)."""
    # Get face bounding boxes (only filter faces, not entire person bodies)
    face_boxes = []
    if haar_face_detector is not None:
        # Use Haar Cascade for face detection
        face_boxes = haar_face_detector.get_face_boxes_from_persons(image_rgb, human_results)
    elif person_detector is not None:
        # Use MediaPipe for face detection
        face_boxes = person_detector.get_face_boxes(image_rgb, human_results)
    
    # Only use face boxes for exclusion (not person boxes)
    exclusion_boxes = face_boxes
    
    if not exclusion_boxes:
        return graffiti_results
    
    # Filter graffiti detections
    filtered_results = []
    for result in graffiti_results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            filtered_results.append(result)
            continue
        
        # Create mask for boxes to keep
        keep_mask = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            graffiti_box = (float(x1), float(y1), float(x2), float(y2))
            
            # Check if graffiti box is contained in or significantly overlaps any exclusion box
            should_filter = False
            for exclusion_box in exclusion_boxes:
                # Check containment (center inside or significant overlap)
                if is_box_contained(graffiti_box, exclusion_box):
                    should_filter = True
                    break
                # Also check IoU as a fallback
                iou = calculate_iou(graffiti_box, exclusion_box)
                if iou > iou_threshold:
                    should_filter = True
                    break
            
            # Keep graffiti if it's NOT filtered out
            keep_mask.append(not should_filter)
        
        # Filter boxes using the mask
        if any(keep_mask):
            import torch
            keep_indices = torch.tensor(keep_mask, device=boxes.xyxy.device, dtype=torch.bool)
            # Create new boxes with filtered indices
            filtered_boxes = boxes[keep_indices]
            result.boxes = filtered_boxes
        else:
            # No boxes to keep - set boxes to empty
            result.boxes = None
        
        filtered_results.append(result)
    
    return filtered_results


def process_image(graffiti_model, person_detector, source, conf, human_conf, iou, 
                  imgsz, device, save, show, project, name, haar_face_detector=None):
    """Process a single image or directory of images with both models and MediaPipe."""
    source_path = Path(source)
    
    # Check if source is a directory
    if source_path.is_dir():
        # Process all images in directory
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(source_path.glob(f'*{ext}')))
            image_files.extend(list(source_path.glob(f'*{ext.upper()}')))
        
        if not image_files:
            print(f"No image files found in {source}")
            return
        
        print(f"Found {len(image_files)} images in directory. Processing...")
        for img_file in image_files:
            print(f"\nProcessing: {img_file.name}")
            process_single_image(graffiti_model, person_detector, str(img_file), conf, human_conf, iou,
                                imgsz, device, save, show, project, name, haar_face_detector)
        return
    
    # Process single image
    process_single_image(graffiti_model, person_detector, source, conf, human_conf, iou,
                        imgsz, device, save, show, project, name, haar_face_detector)


def process_single_image(graffiti_model, person_detector, source, conf, human_conf, iou, 
                         imgsz, device, save, show, project, name, haar_face_detector=None):
    """Process a single image with both models and MediaPipe."""
    # Read image
    img = cv2.imread(source)
    if img is None:
        print(f"Error: Could not read image from {source}")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Run person detection FIRST
    human_results = person_detector.detect_persons(
        img_rgb, conf=human_conf, iou=iou, imgsz=imgsz, device=device
    )
    
    # Run graffiti detection
    graffiti_results = graffiti_model.predict(
        source=img_rgb,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        verbose=False
    )
    
    # Filter out graffiti detections that overlap with persons or faces
    graffiti_results = filter_graffiti_overlapping_persons(graffiti_results, human_results, person_detector, img_rgb, haar_face_detector)
    
    # Draw graffiti detections
    annotated_img = graffiti_results[0].plot()
    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
    annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    
    # Process and draw person detections with MediaPipe
    annotated_img_rgb = person_detector.process_person_detections(annotated_img_rgb, human_results)
    
    # Draw Haar Cascade face boxes if enabled
    if haar_face_detector is not None:
        annotated_img_rgb = haar_face_detector.draw_face_boxes(annotated_img_rgb, human_results)
    
    # Convert back to BGR for saving/displaying
    annotated_img = cv2.cvtColor(annotated_img_rgb, cv2.COLOR_RGB2BGR)
    
    # Save or show result
    if save:
        output_dir = Path(project) / name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / Path(source).name
        cv2.imwrite(str(output_path), annotated_img)
        print(f"Saved annotated image to {output_path}")
    
    if show:
        cv2.imshow('Result', annotated_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_video(graffiti_model, person_detector, source, conf, human_conf, iou, 
                  imgsz, device, save, show, project, name, haar_face_detector=None):
    """Process video or webcam with both models and MediaPipe."""
    # Open video source
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup video writer if saving
    writer = None
    if save:
        output_dir = Path(project) / name
        output_dir.mkdir(parents=True, exist_ok=True)
        if source.isdigit():
            output_path = output_dir / 'webcam_output.mp4'
        else:
            output_path = output_dir / Path(source).name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        print(f"Saving video to {output_path}")
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run person detection FIRST
            human_results = person_detector.detect_persons(
                frame_rgb, conf=human_conf, iou=iou, imgsz=imgsz, device=device
            )
            
            # Run graffiti detection
            graffiti_results = graffiti_model.predict(
                source=frame_rgb,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                device=device,
                verbose=False
            )
            
            # Filter out graffiti detections that overlap with persons or faces
            graffiti_results = filter_graffiti_overlapping_persons(graffiti_results, human_results, person_detector, frame_rgb, haar_face_detector)
            
            # Draw graffiti detections
            annotated_frame = graffiti_results[0].plot()
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Process and draw person detections with MediaPipe
            annotated_frame_rgb = person_detector.process_person_detections(annotated_frame_rgb, human_results)
            
            # Convert back to BGR
            annotated_frame = cv2.cvtColor(annotated_frame_rgb, cv2.COLOR_RGB2BGR)
            
            # Write frame
            if writer is not None:
                writer.write(annotated_frame)
            
            # Show frame
            if show:
                cv2.imshow('Result', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Progress update
            if total_frames > 0 and frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames...")
    
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()
    
    print(f"Processed {frame_count} frames")


if __name__ == '__main__':
    main()
