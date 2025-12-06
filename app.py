#!/usr/bin/env python3
"""
Kavach - Graffiti Detection Web Application
A modern UI/UX web interface for graffiti detection with person and pose estimation.
"""

import os
import json
import sys
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
from werkzeug.utils import secure_filename
import cv2
from ultralytics import YOLO
from person_detection import PersonDetector
from haar_face_detection import HaarFaceDetector
import numpy as np

# Import filtering functions from run_inference
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_inference import filter_graffiti_overlapping_persons, calculate_iou, is_box_contained

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'runs/detect/inference'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'mp4', 'avi', 'mov', 'mkv'}

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Initialize models (lazy loading)
graffiti_model = None
person_detector = None
haar_face_detector = None

def init_models():
    """Initialize ML models (lazy loading)"""
    global graffiti_model, person_detector, haar_face_detector
    
    if graffiti_model is None:
        print("Loading graffiti model...")
        graffiti_model = YOLO('train/weights/best.pt')
        print("Graffiti model loaded!")
    
    if person_detector is None:
        print("Initializing person detector...")
        person_detector = PersonDetector(
            human_model_path='yolo11n.pt',
            enable_pose=True,
            enable_face=False,
            keypoint_color=(255, 0, 255),
            connection_color=(0, 255, 255)
        )
        print("Person detector initialized!")
    
    if haar_face_detector is None:
        print("Loading Haar Cascade...")
        try:
            haar_face_detector = HaarFaceDetector()
            print("Haar Cascade loaded!")
        except Exception as e:
            print(f"Error loading Haar Cascade: {e}")
            haar_face_detector = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def is_image_file(filename):
    """Check if file is an image"""
    return filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def is_video_file(filename):
    """Check if file is a video"""
    return filename.rsplit('.', 1)[1].lower() in {'mp4', 'avi', 'mov', 'mkv'}

def process_image_file(filepath, conf=0.11):
    """Process an image file and return result path"""
    init_models()
    
    # Read image
    img = cv2.imread(filepath)
    if img is None:
        raise ValueError("Could not read image file")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Detect persons
    human_results = person_detector.detect_persons(
        img_rgb, conf=0.25, iou=0.7, imgsz=640
    )
    
    # Detect graffiti
    graffiti_results = graffiti_model.predict(
        source=img_rgb,
        conf=conf,
        iou=0.7,
        imgsz=640,
        verbose=False
    )
    
    # Filter graffiti overlapping with faces
    graffiti_results = filter_graffiti_overlapping_persons(
        graffiti_results, human_results, person_detector, img_rgb, haar_face_detector
    )
    
    # Draw graffiti detections
    annotated_img = graffiti_results[0].plot()
    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
    annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    
    # Process person detections with MediaPipe
    annotated_img_rgb = person_detector.process_person_detections(annotated_img_rgb, human_results)
    
    # Draw Haar Cascade face boxes
    if haar_face_detector is not None:
        annotated_img_rgb = haar_face_detector.draw_face_boxes(annotated_img_rgb, human_results)
    
    # Save result
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_filename = f"kavach_{timestamp}.png"
    result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
    
    annotated_img_bgr = cv2.cvtColor(annotated_img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(result_path, annotated_img_bgr)
    
    return result_filename

def process_video_file(filepath, conf=0.11):
    """Process a video file and return result path"""
    init_models()
    
    # Open video
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        raise ValueError("Could not open video file")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output video
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_filename = f"kavach_{timestamp}.mp4"
    result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(result_path, fourcc, fps, (width, height))
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect persons
        human_results = person_detector.detect_persons(
            frame_rgb, conf=0.25, iou=0.7, imgsz=640
        )
        
        # Detect graffiti
        graffiti_results = graffiti_model.predict(
            source=frame_rgb,
            conf=conf,
            iou=0.7,
            imgsz=640,
            verbose=False
        )
        
        # Filter graffiti overlapping with faces
        graffiti_results = filter_graffiti_overlapping_persons(
            graffiti_results, human_results, person_detector, frame_rgb, haar_face_detector
        )
        
        # Draw graffiti detections
        annotated_frame = graffiti_results[0].plot()
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        # Process person detections with MediaPipe
        annotated_frame_rgb = person_detector.process_person_detections(annotated_frame_rgb, human_results)
        
        # Draw Haar Cascade face boxes
        if haar_face_detector is not None:
            annotated_frame_rgb = haar_face_detector.draw_face_boxes(annotated_frame_rgb, human_results)
        
        # Write frame
        annotated_frame_bgr = cv2.cvtColor(annotated_frame_rgb, cv2.COLOR_RGB2BGR)
        out.write(annotated_frame_bgr)
    
    cap.release()
    out.release()
    
    return result_filename

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    # Get confidence threshold
    conf = float(request.form.get('conf', 0.11))
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
        file.save(filepath)
        
        # Process file
        if is_image_file(filename):
            result_filename = process_image_file(filepath, conf)
            file_type = 'image'
        elif is_video_file(filename):
            result_filename = process_video_file(filepath, conf)
            file_type = 'video'
        else:
            return jsonify({'error': 'Unsupported file type'}), 400
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'result_file': result_filename,
            'file_type': file_type,
            'message': 'Processing completed successfully'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/past-runs', methods=['GET'])
def get_past_runs():
    """Get list of past run results"""
    results = []
    results_dir = Path(app.config['RESULTS_FOLDER'])
    
    if results_dir.exists():
        # Get all image and video files
        for file_path in sorted(results_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.mp4', '.avi', '.mov', '.mkv']:
                    stat = file_path.stat()
                    results.append({
                        'filename': file_path.name,
                        'size': stat.st_size,
                        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        'type': 'image' if ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp'] else 'video'
                    })
    
    return jsonify({'results': results})

@app.route('/api/results/<filename>')
def get_result(filename):
    """Serve result files"""
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

@app.route('/api/results/<filename>', methods=['DELETE'])
def delete_result(filename):
    """Delete a result file"""
    try:
        file_path = os.path.join(app.config['RESULTS_FOLDER'], filename)
        
        # Security check: ensure file is in results folder
        if not os.path.abspath(file_path).startswith(os.path.abspath(app.config['RESULTS_FOLDER'])):
            return jsonify({'error': 'Invalid file path'}), 400
        
        if os.path.exists(file_path):
            os.remove(file_path)
            return jsonify({'success': True, 'message': 'File deleted successfully'})
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/process-frame', methods=['POST'])
def process_frame():
    """Process a single frame from webcam - only save if faces detected"""
    try:
        # Get image data from request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        conf = float(request.form.get('conf', 0.11))
        
        # Save temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            file.save(tmp_file.name)
            tmp_path = tmp_file.name
        
        try:
            # Process the frame
            if is_image_file(tmp_path):
                init_models()
                
                # Read image
                img = cv2.imread(tmp_path)
                if img is None:
                    return jsonify({'error': 'Could not read image file'}), 400
                
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Detect persons
                human_results = person_detector.detect_persons(
                    img_rgb, conf=0.25, iou=0.7, imgsz=640
                )
                
                # Check if faces are detected BEFORE processing
                face_boxes = []
                if haar_face_detector is not None:
                    face_boxes = haar_face_detector.get_face_boxes_from_persons(img_rgb, human_results)
                
                has_faces = len(face_boxes) > 0
                
                # Detect graffiti
                graffiti_results = graffiti_model.predict(
                    source=img_rgb,
                    conf=conf,
                    iou=0.7,
                    imgsz=640,
                    verbose=False
                )
                
                # Filter graffiti overlapping with faces
                graffiti_results = filter_graffiti_overlapping_persons(
                    graffiti_results, human_results, person_detector, img_rgb, haar_face_detector
                )
                
                # Draw graffiti detections
                annotated_img = graffiti_results[0].plot()
                annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
                annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                
                # Process person detections with MediaPipe
                annotated_img_rgb = person_detector.process_person_detections(annotated_img_rgb, human_results)
                
                # Draw Haar Cascade face boxes
                if haar_face_detector is not None:
                    annotated_img_rgb = haar_face_detector.draw_face_boxes(annotated_img_rgb, human_results)
                
                # Convert to base64 for display
                import base64
                annotated_img_bgr = cv2.cvtColor(annotated_img_rgb, cv2.COLOR_RGB2BGR)
                _, buffer = cv2.imencode('.jpg', annotated_img_bgr)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Only save if faces are detected
                saved_filename = None
                if has_faces:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    result_filename = f"kavach_live_{timestamp}.png"
                    result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
                    cv2.imwrite(result_path, annotated_img_bgr)
                    saved_filename = result_filename
                
                return jsonify({
                    'success': True,
                    'image': f'data:image/jpeg;base64,{img_base64}',
                    'has_faces': has_faces,
                    'saved': saved_filename is not None,
                    'filename': saved_filename
                })
            else:
                return jsonify({'error': 'Invalid image format'}), 400
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'Kavach API is running'})

if __name__ == '__main__':
    print("Starting Kavach Web Application...")
    print("Initializing models (this may take a moment)...")
    init_models()
    print("Models initialized!")
    print("Starting Flask server...")
    print("Server will be available at http://localhost:5001")
    app.run(debug=True, host='0.0.0.0', port=5001)
