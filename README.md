# 🛡️ Kavach - Vandalism Detection Web Application

A modern, beautiful web interface for vandalism detection with person and pose estimation using AI.

## 🚀 Powered By AI

Kavach integrates multiple computer vision techniques for robust detection:

- **YOLOv11** – High-performance real-time object detection  
  - Custom-trained vandalism (graffiti) detection model  
  - Human/person detection model
- **MediaPipe** – Human pose estimation (skeleton overlay)
- **Haar Cascade Classifiers** – Lightweight and fast face detection

These models work together to provide **context-aware vandalism detection** with visual explainability.

## Features

- 📸 **Image Upload & Processing** - Upload images and detect vandalism, persons, and poses
- 🎥 **Video Upload & Processing** - Process videos frame by frame
- 🎨 **Modern UI/UX** - Beautiful, responsive design with dark theme
- 📊 **Confidence Threshold Control** - Adjustable detection sensitivity (1-100%)
- 📁 **Past Runs Gallery** - View all previously processed results
- ⚡ **Real-time Processing** - Fast inference with YOLOv11 and MediaPipe

## Installation

1. **Activate virtual environment:**
```bash
source env/bin/activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Ensure models are in place:**
   - Vandalism model: `train/weights/best.pt`
   - Human detection model: `yolo11n.pt`

## Running the Application

1. **Start the Flask server:**
```bash
python app.py
```

2. **Open your browser:**
Navigate to `http://localhost:5000`

## Usage

1. **Upload Media:**
   - Click the upload area or drag & drop an image/video file
   - Supported formats: PNG, JPG, JPEG, GIF, BMP, MP4, AVI, MOV, MKV

2. **Adjust Settings:**
   - Use the confidence slider to adjust detection sensitivity
   - Lower values = more detections (may include false positives)
   - Higher values = fewer detections (more strict)

3. **Process:**
   - Click "Process" to run inference
   - Results will appear below

4. **View Past Runs:**
   - Scroll down to see gallery of all previous results
   - Click any item to view full size

## API Endpoints

- `GET /` - Main web interface
- `POST /api/upload` - Upload and process file
- `GET /api/past-runs` - Get list of past results
- `GET /api/results/<filename>` - Download result file
- `GET /api/health` - Health check

## File Structure

```
vandalism/
├── app.py                 # Flask application
├── templates/
│   └── index.html        # Main HTML template
├── static/
│   ├── css/
│   │   └── style.css     # Stylesheet
│   └── js/
│       └── app.js        # JavaScript
├── uploads/              # Temporary upload storage
└── runs/detect/inference/ # Results storage
```

## Output Examples

### Example 1: Vandalism Detection on Brick Wall

<img width="334" height="211" alt="Vandalism detection on a brick wall" src="https://github.com/user-attachments/assets/237ed3ba-63f7-491e-9dd7-a116191047f2" />

*Graffiti detected on a brick wall with confidence scores displayed.*


### Example 2: Multiple Graffiti Detections

<img width="465" height="194" alt="Screenshot 2025-12-06 at 8 53 42 PM" src="https://github.com/user-attachments/assets/01fb461f-21b7-4c8f-8e68-3187acc5c63c" />

*Detection of multiple graffiti elements with individual bounding boxes and confidence scores*

### Example 3: Combined Detection (Vandalism + Person + Pose)

<img width="634" height="305" alt="Screenshot 2025-12-06 at 8 53 14 PM" src="https://github.com/user-attachments/assets/52d14a76-fdf7-4be6-a2d1-312899807942" />

*Complete detection showing vandalism (red boxes), person detection (red boxes), face detection (yellow boxes), and pose estimation (skeleton overlay)*

## Notes

- Models are loaded lazily on first request (may take a moment)
- Uploaded files are automatically cleaned up after processing
- Results are saved in `runs/detect/inference/` directory
- Maximum file size: 500MB

## Troubleshooting

- **Models not loading:** Ensure `train/weights/best.pt` and `yolo11n.pt` exist
- **Port already in use:** Change port in `app.py` (last line)
- **Memory issues:** Reduce video resolution or process shorter clips

