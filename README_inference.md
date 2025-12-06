# Running YOLOv11 Inference with Human Detection and MediaPipe Pose Estimation

This script runs dual-model inference:
1. **Vandalism Detection**: Your trained YOLOv11 model for detecting vandalism
2. **Human Detection**: YOLOv11 model for detecting people
3. **MediaPipe Pose**: Pose estimation on detected humans

## Installation

Install required packages:
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```bash
# Run on a single image
python run_inference.py --source path/to/image.jpg --save

# Run on a video
python run_inference.py --source path/to/video.mp4 --save

# Run on webcam (camera 0)
python run_inference.py --source 0 --show

# Run on all images in a directory
python run_inference.py --source path/to/images/ --save
```

### Using Different Models

```bash
# Use the best model from training
python run_inference.py --source image.jpg --model train/weights/best.pt --save

# Use the last checkpoint
python run_inference.py --source image.jpg --model train/weights/last.pt --save

# Use the root model file
python run_inference.py --source image.jpg --model my_model.pt --save
```

### Common Options

```bash
# Adjust confidence threshold (default: 0.25)
python run_inference.py --source image.jpg --conf 0.5 --save

# Show results in a window
python run_inference.py --source image.jpg --show

# Save detection labels as .txt files
python run_inference.py --source image.jpg --save --save-txt

# Save with confidence scores in labels
python run_inference.py --source image.jpg --save --save-txt --save-conf

# Use GPU (if available)
python run_inference.py --source image.jpg --device cuda --save

# Use CPU explicitly
python run_inference.py --source image.jpg --device cpu --save

# Human Detection and MediaPipe Options
# Use custom human detection model
python run_inference.py --source image.jpg --human-model yolo11m.pt --save

# Disable MediaPipe pose estimation (only human bounding boxes)
python run_inference.py --source image.jpg --disable-mediapipe --save

# Adjust human detection confidence
python run_inference.py --source image.jpg --human-conf 0.5 --save
```

### Full Example

```bash
python run_inference.py \
    --source path/to/image.jpg \
    --model my_model.pt \
    --conf 0.25 \
    --iou 0.7 \
    --imgsz 640 \
    --save \
    --show \
    --save-txt \
    --save-conf \
    --project runs/detect \
    --name my_inference
```

## Features

### Dual Model Detection
- **Vandalism Detection**: Detects vandalism using your trained model (shown with default YOLO colors)
- **Human Detection**: Detects people using YOLOv11 (shown with blue bounding boxes)
- **MediaPipe Pose**: Applies pose estimation on detected humans (shown with green keypoints and red connections)

### Visualization
- Vandalism detections: Default YOLO annotation style
- Human bounding boxes: Blue rectangles with "Person" label
- Pose keypoints: Green circles for joints
- Pose connections: Red lines connecting body parts

## Output

Results will be saved to `runs/detect/inference/` (or your custom project/name path) with:
- Annotated images/videos showing both vandalism and human detections with pose estimation
- Optional: `.txt` label files (if `--save-txt` is used)

## Command Line Arguments

### Vandalism Detection
- `--model`: Path to vandalism detection model (default: `my_model.pt`)
- `--conf`: Confidence threshold for vandalism (default: 0.25)
- `--iou`: IoU threshold for NMS (default: 0.7)

### Human Detection
- `--human-model`: YOLO model for human detection (default: `yolo11n.pt`)
- `--human-conf`: Confidence threshold for human detection (default: 0.25)
- `--enable-mediapipe`: Enable MediaPipe pose estimation (default: True)

### General Options
- `--source`: Input source (image, video, directory, or "0" for webcam)
- `--save`: Save results
- `--show`: Show results in window
- `--device`: Device to use (cuda, cpu, or None for auto)


