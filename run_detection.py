#!/usr/bin/env python3
"""
Convenience script for running inference with default settings:
- Uses best.pt model
- Enables Haar Cascade face detection
- Includes person detection and MediaPipe pose estimation
"""

import subprocess
import sys
from pathlib import Path

def main():
    # Default settings
    model = 'train/weights/best.pt'
    use_haar_face = True
    
    # Get source from command line arguments
    if len(sys.argv) < 2:
        print("Usage: python run_detection.py <source> [additional_args...]")
        print("\nExample:")
        print("  python run_detection.py ~/Downloads/persontest.jpg")
        print("  python run_detection.py ~/Desktop/image.png --conf 0.15")
        sys.exit(1)
    
    source = sys.argv[1]
    
    # Build command
    cmd = [
        'python', 'run_inference.py',
        '--model', model,
        '--source', source,
        '--save',
        '--use-haar-face'
    ]
    
    # Add any additional arguments passed by user
    if len(sys.argv) > 2:
        cmd.extend(sys.argv[2:])
    
    # Run the inference script
    print(f"Running inference with:")
    print(f"  Model: {model}")
    print(f"  Source: {source}")
    print(f"  Haar Cascade: Enabled")
    print(f"  Additional args: {sys.argv[2:] if len(sys.argv) > 2 else 'None'}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True)
        sys.exit(result.returncode)
    except subprocess.CalledProcessError as e:
        print(f"Error running inference: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)

if __name__ == '__main__':
    main()

