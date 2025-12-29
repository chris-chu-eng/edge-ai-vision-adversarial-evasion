"""
Surveillance Evasion System
---------------------------
A computer vision project exploring the vulnerabilities of real-time object detection.
This program implements a 'visible' sensor layer using YOLOv8 to track targets,
serving as a baseline to later demonstrate how physical adversarial patches can defeat
standard AI models.

Thermal imaging and adversarial patch attacks to be implemented in the future.

Author: Christopher Chu
Date: Dec 2025
Context: Study in Adversarial ML & Defense Systems
"""

import cv2
import math
import cvzone
import sys
from typing import Tuple
from ultralytics import YOLO
from numpy import ndarray

# CONSTANTS
MODEL_VERSION = "yolov8n.pt"  # Nano model for speed and edge device emulation
CONFIDENCE_THRESHOLD = 0.45   # Minimum threshold to flag an object
TARGET_OBJECT = "person"      # The specific object ID to track

# COCO dataset used to train the YOLO model, used as a lookup table
OBJECT_NAMES = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
    "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

def initialize_system(cam_id: int = 0, width: int = 1280, height: int = 720) -> Tuple[cv2.VideoCapture, YOLO]:
    """
    Initializes the hardware connection and loads the YOLO model.

    Args:
        cam_id (int): The device index for the camera (default 0).
        width (int): Desired frame width.
        height (int): Desired frame height.

    Returns:
        Tuple[cv2.VideoCapture, YOLO]: The camera object and the loaded model.
    """
    # 1. Initializes the camera
    cam_capture = cv2.VideoCapture(cam_id)
    cam_capture.set(3, width)
    cam_capture.set(4, height)

    # Checking for camera access
    if not cam_capture.isOpened():
        print(f"ERROR: Could not access Camera device {cam_id}.")
        sys.exit(1)

    # 2. Initialize the YOLO model
    print(f"SYSTEM: Loading YOLO Model - ({MODEL_VERSION})...")
    try:
        model = YOLO(MODEL_VERSION)
    except Exception as e:
        print(f"ERROR: Failed to load model - {e}")
        sys.exit(1)

    return cam_capture, model

def process_frame(img: ndarray, model: YOLO) -> ndarray:
    """
    Processes a single frame and draws boxes around detected targets.

    Args:
        img (ndarray): The raw video frame.
        model (YOLO): The loaded neural network.

    Returns:
        ndarray: The processed frame with overlay graphics.
    """
    # Runs "inference", putting the image through the model
    results = model(img, stream=True, verbose=False) # stream=True for RAM efficiency

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Retrieves confidence score
            conf = math.ceil((box.conf[0] * 100)) / 100
            
            # Retrieves object ID and name
            cls_id = int(box.cls[0])
            current_class = OBJECT_NAMES[cls_id]

            # Use Case Filter: Only track high confidence value humans
            if current_class == TARGET_OBJECT and conf > CONFIDENCE_THRESHOLD:
                # Retrieve coordinates to create the boundary box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1

                # Draws the HUD (boundary box, label, confidence score)
                cvzone.cornerRect(img, (x1, y1, w, h), l=30, t=5, colorR=(0, 0, 255))
                cvzone.putTextRect(
                    img, 
                    f'TARGET: {current_class.upper()} | ACCURACY: {int(conf*100)}%', 
                    (max(0, x1), max(35, y1)), 
                    scale=2, 
                    thickness=2, 
                    offset=10, 
                    colorR=(0,0,0)
                )

    return img

def main():
    """Main execution loop."""
    cap, model = initialize_system()
    print("SYSTEM: Program active. Press 'q' to abort.")

    while True:
        success, img = cap.read()
        if not success:
            print("WARNING: Failed to read frame from camera.")
            break

        # Processs the frame (inference + drawing HUD)
        output_img = process_frame(img, model)

        # Renders to the screen
        cv2.imshow("Surveillance Feed Prototype", output_img)

        # Exit condition, press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("SYSTEM: Program closed.")

if __name__ == "__main__":
    main()
