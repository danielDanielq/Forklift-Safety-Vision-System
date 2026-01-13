import cv2
import numpy as np
import os
from utils import refine_mask, get_person_detections

# Window size configuration
WIN_WIDTH = 640
WIN_HEIGHT = 360

def create_named_window(name, w, h):
    """Creates a resizable window with fixed dimensions."""
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, w, h)

create_named_window('0. Raw (Original)', WIN_WIDTH, WIN_HEIGHT)
create_named_window('1. Blurred (Noise Reduction)', WIN_WIDTH, WIN_HEIGHT)
create_named_window('2. MOG2 Raw Mask', WIN_WIDTH, WIN_HEIGHT)
create_named_window('3. Refined Mask (Morphology)', WIN_WIDTH, WIN_HEIGHT)
create_named_window('4. Final Detection', WIN_WIDTH, WIN_HEIGHT)

video_path = os.path.join('data', 'test_video.mp4')
cap = cv2.VideoCapture(video_path if os.path.exists(video_path) else 0)

# MOG2 Initialization
# High threshold (250) to ignore shelf vibrations
backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=250, detectShadows=True)

while True:
    ret, frame = cap.read()

    # 1. Pre-processing
    frame_blurred = cv2.GaussianBlur(frame, (5, 5), 0)

    # 2. MOG2  
    fg_mask = backSub.apply(frame_blurred)

    # 3. Morphology 
    cleaned_mask = refine_mask(fg_mask)

    # 4 & 5. Segmentation and Analysis
    detections = get_person_detections(cleaned_mask)

    # Draw Alert on the final frame
    frame_final = frame.copy()
    
    if len(detections) > 0:
        # DANGER CASE (RED)
        cv2.putText(frame_final, "ALERT: PERSON DETECTED!", (40, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        for (x, y, w, h) in detections:
            cv2.rectangle(frame_final, (x, y), (x+w, y+h), (0, 0, 255), 4)
    else:
        # SAFETY CASE (GREEN)
        cv2.putText(frame_final, "SAFE ZONE", (40, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)

    cv2.imshow('0. Raw (Original)', frame)
    cv2.imshow('1. Blurred (Noise Reduction)', frame_blurred)
    cv2.imshow('2. MOG2 Raw Mask', fg_mask)
    cv2.imshow('3. Refined Mask (Morphology)', cleaned_mask)
    cv2.imshow('4. Final Detection', frame_final)

    if cv2.waitKey(30) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()