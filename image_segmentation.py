import cv2
import sys
import os
from datetime import datetime
import numpy as np

# Configuration
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
SAVE_INTERVAL = 2  # Save every 2 seconds

def main():
    # Initialize background subtractor
    backSub = cv2.createBackgroundSubtractorMOG2()
    
    # Try to open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        sys.exit(1)
    
    print("Press ESC to quit...")
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    save_frame_count = int(fps * SAVE_INTERVAL)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame. Exiting...")
            break
        
        # Motion detection
        fg_mask = backSub.apply(frame)
        
        # Noise removal
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Find and draw contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 500:
                x,y,w,h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        
        # Display
        cv2.imshow('Live Motion Detection', frame)
        cv2.imshow('Foreground Mask', fg_mask)
        
        # Save frames at specified interval
        if frame_count % save_frame_count == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"{OUTPUT_DIR}/frame_{timestamp}.jpg", frame)
            cv2.imwrite(f"{OUTPUT_DIR}/mask_{timestamp}.jpg", fg_mask)
            print(f"Saved frame and mask at {timestamp}")
        
        frame_count += 1
        
        # Exit on ESC
        if cv2.waitKey(30) == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"All outputs saved to '{OUTPUT_DIR}' folder")

if __name__ == "__main__":
    main()