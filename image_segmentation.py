import cv2
import numpy as np
import os
import time
from datetime import datetime

# Configuration
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
SAVE_INTERVAL = 2  # Seconds between saves

def create_dark_heatmap(mask):
    """Creates heatmap with dark background and bright boundaries"""
    edges = cv2.Canny(mask, 50, 150)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    heatmap = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    colored_motion = cv2.applyColorMap(mask, cv2.COLORMAP_INFERNO)
    heatmap[edges > 0] = colored_motion[edges > 0]
    return cv2.addWeighted(heatmap, 1.5, np.zeros_like(heatmap), 0, 30)

def add_title_to_image(image, title):
    """Adds a title to the top of an image"""
    title_bar = np.zeros((40, image.shape[1], 3), dtype=np.uint8)
    cv2.putText(title_bar, title, (image.shape[1]//2 - 100, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return cv2.vconcat([title_bar, image])

def add_info_panel(frame, fps, obj_count):
    """Adds organized information panel to the frame"""
    info_panel = np.zeros((100, frame.shape[1], 3), dtype=np.uint8)
    
    # Column 1: System info
    cv2.putText(info_panel, f"Time: {datetime.now().strftime('%H:%M:%S')}", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    cv2.putText(info_panel, f"FPS: {fps:.1f}", 
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    
    # Column 2: Detection info
    cv2.putText(info_panel, f"Objects: {obj_count}", 
               (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    cv2.putText(info_panel, f"Frame: {frame_count}", 
               (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    
    return cv2.vconcat([frame, info_panel])

def main():
    global frame_count
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot access camera")
        return

    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, detectShadows=False)
    fps = cap.get(cv2.CAP_PROP_FPS)
    save_frame_count = int(fps * SAVE_INTERVAL)
    frame_count = 0
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # Processing
        frame = cv2.resize(frame, (640, 480))
        fg_mask = fgbg.apply(frame)
        fg_mask = cv2.medianBlur(fg_mask, 5)
        
        # Object detection
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        obj_count = sum(1 for cnt in contours if cv2.contourArea(cnt) > 500)
        
        # Original frame with objects marked
        original_display = frame.copy()
        for i, cnt in enumerate(contours):
            if cv2.contourArea(cnt) > 500:
                x,y,w,h = cv2.boundingRect(cnt)
                cv2.rectangle(original_display, (x,y), (x+w,y+h), (0,255,0), 2)
                cv2.putText(original_display, str(i+1), (x+5,y+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        
        # Add organized info panel
        original_display = add_info_panel(original_display, fps, obj_count)
        
        # Create heatmap (unchanged)
        heatmap_display = create_dark_heatmap(fg_mask)
        for cnt in contours:
            if cv2.contourArea(cnt) > 500:
                cv2.drawContours(heatmap_display, [cnt], -1, (0,255,255), 2)
        
        # Add titles to each image
        original_display = add_title_to_image(original_display, "Original Image")
        heatmap_display = add_title_to_image(heatmap_display, "Motion Heatmap")
        
        # Combine views
        combined = np.hstack((
            cv2.resize(original_display, (640, 620)),
            cv2.resize(heatmap_display, (640, 620))
        ))
        
        # Save output
        if frame_count % save_frame_count == 0:
            filename = f"{OUTPUT_DIR}/result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(filename, combined)
            print(f"SAVED: {filename}")

        # Display
        cv2.imshow("Motion Analysis (ESC to quit)", combined)
        frame_count += 1

        if cv2.waitKey(30) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Results saved to '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()