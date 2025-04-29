import cv2
import numpy as np
import os
from datetime import datetime

# Configuration
OUTPUT_DIR = "motion_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
SAVE_INTERVAL = 2  # Seconds between saves

def enhance_edges(mask):
    """Sharpens motion boundaries using edge detection"""
    edges = cv2.Canny(mask, 50, 150)
    kernel = np.ones((3,3), np.uint8)
    return cv2.dilate(edges, kernel, iterations=1)

def create_heatmap(frame, mask):
    """Generates high-contrast heatmap with clear edges"""
    # Step 1: Enhance edges
    sharp_mask = cv2.GaussianBlur(mask, (5,5), 0)
    sharp_mask = cv2.addWeighted(mask, 1.5, sharp_mask, -0.5, 0)
    
    # Step 2: Apply color mapping
    heatmap = cv2.applyColorMap(sharp_mask, cv2.COLORMAP_INFERNO)
    
    # Step 3: Highlight edges
    edges = enhance_edges(mask)
    heatmap[edges > 0] = [0, 255, 255]  # Yellow edges
    
    # Step 4: Blend with original
    return cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot access camera")
        return

    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, detectShadows=False)
    fps = cap.get(cv2.CAP_PROP_FPS)
    save_frame_count = int(fps * SAVE_INTERVAL)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("WARNING: Frame read error")
            break

        # Processing pipeline
        frame = cv2.resize(frame, (640, 480))
        fg_mask = fgbg.apply(frame)
        fg_mask = cv2.medianBlur(fg_mask, 5)
        
        # Create displays
        original_display = frame.copy()
        heatmap_display = create_heatmap(frame, fg_mask)
        
        # Draw precise contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 500:
                # Original frame (green box)
                x,y,w,h = cv2.boundingRect(cnt)
                cv2.rectangle(original_display, (x,y), (x+w,y+h), (0,255,0), 2)
                
                # Heatmap frame (red contour)
                cv2.drawContours(heatmap_display, [cnt], -1, (0,0,255), 2)

        # Combine and annotate
        combined = np.hstack((original_display, heatmap_display))
        cv2.putText(combined, f"Original | {datetime.now().strftime('%H:%M:%S')}", (10,30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(combined, "Enhanced Heatmap", (650,30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        # Save output
        if frame_count % save_frame_count == 0:
            filename = f"{OUTPUT_DIR}/result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, combined, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"SAVED: {filename}")

        # Display
        cv2.imshow("Live Motion Analysis (ESC to quit)", combined)
        frame_count += 1

        if cv2.waitKey(30) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Processing complete. Results saved in '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()