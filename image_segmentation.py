import cv2
import numpy as np
import os
from datetime import datetime

# Configuration
OUTPUT_DIR = "motion3_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
MIN_AREA = 500  # Minimum contour area in pixels

def main():
    # Initialize video capture (0 for webcam, or path for video file)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video source")
        return

    # Background subtractor with shadow detection
    backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    backSub.setShadowValue(127)

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame")
            break

        # Preprocessing
        frame = cv2.resize(frame, (640, 480))
        h, w = frame.shape[:2]

        # Motion detection
        fg_mask = backSub.apply(frame)
        fg_mask[fg_mask == 127] = 0  # Remove shadows

        # Noise removal
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Left panel: Original with detection
        left_panel = frame.copy()
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > MIN_AREA:
                x, y, w_, h_ = cv2.boundingRect(cnt)
                cv2.rectangle(left_panel, (x,y), (x+w_,y+h_), (0,255,0), 2)
                cv2.putText(left_panel, f"{area}px", (x,y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        # Right panel: Enhanced heatmap
        normalized = cv2.normalize(fg_mask, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
        
        # Blend with original (40% heatmap)
        right_panel = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)
        
        # Draw enhanced contours
        for cnt in contours:
            if cv2.contourArea(cnt) > MIN_AREA:
                # Outer contour (3px)
                cv2.drawContours(right_panel, [cnt], -1, (0,255,255), 3)
                # Inner contour (1px)
                cv2.drawContours(right_panel, [cnt], -1, (0,0,0), 1)

        # Combine panels
        comparison = np.hstack((left_panel, right_panel))

        # Add metrics panel
        motion_pixels = np.sum(fg_mask > 0)
        motion_percent = (motion_pixels / (h*w)) * 100
        moving_objects = len([cnt for cnt in contours if cv2.contourArea(cnt) > MIN_AREA])
        
        metrics = np.zeros((100, w*2, 3), dtype=np.uint8)
        cv2.putText(metrics, "Motion Analysis Report", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(metrics, f"Objects: {moving_objects} | Coverage: {motion_percent:.1f}%", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.putText(metrics, f"Frame: {frame_count} | Time: {datetime.now().strftime('%H:%M:%S')}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        final_output = np.vstack((comparison, metrics))

        # Display
        cv2.imshow("Motion Segmentation Analysis", final_output)

        # Save frame every 2 seconds (assuming ~30fps)
        if frame_count % 60 == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"{OUTPUT_DIR}/frame_{timestamp}.png", final_output)

        frame_count += 1

        # Exit on ESC
        if cv2.waitKey(30) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Processing complete. Output saved to '{OUTPUT_DIR}' folder")

if __name__ == "__main__":
    main()