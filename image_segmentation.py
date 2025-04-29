import cv2
import numpy as np
import os
from datetime import datetime

# Configuration
OUTPUT_DIR = "motion_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    # Initialize video capture (0 for webcam, or file path)
    cap = cv2.VideoCapture(0)  # Change to your video path if needed
    
    # Background subtractor with shadow detection
    backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    backSub.setShadowValue(127)  # Gray values are shadows
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize for faster processing
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
        
        # Create comparison frame
        comparison = np.zeros((h, w*2, 3), dtype=np.uint8)
        
        # Left panel: Original with bounding boxes
        left_panel = frame.copy()
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:  # Filter small areas
                x,y,w_,h_ = cv2.boundingRect(cnt)
                cv2.rectangle(left_panel, (x,y), (x+w_,y+h_), (0,255,0), 2)
                cv2.putText(left_panel, f"{area}px", (x,y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        
        # Right panel: Segmentation heatmap
        right_panel = cv2.applyColorMap(fg_mask, cv2.COLORMAP_JET)
        for cnt in contours:
            if cv2.contourArea(cnt) > 500:
                cv2.drawContours(right_panel, [cnt], -1, (0,255,255), 2)  # Yellow contours
        
        # Combine panels
        comparison[:, :w] = left_panel
        comparison[:, w:] = right_panel
        
        # Add titles and metrics
        motion_pixels = np.sum(fg_mask > 0)
        motion_percent = (motion_pixels / (h*w)) * 100
        
        cv2.putText(comparison, "Original (Detection)", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(comparison, "Segmentation (Analysis)", (w+10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(comparison, f"Frame: {frame_count} | Motion: {motion_percent:.1f}%", 
                   (w//2, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        # Add this right before cv2.imshow() in your existing code:

        # 1. Create a clean metrics panel
        metrics_panel = np.zeros((100, w*2, 3), dtype=np.uint8)  # 100px tall bar

        # 2. Calculate advanced metrics
        moving_objects = len([cnt for cnt in contours if cv2.contourArea(cnt) > 500])
        max_area = max([cv2.contourArea(cnt) for cnt in contours], default=0)

        # 3. Format professional-looking text
        cv2.putText(metrics_panel, "Motion Analysis Report", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(metrics_panel, f"Objects: {moving_objects} | Largest: {max_area}px | Coverage: {motion_percent:.1f}%", 
           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.putText(metrics_panel, f"Frame: {frame_count} | Time: {datetime.now().strftime('%H:%M:%S')}", 
           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # 4. Combine with main visualization
        final_output = np.vstack((comparison, metrics_panel))

        # 5. Display
        cv2.imshow("Advanced Motion Segmentation", final_output)
        # Display
        cv2.imshow("Motion Segmentation Analysis", comparison)
        
        # Save key frames (every 2 seconds if 30fps)
        if frame_count % 60 == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"{OUTPUT_DIR}/frame_{timestamp}.png", comparison)
        
        frame_count += 1
        
        # Exit on ESC
        if cv2.waitKey(30) == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Output saved to {OUTPUT_DIR} folder")

if __name__ == "__main__":
    main()