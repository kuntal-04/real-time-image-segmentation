import cv2
import numpy as np
import random

# Initialize background subtractor
back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)

# Dictionary to store object colors and tracking
object_colors = {}
color_options = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (255, 165, 0)   # Orange
]

cap = cv2.VideoCapture(0)  # Use 0 for webcam or video path

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for faster processing
    frame = cv2.resize(frame, (640, 480))
    
    # Apply background subtraction
    fg_mask = back_sub.apply(frame)
    
    # Remove noise and enhance objects
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)
    
    # Find contours of moving objects
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create blank segmentation mask
    segmentation = np.zeros_like(frame)
    
    # Process each detected object
    for i, cnt in enumerate(contours):
        if cv2.contourArea(cnt) < 500:  # Skip small objects
            continue
            
        # Get object ID (using centroid position as simple identifier)
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        obj_id = f"{cx}_{cy}"
        
        # Assign or get color for this object
        if obj_id not in object_colors:
            object_colors[obj_id] = random.choice(color_options)
        color = object_colors[obj_id]
        
        # Draw filled contour on segmentation mask
        cv2.drawContours(segmentation, [cnt], -1, color, -1)
        
        # Draw bounding box with object's color
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
        
        # Label with object ID
        cv2.putText(frame, f"Obj {i+1}", (x,y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Blend original with segmentation
    alpha = 0.5
    blended = cv2.addWeighted(frame, alpha, segmentation, 1-alpha, 0)
    
    # Display results
    cv2.imshow('Original with Detection', frame)
    cv2.imshow('Segmentation Mask', segmentation)
    cv2.imshow('Blended Output', blended)
    
    if cv2.waitKey(30) == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()