import cv2
import numpy as np

# Load the pre-trained classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
        
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # For each face, find the eyes
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # Detect eyes within the face region
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        # For each eye, draw a green dot at its center
        for (ex, ey, ew, eh) in eyes:
            # Calculate center of eye
            eye_center_x = x + ex + ew // 2
            eye_center_y = y + ey + eh // 2
            
            # Draw green dot
            cv2.circle(frame, (eye_center_x, eye_center_y), 3, (0, 255, 0), -1)
    
    # Display the resulting frame
    cv2.imshow('Eye Tracking', frame)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
