import cv2
import dlib
import numpy as np
from collections import deque

# Initialize face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../data/shape_predictor_68_face_landmarks.dat")

# Start webcam
cap = cv2.VideoCapture(0)

# Create windows
cv2.namedWindow('Eye Tracking')
cv2.namedWindow('Left Eye View')
cv2.namedWindow('Right Eye View')

# Constants for eye display size
EYE_DISPLAY_WIDTH = 200
EYE_DISPLAY_HEIGHT = 100

# History for smoothing - reduce to 5 for more responsiveness
left_pupil_history = deque(maxlen=4)  # Reduced from 7 to 4
right_pupil_history = deque(maxlen=4)

def extract_eye(gray, eye_points):
    # Extract eye region from facial landmarks
    region = np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in eye_points])
    
    # Find bounding box of eye region
    min_x = np.min(region[:, 0])
    max_x = np.max(region[:, 0])
    min_y = np.min(region[:, 1])
    max_y = np.max(region[:, 1])
    
    # Add some padding
    padding = 5
    min_x = max(0, min_x - padding)
    max_x = min(gray.shape[1], max_x + padding)
    min_y = max(0, min_y - padding)
    max_y = min(gray.shape[0], max_y + padding)
    
    # Extract eye from image
    eye = gray[min_y:max_y, min_x:max_x]
    
    # Return eye image and eye region position
    return eye, (min_x, min_y)

def detect_pupil(eye_img):
    """Enhanced pupil detection specifically for zoomed eye images"""
    if eye_img is None or eye_img.size == 0:
        return None, None, None
        
    # Convert to grayscale if the image is color
    if len(eye_img.shape) == 3:
        eye_gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
    else:
        eye_gray = eye_img
    
    # Create a debug image to visualize processing steps
    debug_img = cv2.cvtColor(eye_gray, cv2.COLOR_GRAY2BGR)
    
    # Apply histogram equalization to enhance contrast
    eye_gray = cv2.equalizeHist(eye_gray)
    
    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(eye_gray, (5, 5), 0)  # Smaller kernel for faster processing
    
    # Use gradient-based approach to find dark circular regions
    sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobelx, sobely)
    
    # Normalize and convert to uint8
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Apply binary threshold 
    _, thresh_gradient = cv2.threshold(magnitude, 40, 255, cv2.THRESH_BINARY)
    
    # Apply circle Hough transform to find pupils as circles
    try:
        circles = cv2.HoughCircles(
            blur, 
            cv2.HOUGH_GRADIENT, 
            dp=1.2,
            minDist=10,
            param1=50,
            param2=30,
            minRadius=5,
            maxRadius=30
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            
            # Find the darkest circle (pupil should be the darkest)
            best_circle = None
            best_darkness = 255
            
            for i in circles[0, :]:
                # Create a mask for this circle
                mask = np.zeros_like(eye_gray)
                cv2.circle(mask, (i[0], i[1]), i[2]//2, 255, -1)
                
                # Calculate average intensity in this circle
                mean_val = np.mean(eye_gray[mask > 0])
                
                # If this is the darkest circle so far, remember it
                if mean_val < best_darkness:
                    best_darkness = mean_val
                    best_circle = i
            
            if best_circle is not None:
                # Draw the best circle on the debug image
                cv2.circle(debug_img, (best_circle[0], best_circle[1]), 
                          best_circle[2], (0, 255, 0), 2)
                
                return best_circle[0], best_circle[1], debug_img
    except:
        pass
    
    # Fallback method if Hough transform fails
    
    # Method 1: Use adaptive thresholding
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 5)
    
    # Apply morphological operations to clean up the image
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # If no contours found, try a different method
    if not contours:
        # Method 2: Use fixed thresholding as fallback
        _, thresh = cv2.threshold(blur, 40, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Process contours if they exist
    if contours:
        # Sort by area, largest first
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Try to find the pupil contour
        best_center = None
        best_score = float('-inf')
        
        for cnt in contours[:5]:  # Check only the 5 largest contours
            area = cv2.contourArea(cnt)
            
            # Size filter: exclude too large or too small contours
            if area < (eye_gray.shape[0] * eye_gray.shape[1] * 0.4) and area > 20:
                # Check circularity
                perimeter = cv2.arcLength(cnt, True)
                circularity = 0
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                # Calculate center
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Check darkness (pupils are darker)
                    mask = np.zeros_like(eye_gray)
                    cv2.drawContours(mask, [cnt], 0, 255, -1)
                    mean_val = np.mean(eye_gray[mask > 0])
                    darkness = 1.0 - (mean_val / 255.0)
                    
                    # Position score: prefer centers closer to the middle
                    eye_center_x = eye_gray.shape[1] // 2
                    eye_center_y = eye_gray.shape[0] // 2
                    distance_from_center = np.sqrt((cx - eye_center_x)**2 + (cy - eye_center_y)**2)
                    position_score = 1.0 - (distance_from_center / (eye_gray.shape[1] / 2))
                    
                    # Combined score, weighting factors can be adjusted
                    score = (circularity * 0.3 +     # Prefer circular shapes
                             darkness * 0.5 +         # Prefer dark regions (pupils)
                             position_score * 0.2)    # Prefer center positions
                    
                    # Check if this is the best contour so far
                    if score > best_score:
                        best_score = score
                        best_center = (cx, cy)
                        # Draw the best contour on the debug image
                        cv2.drawContours(debug_img, [cnt], 0, (0, 255, 0), 1)
        
        if best_center:
            return best_center[0], best_center[1], debug_img
    
    # If all else fails, try to find the darkest spot
    # Apply a minimum filter to find dark spots
    min_filtered = cv2.erode(blur, np.ones((5, 5), np.uint8))
    min_val, _, min_loc, _ = cv2.minMaxLoc(min_filtered)
    
    # Check if the dark spot is near the center of the eye
    eye_center_x = eye_gray.shape[1] // 2
    eye_center_y = eye_gray.shape[0] // 2
    dist_from_center = np.sqrt((min_loc[0]-eye_center_x)**2 + (min_loc[1]-eye_center_y)**2)
    
    # Only return the dark spot if it's reasonably close to the center
    if dist_from_center < eye_gray.shape[1]/2:
        return min_loc[0], min_loc[1], debug_img
    else:
        # Final fallback: use the center of the eye
        return eye_center_x, eye_center_y, debug_img

def smooth_position(current_pos, history):
    """Apply smoothing to reduce jitter while maintaining responsiveness"""
    if current_pos is None:
        return None
    
    # Add current position to history
    history.append(current_pos)
    
    # If not enough history, just return current position
    if len(history) < 2:
        return current_pos
    
    # Calculate smoothed position with modified weighting
    x_sum = 0
    y_sum = 0
    weight_sum = 0
    
    # Exponential weighting for more responsiveness
    for i, (x, y) in enumerate(history):
        # Current position gets much higher weight
        if i == len(history) - 1:
            weight = len(history)  # Current position gets higher weight
        else:
            weight = i + 1
            
        x_sum += x * weight
        y_sum += y * weight
        weight_sum += weight
    
    if weight_sum > 0:
        return (int(x_sum / weight_sum), int(y_sum / weight_sum))
    else:
        return current_pos

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break
        
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = detector(gray)
    
    # Create blank eye views to display if no eyes are detected
    left_eye_display = np.zeros((EYE_DISPLAY_HEIGHT, EYE_DISPLAY_WIDTH, 3), dtype=np.uint8)
    right_eye_display = np.zeros((EYE_DISPLAY_HEIGHT, EYE_DISPLAY_WIDTH, 3), dtype=np.uint8)
    
    for face in faces:
        # Get facial landmarks
        landmarks = predictor(gray, face)
        
        # Left eye indices (points 36-41) - but these are flipped in the camera
        # This is actually the RIGHT eye in the image (left side of screen)
        right_eye_points = range(36, 42)
        
        # Right eye indices (points 42-47) - but these are flipped in the camera
        # This is actually the LEFT eye in the image (right side of screen)
        left_eye_points = range(42, 48)
        
        # Extract eye regions
        left_eye, left_eye_pos = extract_eye(gray, left_eye_points)
        right_eye, right_eye_pos = extract_eye(gray, right_eye_points)
        
        # Process left eye (which is on the right side of the flipped image)
        if left_eye.size > 0:
            # Get color version of eye for display
            left_eye_color = frame[left_eye_pos[1]:left_eye_pos[1]+left_eye.shape[0], 
                                 left_eye_pos[0]:left_eye_pos[0]+left_eye.shape[1]]
            
            if left_eye_color.size > 0:
                # Resize to constant size
                left_eye_display = cv2.resize(left_eye_color, (EYE_DISPLAY_WIDTH, EYE_DISPLAY_HEIGHT))
                
                # Convert the zoomed eye to grayscale for pupil detection
                left_eye_display_gray = cv2.cvtColor(left_eye_display, cv2.COLOR_BGR2GRAY)
                
                # Detect pupil on the zoomed eye image
                left_pupil_x, left_pupil_y, _ = detect_pupil(left_eye_display_gray)
                
                # Draw pupil on zoomed view if detected
                if left_pupil_x is not None and left_pupil_y is not None:
                    # Apply smoothing
                    smooth_left_pupil = smooth_position((left_pupil_x, left_pupil_y), left_pupil_history)
                    
                    if smooth_left_pupil:
                        # Use anti-aliased circle for smoother appearance
                        cv2.circle(left_eye_display, smooth_left_pupil, 5, (0, 255, 0), -1, cv2.LINE_AA)
                        
                        # Convert back to original frame coordinates for main display
                        original_x = int(smooth_left_pupil[0] * left_eye.shape[1] / EYE_DISPLAY_WIDTH)
                        original_y = int(smooth_left_pupil[1] * left_eye.shape[0] / EYE_DISPLAY_HEIGHT)
                        frame_x = left_eye_pos[0] + original_x
                        frame_y = left_eye_pos[1] + original_y
                        
                        # Use anti-aliased circle for smoother appearance
                        cv2.circle(frame, (frame_x, frame_y), 3, (0, 255, 0), -1, cv2.LINE_AA)
        
        # Process right eye (which is on the left side of the flipped image)
        if right_eye.size > 0:
            # Get color version of eye for display
            right_eye_color = frame[right_eye_pos[1]:right_eye_pos[1]+right_eye.shape[0], 
                                  right_eye_pos[0]:right_eye_pos[0]+right_eye.shape[1]]
            
            if right_eye_color.size > 0:
                # Resize to constant size
                right_eye_display = cv2.resize(right_eye_color, (EYE_DISPLAY_WIDTH, EYE_DISPLAY_HEIGHT))
                
                # Convert the zoomed eye to grayscale for pupil detection
                right_eye_display_gray = cv2.cvtColor(right_eye_display, cv2.COLOR_BGR2GRAY)
                
                # Detect pupil on the zoomed eye image
                right_pupil_x, right_pupil_y, _ = detect_pupil(right_eye_display_gray)
                
                # Draw pupil on zoomed view if detected
                if right_pupil_x is not None and right_pupil_y is not None:
                    # Apply smoothing
                    smooth_right_pupil = smooth_position((right_pupil_x, right_pupil_y), right_pupil_history)
                    
                    if smooth_right_pupil:
                        # Use anti-aliased circle for smoother appearance
                        cv2.circle(right_eye_display, smooth_right_pupil, 5, (0, 255, 0), -1, cv2.LINE_AA)
                        
                        # Convert back to original frame coordinates for main display
                        original_x = int(smooth_right_pupil[0] * right_eye.shape[1] / EYE_DISPLAY_WIDTH)
                        original_y = int(smooth_right_pupil[1] * right_eye.shape[0] / EYE_DISPLAY_HEIGHT)
                        frame_x = right_eye_pos[0] + original_x
                        frame_y = right_eye_pos[1] + original_y
                        
                        # Use anti-aliased circle for smoother appearance
                        cv2.circle(frame, (frame_x, frame_y), 3, (0, 255, 0), -1, cv2.LINE_AA)
        
        # Draw eye outlines for reference (using smooth circles)
        for point in left_eye_points:
            pos = (landmarks.part(point).x, landmarks.part(point).y)
            cv2.circle(frame, pos, 2, (0, 0, 255), -1, cv2.LINE_AA)
            
        for point in right_eye_points:
            pos = (landmarks.part(point).x, landmarks.part(point).y)
            cv2.circle(frame, pos, 2, (0, 0, 255), -1, cv2.LINE_AA)
    
    # Display the frames
    cv2.imshow('Eye Tracking', frame)
    cv2.imshow('Left Eye View', left_eye_display)  # This shows the right eye in the mirrored image
    cv2.imshow('Right Eye View', right_eye_display)  # This shows the left eye in the mirrored image
    
    # Position the windows
    cv2.moveWindow('Left Eye View', 20, frame.shape[0] + 40)
    cv2.moveWindow('Right Eye View', 20 + EYE_DISPLAY_WIDTH + 20, frame.shape[0] + 40)
    
    # Quit if 'q' is pressed or main window is closed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or cv2.getWindowProperty('Eye Tracking', cv2.WND_PROP_VISIBLE) < 1:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
