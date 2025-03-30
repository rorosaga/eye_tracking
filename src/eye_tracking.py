import cv2
import dlib
import numpy as np
from collections import deque
import time  # For tracking timeout

# Initialize face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../data/shape_predictor_68_face_landmarks.dat")

# Start webcam
cap = cv2.VideoCapture(0)

# Create single window
cv2.namedWindow('Eye Tracking')

# Constants for eye display size
EYE_DISPLAY_WIDTH = 150
EYE_DISPLAY_HEIGHT = 75

# History for smoothing - using shorter history with more weight on new positions
left_pupil_history = deque(maxlen=3)
right_pupil_history = deque(maxlen=3)

# Time tracking for pupil detection
left_pupil_last_updated = 0
right_pupil_last_updated = 0
RESET_TIMEOUT = 0.5  # Reset tracking if no update for 0.5 seconds

# Stuck detection variables
left_last_pos = None
right_last_pos = None
left_stuck_since = 0
right_stuck_since = 0
STUCK_TIMEOUT = 2.0  # Reset tracking if position hasn't changed for 2 seconds
STUCK_THRESHOLD = 3  # How many pixels of movement to consider not stuck

# Blink detection threshold - ratio of eye height to width
EYE_AR_THRESHOLD = 0.2  # If the eye aspect ratio falls below this, eye is considered closed

def eye_aspect_ratio(eye_points_array):
    """Calculate the eye aspect ratio to detect blinks"""
    # Compute the euclidean distances between the two sets of vertical eye landmarks
    A = np.linalg.norm(eye_points_array[1] - eye_points_array[5])
    B = np.linalg.norm(eye_points_array[2] - eye_points_array[4])
    
    # Compute the euclidean distance between the horizontal eye landmarks
    C = np.linalg.norm(eye_points_array[0] - eye_points_array[3])
    
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    
    return ear

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
    """Improved pupil detection algorithm"""
    if eye_img is None or eye_img.size == 0:
        return None, None
        
    # Convert to grayscale if the image is color
    if len(eye_img.shape) == 3:
        eye_gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
    else:
        eye_gray = eye_img
    
    # Apply histogram equalization to enhance contrast
    eye_gray = cv2.equalizeHist(eye_gray)
    
    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(eye_gray, (5, 5), 0)
    
    # Get eye dimensions
    height, width = blur.shape
    eye_center_x = width // 2
    eye_center_y = height // 2
    
    # IMPROVED METHOD: Combine multiple approaches for better accuracy
    
    # 1. Find dark regions (pupils are typically dark)
    _, dark_regions = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY_INV)  # Slightly more sensitive threshold
    kernel = np.ones((3, 3), np.uint8)
    dark_regions = cv2.morphologyEx(dark_regions, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # 2. Apply adaptive thresholding for better edge detection
    adaptive_thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                           cv2.THRESH_BINARY_INV, 11, 2)
    
    # 3. Combine the methods - only keep areas that are detected by both methods
    combined = cv2.bitwise_and(dark_regions, adaptive_thresh)
    
    # 4. Find contours in the combined result
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If we found contours, process them
    if contours:
        # Filter and score contours
        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            # Size filter
            if area < (height * width * 0.5) and area > 20:  # Lower minimum area for smaller pupils
                # Calculate center of contour
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Calculate circularity
                    perimeter = cv2.arcLength(cnt, True)
                    circularity = 0
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    # Calculate darkness
                    mask = np.zeros_like(eye_gray)
                    cv2.drawContours(mask, [cnt], 0, 255, -1)
                    mean_val = np.mean(eye_gray[mask > 0])
                    darkness = 1.0 - (mean_val / 255.0)
                    
                    # Calculate distance from center
                    distance = np.sqrt((cx - eye_center_x)**2 + (cy - eye_center_y)**2)
                    center_score = 1.0 - min(1.0, distance / (width * 0.5))
                    
                    # Calculate overall score with higher center weighting
                    score = (circularity * 0.2 +  # Roundness
                             darkness * 0.3 +     # Darkness
                             center_score * 0.5)  # Center position (higher weight)
                    
                    valid_contours.append((cx, cy, score))
        
        # If we have valid contours, use the one with the highest score
        if valid_contours:
            best_contour = max(valid_contours, key=lambda x: x[2])
            return best_contour[0], best_contour[1]
    
    # If contour method failed, try circle detection
    try:
        circles = cv2.HoughCircles(
            blur, 
            cv2.HOUGH_GRADIENT, 
            dp=1.2,
            minDist=width//4,  # Only look for one circle in the eye
            param1=30,  # Lower for better detection
            param2=15,  # More sensitive to find more circles
            minRadius=max(2, width//20),  # Smaller minimum radius
            maxRadius=max(10, width//5)
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            
            # Find the darkest/best circle
            best_circle = None
            best_score = float('-inf')
            
            for i in circles[0, :]:
                # Create mask for this circle
                mask = np.zeros_like(eye_gray)
                cv2.circle(mask, (i[0], i[1]), i[2]//2, 255, -1)
                
                # Calculate darkness
                mean_val = np.mean(eye_gray[mask > 0])
                darkness = 1.0 - (mean_val / 255.0)
                
                # Calculate center score
                dist_from_center = np.sqrt((i[0] - eye_center_x)**2 + (i[1] - eye_center_y)**2)
                center_score = 1.0 - min(1.0, dist_from_center / (width * 0.5))
                
                # Calculate overall score
                score = darkness * 0.4 + center_score * 0.6
                
                if score > best_score:
                    best_score = score
                    best_circle = i
            
            if best_circle is not None:
                return best_circle[0], best_circle[1]
    except:
        pass
        
    # Final fallback - find the darkest region near the center
    min_val, _, min_loc, _ = cv2.minMaxLoc(cv2.GaussianBlur(255 - blur, (11, 11), 0))
    
    # Check if it's close enough to center
    dist_to_center = np.sqrt((min_loc[0] - eye_center_x)**2 + (min_loc[1] - eye_center_y)**2)
    if dist_to_center < width * 0.5:  # Expanded search radius to 50% of eye width
        return min_loc[0], min_loc[1]
    else:
        # If all else fails, return None to force reset
        return None, None

def smooth_position(current_pos, history, last_updated_time, last_pos, stuck_since):
    """Apply advanced smoothing with timeout reset and stuck detection"""
    current_time = time.time()
    
    # If position is None, we couldn't detect the pupil
    if current_pos is None:
        # If we haven't detected a pupil for a while, clear history to avoid sticking
        if current_time - last_updated_time > RESET_TIMEOUT and len(history) > 0:
            history.clear()
            last_pos = None
            stuck_since = 0
        return None, last_updated_time, last_pos, stuck_since
    
    # We have a valid current position, update the last detected time
    last_updated_time = current_time
    
    # Check if current position is drastically different from history
    if history and len(history) > 0:
        prev_pos = history[-1]
        dist = np.sqrt((current_pos[0] - prev_pos[0])**2 + (current_pos[1] - prev_pos[1])**2)
        
        # If position jumped too far and we've had stable tracking
        if dist > 15 and len(history) >= 2:  # Reduced threshold
            # If we've recently had valid tracking, be more cautious
            time_diff = current_time - last_updated_time
            if time_diff < 0.2:  # Recent valid tracking
                # Don't add this position to history as it's likely wrong
                return prev_pos, last_updated_time, last_pos, stuck_since
            else:
                # It's been a while, accept the new position and reset history
                history.clear()
                history.append(current_pos)
                return current_pos, last_updated_time, current_pos, current_time
    
    # Add current position to history
    history.append(current_pos)
    
    # If we don't have enough history, just return current position
    if len(history) < 2:
        return current_pos, last_updated_time, current_pos, current_time
    
    # Calculate weighted average with higher emphasis on recent positions
    total_x = 0
    total_y = 0
    total_weight = 0
    
    # Exponential weighting - much higher weight for recent positions
    weights = [1, 3, 8]  # Increasing weight for current position
    weights = weights[-len(history):]  # Adjust to actual history length
    
    for i, (x, y) in enumerate(history):
        weight = weights[i]
        total_x += x * weight
        total_y += y * weight
        total_weight += weight
    
    smoothed_x = int(total_x / total_weight)
    smoothed_y = int(total_y / total_weight)
    smoothed_pos = (smoothed_x, smoothed_y)
    
    # Check for stuck position (hasn't moved enough for 2 seconds)
    if last_pos is not None:
        dist = np.sqrt((smoothed_x - last_pos[0])**2 + (smoothed_y - last_pos[1])**2)
        
        if dist < STUCK_THRESHOLD:
            # Position hasn't changed significantly
            if stuck_since == 0:
                # First time we noticed it's stuck
                stuck_since = current_time
            elif current_time - stuck_since > STUCK_TIMEOUT:
                # Been stuck for too long, force reset
                history.clear()
                stuck_since = 0
                last_pos = None
                return None, last_updated_time, None, 0
        else:
            # Position has changed, reset stuck timer
            stuck_since = 0
    
    # Update last position
    last_pos = smoothed_pos
    
    return smoothed_pos, last_updated_time, last_pos, stuck_since

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break
    
    # Get frame dimensions
    frame_height, frame_width = frame.shape[:2]
    
    # Create a copy of the frame for displaying eye zones
    display_frame = frame.copy()
        
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = detector(gray)
    
    # If no faces detected for a while, reset pupil tracking
    if not faces:
        current_time = time.time()
        if current_time - left_pupil_last_updated > RESET_TIMEOUT:
            left_pupil_history.clear()
            left_last_pos = None
            left_stuck_since = 0
        if current_time - right_pupil_last_updated > RESET_TIMEOUT:
            right_pupil_history.clear()
            right_last_pos = None
            right_stuck_since = 0
    
    # Create blank eye views
    left_eye_display = np.zeros((EYE_DISPLAY_HEIGHT, EYE_DISPLAY_WIDTH, 3), dtype=np.uint8)
    right_eye_display = np.zeros((EYE_DISPLAY_HEIGHT, EYE_DISPLAY_WIDTH, 3), dtype=np.uint8)
    
    # Eye blink status - default to eyes closed
    left_eye_open = False
    right_eye_open = False
    
    for face in faces:
        # Get facial landmarks
        landmarks = predictor(gray, face)
        
        # Left eye indices (points 36-41) - but these are flipped in the camera
        # This is actually the RIGHT eye in the image (left side of screen)
        right_eye_points = range(36, 42)
        
        # Right eye indices (points 42-47) - but these are flipped in the camera
        # This is actually the LEFT eye in the image (right side of screen)
        left_eye_points = range(42, 48)
        
        # Check if eyes are open using aspect ratio
        right_eye_landmarks = np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in right_eye_points])
        left_eye_landmarks = np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in left_eye_points])
        
        right_ear = eye_aspect_ratio(right_eye_landmarks)
        left_ear = eye_aspect_ratio(left_eye_landmarks)
        
        # Determine if eyes are open
        right_eye_open = right_ear > EYE_AR_THRESHOLD
        left_eye_open = left_ear > EYE_AR_THRESHOLD
        
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
                
                # Only detect pupil if eye is open
                if left_eye_open:
                    # Convert the zoomed eye to grayscale for pupil detection
                    left_eye_display_gray = cv2.cvtColor(left_eye_display, cv2.COLOR_BGR2GRAY)
                    
                    # Detect pupil on the zoomed eye image
                    left_pupil_x, left_pupil_y = detect_pupil(left_eye_display_gray)
                    
                    # Draw pupil on zoomed view if detected
                    if left_pupil_x is not None and left_pupil_y is not None:
                        # Apply smoothing with timeout tracking and stuck detection
                        smooth_left_pupil, left_pupil_last_updated, left_last_pos, left_stuck_since = smooth_position(
                            (left_pupil_x, left_pupil_y), 
                            left_pupil_history,
                            left_pupil_last_updated,
                            left_last_pos,
                            left_stuck_since
                        )
                        
                        if smooth_left_pupil:
                            # Draw on zoomed view
                            cv2.circle(left_eye_display, smooth_left_pupil, 5, (0, 255, 0), -1, cv2.LINE_AA)
                            
                            # Convert back to original frame coordinates for main display
                            original_x = int(smooth_left_pupil[0] * left_eye.shape[1] / EYE_DISPLAY_WIDTH)
                            original_y = int(smooth_left_pupil[1] * left_eye.shape[0] / EYE_DISPLAY_HEIGHT)
                            frame_x = left_eye_pos[0] + original_x
                            frame_y = left_eye_pos[1] + original_y
                            
                            # Draw on main frame
                            cv2.circle(display_frame, (frame_x, frame_y), 4, (0, 255, 0), -1, cv2.LINE_AA)
                else:
                    # Eye is closed, clear history for fresh tracking when reopened
                    left_pupil_history.clear()
                    left_last_pos = None
                    left_stuck_since = 0
        
        # Process right eye (which is on the left side of the flipped image)
        if right_eye.size > 0:
            # Get color version of eye for display
            right_eye_color = frame[right_eye_pos[1]:right_eye_pos[1]+right_eye.shape[0], 
                                  right_eye_pos[0]:right_eye_pos[0]+right_eye.shape[1]]
            
            if right_eye_color.size > 0:
                # Resize to constant size
                right_eye_display = cv2.resize(right_eye_color, (EYE_DISPLAY_WIDTH, EYE_DISPLAY_HEIGHT))
                
                # Only detect pupil if eye is open
                if right_eye_open:
                    # Convert the zoomed eye to grayscale for pupil detection
                    right_eye_display_gray = cv2.cvtColor(right_eye_display, cv2.COLOR_BGR2GRAY)
                    
                    # Detect pupil on the zoomed eye image
                    right_pupil_x, right_pupil_y = detect_pupil(right_eye_display_gray)
                    
                    # Draw pupil on zoomed view if detected
                    if right_pupil_x is not None and right_pupil_y is not None:
                        # Apply smoothing with timeout tracking and stuck detection
                        smooth_right_pupil, right_pupil_last_updated, right_last_pos, right_stuck_since = smooth_position(
                            (right_pupil_x, right_pupil_y), 
                            right_pupil_history,
                            right_pupil_last_updated,
                            right_last_pos,
                            right_stuck_since
                        )
                        
                        if smooth_right_pupil:
                            # Draw on zoomed view
                            cv2.circle(right_eye_display, smooth_right_pupil, 5, (0, 255, 0), -1, cv2.LINE_AA)
                            
                            # Convert back to original frame coordinates for main display
                            original_x = int(smooth_right_pupil[0] * right_eye.shape[1] / EYE_DISPLAY_WIDTH)
                            original_y = int(smooth_right_pupil[1] * right_eye.shape[0] / EYE_DISPLAY_HEIGHT)
                            frame_x = right_eye_pos[0] + original_x
                            frame_y = right_eye_pos[1] + original_y
                            
                            # Draw on main frame
                            cv2.circle(display_frame, (frame_x, frame_y), 4, (0, 255, 0), -1, cv2.LINE_AA)
                else:
                    # Eye is closed, clear history for fresh tracking when reopened
                    right_pupil_history.clear()
                    right_last_pos = None
                    right_stuck_since = 0
        
        # Draw thin green lines around the eyes
        # Left eye outline
        left_eye_points_array = np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in left_eye_points])
        cv2.polylines(display_frame, [left_eye_points_array], True, (0, 255, 0), 1, cv2.LINE_AA)
        
        # Right eye outline
        right_eye_points_array = np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in right_eye_points])
        cv2.polylines(display_frame, [right_eye_points_array], True, (0, 255, 0), 1, cv2.LINE_AA)
    
    # Overlay eye views on top of the main video
    # Calculate positions for eye views
    margin = 20
    
    # Left eye overlay position (top-left corner)
    left_x = margin
    left_y = margin
    
    # Right eye overlay position (top-right corner)
    right_x = frame_width - EYE_DISPLAY_WIDTH - margin
    right_y = margin
    
    # Create a semi-transparent black background for the eye views
    overlay = display_frame.copy()
    
    # Draw background rectangles for eye views
    cv2.rectangle(overlay, (left_x-5, left_y-5), 
                 (left_x+EYE_DISPLAY_WIDTH+5, left_y+EYE_DISPLAY_HEIGHT+25), (0, 0, 0), -1)
    cv2.rectangle(overlay, (right_x-5, right_y-5), 
                 (right_x+EYE_DISPLAY_WIDTH+5, right_y+EYE_DISPLAY_HEIGHT+25), (0, 0, 0), -1)
    
    # Apply transparency
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0, display_frame)
    
    # Overlay the eye views
    display_frame[left_y:left_y+EYE_DISPLAY_HEIGHT, left_x:left_x+EYE_DISPLAY_WIDTH] = left_eye_display
    display_frame[right_y:right_y+EYE_DISPLAY_HEIGHT, right_x:right_x+EYE_DISPLAY_WIDTH] = right_eye_display
    
    # Add green text labels under each eye view
    cv2.putText(display_frame, "Left", (left_x, left_y+EYE_DISPLAY_HEIGHT+20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(display_frame, "Right", (right_x, right_y+EYE_DISPLAY_HEIGHT+20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    
    # Display the frame
    cv2.imshow('Eye Tracking', display_frame)
    
    # Quit if 'q' is pressed or window is closed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or cv2.getWindowProperty('Eye Tracking', cv2.WND_PROP_VISIBLE) < 1:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
