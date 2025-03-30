import cv2
import dlib
import numpy as np
from collections import deque
import time  # For tracking timeout
import os  # For screen size detection

# Initialize face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../data/shape_predictor_68_face_landmarks.dat")

# Get screen resolution
try:
    # Try to get screen resolution using cv2
    screen_width = cv2.getWindowByName("").shape[1]  # Default 1920 if fails
    screen_height = cv2.getWindowByName("").shape[0]  # Default 1080 if fails
except:
    # Fallback dimensions if detection fails
    screen_width = 1920
    screen_height = 1080

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

# Store calibration data for proper gaze estimation
calibration_map = None

# Add new calibration function
def calibrate_eye_tracking(detector, predictor):
    # Create a window with a specific name for fullscreen
    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # Use screen dimensions for calibration points
    calibration_points = [
        (int(screen_width * 0.1), int(screen_height * 0.1)),   # Top-left
        (int(screen_width * 0.5), int(screen_height * 0.1)),   # Top-center
        (int(screen_width * 0.9), int(screen_height * 0.1)),   # Top-right
        (int(screen_width * 0.1), int(screen_height * 0.5)),   # Middle-left
        (int(screen_width * 0.5), int(screen_height * 0.5)),   # Center
        (int(screen_width * 0.9), int(screen_height * 0.5)),   # Middle-right
        (int(screen_width * 0.1), int(screen_height * 0.9)),   # Bottom-left
        (int(screen_width * 0.5), int(screen_height * 0.9)),   # Bottom-center
        (int(screen_width * 0.9), int(screen_height * 0.9)),   # Bottom-right
    ]
    
    # Dictionary to store calibration data
    calibration_data = {point: {"left_pupil": [], "right_pupil": []} for point in calibration_points}
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return None, None
    
    # Used to track which calibration point we're on
    current_point_idx = 0
    samples_collected = 0
    max_samples = 15  # Increased number of samples for better accuracy
    recalibrating = False  # Flag to indicate if we're recalibrating a point
    
    # Initialize mapping data structures
    left_pupil_to_screen = []
    right_pupil_to_screen = []
    
    # Get user satisfaction input
    user_satisfied = False
    
    while not user_satisfied:
        while current_point_idx < len(calibration_points):
            # Get current point
            point = calibration_points[current_point_idx]
            
            # Frame processing
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break
                
            # Mirror frame horizontally for more intuitive interaction
            frame = cv2.flip(frame, 1)
            
            # Create a blank calibration image
            calib_img = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
            
            # Draw all calibration points as small circles
            for p in calibration_points:
                color = (100, 100, 100)  # Gray for inactive points
                if p == point:
                    color = (0, 0, 255)  # Red for active point
                cv2.circle(calib_img, p, 15, color, -1)
            
            # Draw current calibration point as a larger circle
            cv2.circle(calib_img, point, 20, (0, 0, 255), -1)
            
            # Process face and detect pupils
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            
            # Calculate frame position to avoid covering the active calibration point
            frame_height, frame_width = frame.shape[:2]
            frame_resized = cv2.resize(frame, (int(frame_width/4), int(frame_height/4)))
            frame_resized_h, frame_resized_w = frame_resized.shape[:2]
            
            # Determine position for webcam preview to avoid active point
            # We'll place the preview in the quadrant opposite to the current point
            padding = 20
            
            if point[0] < screen_width / 2:  # Left side
                preview_x = screen_width - frame_resized_w - padding
            else:  # Right side
                preview_x = padding
                
            if point[1] < screen_height / 2:  # Top half
                preview_y = screen_height - frame_resized_h - padding
            else:  # Bottom half
                preview_y = padding
                
            # Display frame in position that won't overlap with active point
            calib_img[preview_y:preview_y+frame_resized_h, preview_x:preview_x+frame_resized_w] = frame_resized
            
            # Add rectangle around preview
            cv2.rectangle(calib_img, 
                         (preview_x-1, preview_y-1), 
                         (preview_x+frame_resized_w+1, preview_y+frame_resized_h+1), 
                         (255, 255, 255), 1)
            
            # Instruction text
            cv2.putText(calib_img, f"Look at the red circle and press SPACE ({samples_collected}/{max_samples})", 
                       (int(screen_width/2 - 250), int(screen_height - 50)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            if recalibrating:
                cv2.putText(calib_img, "RECALIBRATING THIS POINT", 
                           (int(screen_width/2 - 150), int(screen_height - 100)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            
            # If face detected, try to get pupil positions
            left_pupil_pos = None
            right_pupil_pos = None
            
            if len(faces) > 0:
                landmarks = predictor(gray, faces[0])
                
                # Extract eye regions - now unpack the tuple correctly
                left_eye_data = extract_eye(gray, range(36, 42), landmarks)
                right_eye_data = extract_eye(gray, range(42, 48), landmarks)
                
                # Unpack the tuple returned by extract_eye
                left_eye_img, left_eye_pos = left_eye_data
                right_eye_img, right_eye_pos = right_eye_data
                
                # Detect pupils
                if left_eye_img is not None:
                    left_pupil_pos = detect_pupil(left_eye_img)
                
                if right_eye_img is not None:
                    right_pupil_pos = detect_pupil(right_eye_img)
                
                # Display detected pupil positions
                if left_pupil_pos is not None and right_pupil_pos is not None:
                    # Show pupil positions in a non-overlapping area
                    pupil_text_y = preview_y + frame_resized_h + 30 if preview_y + frame_resized_h + 60 < screen_height else preview_y - 60
                    cv2.putText(calib_img, f"Left pupil: {left_pupil_pos}", 
                              (preview_x, pupil_text_y), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(calib_img, f"Right pupil: {right_pupil_pos}", 
                              (preview_x, pupil_text_y + 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Display the calibration image
            cv2.imshow("Calibration", calib_img)
            
            # Check for key press - space to record data, ESC to cancel
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == 32:  # Space bar
                # Record data when user confirms they're looking at the point
                if left_pupil_pos is not None and right_pupil_pos is not None:
                    calibration_data[point]["left_pupil"].append(left_pupil_pos)
                    calibration_data[point]["right_pupil"].append(right_pupil_pos)
                    samples_collected += 1
                    print(f"Sample {samples_collected} collected for point {current_point_idx+1}")
                    
                    # If we've collected enough samples, move to the next point
                    if samples_collected >= max_samples:
                        # Calculate average pupil position for this calibration point
                        avg_left = np.mean(calibration_data[point]["left_pupil"], axis=0)
                        avg_right = np.mean(calibration_data[point]["right_pupil"], axis=0)
                        
                        # Add mapping data
                        left_pupil_to_screen.append((avg_left, point))
                        right_pupil_to_screen.append((avg_right, point))
                        
                        current_point_idx += 1
                        samples_collected = 0
                        recalibrating = False
            elif key == ord('r'):  # 'r' to recalibrate current point
                recalibrating = True
                samples_collected = 0
                calibration_data[point]["left_pupil"] = []
                calibration_data[point]["right_pupil"] = []
                print(f"Recalibrating point {current_point_idx+1}")
        
        # After all points are calibrated, evaluate the results
        if current_point_idx >= len(calibration_points):
            user_satisfied = evaluate_calibration(cap, left_pupil_to_screen, right_pupil_to_screen, calibration_points)
            
            if not user_satisfied:
                # Reset calibration process
                current_point_idx = 0
                samples_collected = 0
                calibration_data = {point: {"left_pupil": [], "right_pupil": []} for point in calibration_points}
                left_pupil_to_screen = []
                right_pupil_to_screen = []
                print("Restarting calibration process...")
    
    # Close the calibration window
    cv2.destroyWindow("Calibration")
    cap.release()
    
    return left_pupil_to_screen, right_pupil_to_screen

def evaluate_calibration(cap, left_pupil_to_screen, right_pupil_to_screen, calibration_points):
    """
    Evaluate the calibration results and ask user if they are satisfied.
    
    Args:
        cap: Video capture object
        left_pupil_to_screen: Mapping data for left pupil
        right_pupil_to_screen: Mapping data for right pupil
        calibration_points: List of calibration points
    
    Returns:
        bool: True if user is satisfied, False otherwise
    """
    cv2.namedWindow("Evaluation", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Evaluation", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # Increase number of test points for better evaluation
    random_point_indices = np.random.choice(len(calibration_points), 
                                            size=min(7, len(calibration_points)), 
                                            replace=False)
    test_points = [calibration_points[i] for i in random_point_indices]
    
    success_count = 0
    test_count = 0
    
    # Lower threshold for success to make calibration easier
    success_threshold_percentage = 0.08  # 8% of screen dimension instead of 10%
    
    for point in test_points:
        # Create a blank image
        eval_img = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
        
        # Draw the test point
        cv2.circle(eval_img, point, 30, (0, 255, 0), -1)
        
        cv2.putText(eval_img, "Look at the green circle", 
                   (int(screen_width/2 - 200), int(screen_height - 100)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(eval_img, "Press SPACE once you're looking at it", 
                   (int(screen_width/2 - 250), int(screen_height - 50)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        looking_at_point = False
        start_time = time.time()
        
        while time.time() - start_time < 10:  # 10 second timeout
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            
            # Process face and detect pupils
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            
            # Calculate frame position to avoid covering the test point
            frame_height, frame_width = frame.shape[:2]
            frame_resized = cv2.resize(frame, (int(frame_width/4), int(frame_height/4)))
            frame_resized_h, frame_resized_w = frame_resized.shape[:2]
            
            # Determine position for webcam preview to avoid test point
            padding = 20
            
            if point[0] < screen_width / 2:  # Left side
                preview_x = screen_width - frame_resized_w - padding
            else:  # Right side
                preview_x = padding
                
            if point[1] < screen_height / 2:  # Top half
                preview_y = screen_height - frame_resized_h - padding
            else:  # Bottom half
                preview_y = padding
                
            # Display frame in position that won't overlap with test point
            eval_img[preview_y:preview_y+frame_resized_h, preview_x:preview_x+frame_resized_w] = frame_resized
            
            # Add rectangle around preview
            cv2.rectangle(eval_img, 
                        (preview_x-1, preview_y-1), 
                        (preview_x+frame_resized_w+1, preview_y+frame_resized_h+1), 
                        (255, 255, 255), 1)
            
            # Display the evaluation image
            cv2.imshow("Evaluation", eval_img)
            
            # Calculate gaze position if face detected
            gaze_point = None
            if len(faces) > 0:
                landmarks = predictor(gray, faces[0])
                
                # Extract eye regions
                left_eye_data = extract_eye(gray, range(36, 42), landmarks)
                right_eye_data = extract_eye(gray, range(42, 48), landmarks)
                
                # Unpack the tuple returned by extract_eye
                left_eye_img, left_eye_pos = left_eye_data
                right_eye_img, right_eye_pos = right_eye_data
                
                # Detect pupils
                left_pupil_pos = detect_pupil(left_eye_img) if left_eye_img is not None else None
                right_pupil_pos = detect_pupil(right_eye_img) if right_eye_img is not None else None
                
                if left_pupil_pos is not None and right_pupil_pos is not None:
                    # Use the calibration data to estimate gaze position
                    gaze_point = estimate_gaze_point(left_pupil_pos, right_pupil_pos, 
                                                    left_pupil_to_screen, right_pupil_to_screen)
                    
                    if gaze_point is not None:
                        # Draw estimated gaze point
                        cv2.circle(eval_img, (int(gaze_point[0]), int(gaze_point[1])), 10, (255, 0, 0), -1)
                        
                        # Calculate distance between gaze and test point
                        distance = np.sqrt((gaze_point[0] - point[0])**2 + (gaze_point[1] - point[1])**2)
                        
                        # Show distance in non-overlapping area
                        distance_text_y = preview_y + frame_resized_h + 30 if preview_y + frame_resized_h + 30 < screen_height else preview_y - 30
                        cv2.putText(eval_img, f"Distance: {int(distance)}px", 
                                  (preview_x, distance_text_y), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                return False
            elif key == 32:  # Space bar - user confirms they're looking at the point
                test_count += 1
                
                if gaze_point is not None:
                    distance = np.sqrt((gaze_point[0] - point[0])**2 + (gaze_point[1] - point[1])**2)
                    
                    # If gaze is within a threshold of the point, count as success
                    threshold = min(screen_width, screen_height) * success_threshold_percentage
                    if distance < threshold:
                        success_count += 1
                        cv2.putText(eval_img, "SUCCESS!", (int(screen_width/2 - 100), int(screen_height/2)), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                    else:
                        cv2.putText(eval_img, "MISS", (int(screen_width/2 - 75), int(screen_height/2)), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                
                # Show result for 2 seconds
                cv2.imshow("Evaluation", eval_img)
                cv2.waitKey(2000)
                break
    
    # Show final results
    result_img = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    accuracy = (success_count / test_count * 100) if test_count > 0 else 0
    
    cv2.putText(result_img, f"Calibration Accuracy: {accuracy:.1f}%", 
               (int(screen_width/2 - 300), int(screen_height/2 - 50)), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    cv2.putText(result_img, "Are you satisfied with the calibration?", 
               (int(screen_width/2 - 350), int(screen_height/2 + 50)), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    cv2.putText(result_img, "Y - Yes, continue | N - No, recalibrate", 
               (int(screen_width/2 - 350), int(screen_height/2 + 150)), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    cv2.imshow("Evaluation", result_img)
    
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('y'):
            cv2.destroyWindow("Evaluation")
            return True
        elif key == ord('n'):
            cv2.destroyWindow("Evaluation")
            return False
    
    return False

def estimate_gaze_point(left_pupil_pos, right_pupil_pos, left_pupil_to_screen, right_pupil_to_screen):
    """
    Estimate gaze point based on pupil positions and calibration data.
    Uses an improved weighted average with emphasis on closest matches.
    
    Args:
        left_pupil_pos: Current left pupil position
        right_pupil_pos: Current right pupil position
        left_pupil_to_screen: Mapping data for left pupil
        right_pupil_to_screen: Mapping data for right pupil
    
    Returns:
        tuple: Estimated (x, y) coordinates on screen
    """
    if not left_pupil_to_screen or not right_pupil_to_screen:
        return None
    
    # Calculate weighted average position from both eyes
    left_weights = []
    left_points = []
    right_weights = []
    right_points = []
    
    # Process left eye data
    for pupil_pos, screen_pos in left_pupil_to_screen:
        # Calculate distance (similarity) between current pupil and calibrated pupil
        distance = np.sqrt(np.sum((np.array(left_pupil_pos) - np.array(pupil_pos))**2))
        if distance < 0.001:  # Avoid division by zero
            distance = 0.001
        # Use a squared inverse distance for stronger weighting of close matches
        weight = 1.0 / (distance * distance)
        left_weights.append(weight)
        left_points.append(screen_pos)
    
    # Process right eye data
    for pupil_pos, screen_pos in right_pupil_to_screen:
        distance = np.sqrt(np.sum((np.array(right_pupil_pos) - np.array(pupil_pos))**2))
        if distance < 0.001:
            distance = 0.001
        # Use a squared inverse distance for stronger weighting of close matches
        weight = 1.0 / (distance * distance)
        right_weights.append(weight)
        right_points.append(screen_pos)
    
    # Normalize weights
    left_weights = np.array(left_weights)
    left_weights = left_weights / np.sum(left_weights)
    right_weights = np.array(right_weights)
    right_weights = right_weights / np.sum(right_weights)
    
    # Calculate weighted average positions - use only top 3 weights for better accuracy
    if len(left_weights) > 3:
        top_indices = np.argsort(left_weights)[-3:]
        left_weights = left_weights[top_indices]
        left_weights = left_weights / np.sum(left_weights)  # Renormalize
        left_points = [left_points[i] for i in top_indices]
    
    if len(right_weights) > 3:
        top_indices = np.argsort(right_weights)[-3:]
        right_weights = right_weights[top_indices]
        right_weights = right_weights / np.sum(right_weights)  # Renormalize
        right_points = [right_points[i] for i in top_indices]
    
    # Calculate weighted average positions
    left_x = np.sum([w * p[0] for w, p in zip(left_weights, left_points)])
    left_y = np.sum([w * p[1] for w, p in zip(left_weights, left_points)])
    right_x = np.sum([w * p[0] for w, p in zip(right_weights, right_points)])
    right_y = np.sum([w * p[1] for w, p in zip(right_weights, right_points)])
    
    # Combine estimates from both eyes (weighted by confidence)
    # Determine confidence based on the highest weight found for each eye
    left_confidence = np.max(left_weights) if len(left_weights) > 0 else 0
    right_confidence = np.max(right_weights) if len(right_weights) > 0 else 0
    
    total_confidence = left_confidence + right_confidence
    if total_confidence > 0:
        # Use weighted average based on confidence
        gaze_x = (left_x * left_confidence + right_x * right_confidence) / total_confidence
        gaze_y = (left_y * left_confidence + right_y * right_confidence) / total_confidence
    else:
        # Simple average as fallback
        gaze_x = (left_x + right_x) / 2
        gaze_y = (left_y + right_y) / 2
    
    return (gaze_x, gaze_y)

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

def extract_eye(gray, eye_points, landmarks):
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
    current_time = time.time()
    
    # Check for timeout - if we haven't updated in more than 0.5 seconds
    if current_pos is None:
        if current_time - last_updated_time > 0.5:
            # We're stuck - return the last position but mark as stuck
            if stuck_since == 0:  # First time getting stuck
                stuck_since = current_time
            return None, last_updated_time, last_pos, stuck_since
        else:
            # Return the last known position
            return last_pos, last_updated_time, last_pos, stuck_since
    
    # If we have a valid new position, we're not stuck
    stuck_since = 0
    
    # Add the new position to history
    history.append(current_pos)
    
    # Calculate weighted average - give more weight to recent positions
    # Handle case where history has fewer elements than expected
    history_len = len(history)
    if history_len == 0:
        # No history data available
        return current_pos, current_time, current_pos, 0
    elif history_len == 1:
        # Only one position in history
        return history[0], current_time, history[0], 0
    else:
        # Create weights appropriate for the actual history length
        weights = [0.2 * (i + 1) for i in range(history_len)]
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]  # Normalize weights
        
        # Calculate weighted average
        avg_x = 0
        avg_y = 0
        for i, pos in enumerate(history):
            weight = weights[i]
            avg_x += pos[0] * weight
            avg_y += pos[1] * weight
        
        smoothed_pos = (int(avg_x), int(avg_y))
        return smoothed_pos, current_time, smoothed_pos, 0

# Main eye tracking function
def run_eye_tracking():
    # Declare global variables that we'll be modifying
    global left_pupil_history, right_pupil_history
    global left_pupil_last_updated, right_pupil_last_updated
    
    # Reset history
    left_pupil_history.clear()
    right_pupil_history.clear()
    
    # Run calibration first
    left_pupil_to_screen, right_pupil_to_screen = calibrate_eye_tracking(detector, predictor)
    
    if left_pupil_to_screen is None or right_pupil_to_screen is None:
        print("Calibration failed or was cancelled.")
        return
    
    # Create fullscreen window for tracking
    cv2.namedWindow("Eye Tracking", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Eye Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return
        
    # Variables for tracking stuck detection
    left_pupil_stuck_since = 0
    right_pupil_stuck_since = 0
    last_left_pupil_pos = None
    last_right_pupil_pos = None
    
    # For tracking where gaze position is
    last_gaze_point = None
    gaze_smoothing_queue = deque(maxlen=5)  # For smoother gaze tracking
    
    # Run until ESC pressed
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
            
        # Mirror the frame horizontally
        frame = cv2.flip(frame, 1)
        
        # Create a tracking display
        tracking_display = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = detector(gray)
        
        # Variables to store detected gaze
        gaze_point = None
        
        if len(faces) > 0:
            # Get facial landmarks
            landmarks = predictor(gray, faces[0])
            
            # Process left eye - properly unpack the tuple
            left_eye_data = extract_eye(gray, range(36, 42), landmarks)
            left_eye_img, left_eye_pos = left_eye_data
            left_pupil_pos = None
            
            if left_eye_img is not None:
                left_eye_display = cv2.cvtColor(left_eye_img, cv2.COLOR_GRAY2BGR)
                left_pupil_pos = detect_pupil(left_eye_img)
                
                if left_pupil_pos is not None:
                    # Update time of last detection
                    left_pupil_last_updated = time.time()
                    
                    # Draw pupil on eye image
                    cv2.circle(left_eye_display, (int(left_pupil_pos[0]), int(left_pupil_pos[1])), 
                              3, (0, 0, 255), -1)
                    
                    # Calculate screen position using calibration data
                    smooth_left_pupil_pos, left_pupil_stuck_since = smooth_position(
                        left_pupil_pos, left_pupil_history, left_pupil_last_updated, 
                        last_left_pupil_pos, left_pupil_stuck_since
                    )
                    last_left_pupil_pos = smooth_left_pupil_pos
            
            # Process right eye - properly unpack the tuple
            right_eye_data = extract_eye(gray, range(42, 48), landmarks)
            right_eye_img, right_eye_pos = right_eye_data
            right_pupil_pos = None
            
            if right_eye_img is not None:
                right_eye_display = cv2.cvtColor(right_eye_img, cv2.COLOR_GRAY2BGR)
                right_pupil_pos = detect_pupil(right_eye_img)
                
                if right_pupil_pos is not None:
                    # Update time of last detection
                    right_pupil_last_updated = time.time()
                    
                    # Draw pupil on eye image
                    cv2.circle(right_eye_display, (int(right_pupil_pos[0]), int(right_pupil_pos[1])), 
                              3, (0, 0, 255), -1)
                    
                    # Calculate screen position using calibration data
                    smooth_right_pupil_pos, right_pupil_stuck_since = smooth_position(
                        right_pupil_pos, right_pupil_history, right_pupil_last_updated, 
                        last_right_pupil_pos, right_pupil_stuck_since
                    )
                    last_right_pupil_pos = smooth_right_pupil_pos
            
            # Estimate gaze point if both pupils detected
            if left_pupil_pos is not None and right_pupil_pos is not None:
                gaze_point = estimate_gaze_point(left_pupil_pos, right_pupil_pos, 
                                              left_pupil_to_screen, right_pupil_to_screen)
                
                if gaze_point is not None:
                    # Apply additional smoothing to gaze point
                    gaze_smoothing_queue.append(gaze_point)
                    if len(gaze_smoothing_queue) >= 3:  # Need at least 3 points for smoothing
                        # Calculate average of recent gaze points
                        smoothed_x = sum(p[0] for p in gaze_smoothing_queue) / len(gaze_smoothing_queue)
                        smoothed_y = sum(p[1] for p in gaze_smoothing_queue) / len(gaze_smoothing_queue)
                        gaze_point = (smoothed_x, smoothed_y)
                        
                    last_gaze_point = gaze_point
        
        # Add frame to corner - position depends on gaze
        frame_height, frame_width = frame.shape[:2]
        frame_resized = cv2.resize(frame, (int(frame_width/4), int(frame_height/4)))
        frame_resized_h, frame_resized_w = frame_resized.shape[:2]
        padding = 20
        
        # Determine position for webcam preview based on current gaze point
        # By default, place in top-left
        preview_x = padding
        preview_y = padding
        
        if last_gaze_point is not None:
            # Place preview opposite to where the user is looking
            if last_gaze_point[0] < screen_width / 2:  # Left side
                preview_x = screen_width - frame_resized_w - padding
            else:  # Right side
                preview_x = padding
                
            if last_gaze_point[1] < screen_height / 2:  # Top half
                preview_y = screen_height - frame_resized_h - padding
            else:  # Bottom half
                preview_y = padding
                
        # Display frame in the determined position
        tracking_display[preview_y:preview_y+frame_resized_h, 
                       preview_x:preview_x+frame_resized_w] = frame_resized
                
        # Add rectangle around preview
        cv2.rectangle(tracking_display, 
                    (preview_x-1, preview_y-1), 
                    (preview_x+frame_resized_w+1, preview_y+frame_resized_h+1), 
                    (255, 255, 255), 1)
            
        # Display eye images
        if left_eye_img is not None and 'left_eye_display' in locals():
            left_eye_resized = cv2.resize(left_eye_display, (EYE_DISPLAY_WIDTH, EYE_DISPLAY_HEIGHT))
            
            # Position eye displays based on webcam position
            if preview_x < screen_width / 2:  # Webcam on left
                eye_x = screen_width - EYE_DISPLAY_WIDTH * 2 - padding
            else:  # Webcam on right
                eye_x = padding
                
            if preview_y < screen_height / 2:  # Webcam on top
                eye_y = screen_height - EYE_DISPLAY_HEIGHT - padding
            else:  # Webcam on bottom
                eye_y = padding
                
            tracking_display[eye_y:eye_y+EYE_DISPLAY_HEIGHT, 
                           eye_x:eye_x+EYE_DISPLAY_WIDTH] = left_eye_resized
        
        if right_eye_img is not None and 'right_eye_display' in locals():
            right_eye_resized = cv2.resize(right_eye_display, (EYE_DISPLAY_WIDTH, EYE_DISPLAY_HEIGHT))
            tracking_display[eye_y:eye_y+EYE_DISPLAY_HEIGHT, 
                           eye_x+EYE_DISPLAY_WIDTH:eye_x+EYE_DISPLAY_WIDTH*2] = right_eye_resized
        
        # Draw gaze point
        if gaze_point is not None:
            # Draw gaze point on tracking display
            cv2.circle(tracking_display, (int(gaze_point[0]), int(gaze_point[1])), 20, (0, 255, 0), -1)
            
            # Draw gaze coordinates - position text near the webcam
            text = f"Gaze: ({int(gaze_point[0])}, {int(gaze_point[1])})"
            text_y = preview_y + frame_resized_h + 30 if preview_y + frame_resized_h + 30 < screen_height else preview_y - 30
            cv2.putText(tracking_display, text, 
                      (preview_x, text_y), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Instruction for exiting
        cv2.putText(tracking_display, "Press ESC to exit", 
                  (padding, screen_height-padding), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display the main tracking window
        cv2.imshow("Eye Tracking", tracking_display)
        
        # Check for exit key
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_eye_tracking()
