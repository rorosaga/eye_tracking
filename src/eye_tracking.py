import cv2
import dlib
import numpy as np
from collections import deque
import time
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.spatial import distance

# Global variables
detector = None
predictor = None
eye_tracker_state = {
    'prev_left_pupil': None,
    'prev_right_pupil': None,
    'gaze_history': deque(maxlen=5),
    'last_blink_time': 0,
    'is_calibrated': False
}

# Calibration data
calibration_data = {
    'points': [],
    'eye_features': []
}

# Gaze estimation models
gaze_models = {
    'x': None,
    'y': None
}

# Constants
FACE_POSITION_THRESHOLD = 50
EYE_AR_THRESHOLD = 0.2
PUPIL_DARKNESS_THRESHOLD = 40  # Lower = darker
SMOOTHING_FACTOR = 0.65        # Higher = smoother
BLINK_TIMEOUT = 0.15          # Seconds

# Eye landmarks indices (for dlib's 68 point model)
LEFT_EYE_LANDMARKS = list(range(42, 48))   # User's left eye
RIGHT_EYE_LANDMARKS = list(range(36, 42))  # User's right eye

def init_eye_tracker(predictor_path="../data/shape_predictor_68_face_landmarks.dat"):
    """Initialize the eye tracker with necessary components"""
    global detector, predictor
    
    # Initialize face detector and landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    
    return True

def eye_aspect_ratio(eye_points):
    """Calculate the eye aspect ratio to detect blinks"""
    # Compute the euclidean distances between vertical eye landmarks
    A = distance.euclidean(eye_points[1], eye_points[5])
    B = distance.euclidean(eye_points[2], eye_points[4])
    
    # Compute the euclidean distance between horizontal eye landmarks
    C = distance.euclidean(eye_points[0], eye_points[3])
    
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    
    return ear

def extract_eye_region(frame, landmarks, eye_landmarks_indices):
    """Extract eye region with padding and normalization"""
    points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in eye_landmarks_indices])
    
    # Get bounding rectangle
    min_x = np.min(points[:, 0])
    max_x = np.max(points[:, 0])
    min_y = np.min(points[:, 1])
    max_y = np.max(points[:, 1])
    
    # Add padding
    padding_x = int((max_x - min_x) * 0.2)
    padding_y = int((max_y - min_y) * 0.25)
    
    # Ensure within frame bounds
    min_x = max(0, min_x - padding_x)
    min_y = max(0, min_y - padding_y)
    max_x = min(frame.shape[1] - 1, max_x + padding_x)
    max_y = min(frame.shape[0] - 1, max_y + padding_y)
    
    # Extract eye region
    eye_region = frame[min_y:max_y, min_x:max_x]
    
    # Convert to grayscale if needed
    if len(eye_region.shape) == 3:
        eye_region_gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    else:
        eye_region_gray = eye_region
    
    # Apply histogram equalization for better contrast
    eye_region_gray = cv2.equalizeHist(eye_region_gray)
    
    return eye_region_gray, (min_x, min_y, max_x, max_y)

def robust_pupil_detection(eye_img):
    """Advanced pupil detection using multiple techniques combined"""
    if eye_img is None or eye_img.size == 0 or eye_img.shape[0] < 10 or eye_img.shape[1] < 10:
        return None
    
    # Image dimensions
    height, width = eye_img.shape
    center_x, center_y = width // 2, height // 2
    
    # Apply preprocessing
    # Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(eye_img, (5, 5), 0)
    
    # Create a binary threshold to identify the darkest regions
    _, thresholded = cv2.threshold(
        blurred, 
        np.percentile(blurred, 20),  # Use 20th percentile as threshold
        255, 
        cv2.THRESH_BINARY_INV
    )
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel, iterations=1)
    thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If no contours found, try another approach with adaptive thresholding
    if not contours:
        adaptive_thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 5)
        contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Process contours if found
    if contours:
        valid_contours = []
        
        for contour in contours:
            # Filter out tiny contours
            area = cv2.contourArea(contour)
            if area < 10:
                continue
                
            # Filter by size relative to eye
            min_area = width * height * 0.01  # Min 1% of eye area
            max_area = width * height * 0.35  # Max 35% of eye area
            
            if not (min_area <= area <= max_area):
                continue
            
            # Calculate contour center
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
                
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Score based on:
            # 1. Circularity (how round the contour is)
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # 2. Darkness (pupil should be dark)
            mask = np.zeros_like(eye_img)
            cv2.drawContours(mask, [contour], 0, 255, -1)
            mean_intensity = np.mean(eye_img[mask > 0]) if np.sum(mask > 0) > 0 else 255
            darkness = 1.0 - (mean_intensity / 255.0)
            
            # 3. Centrality (pupils are typically near center)
            dist_from_center = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
            max_dist = np.sqrt((width/2)**2 + (height/2)**2)
            centrality = 1.0 - (dist_from_center / max_dist)
            
            # Calculate overall score (weighted sum)
            score = circularity * 0.3 + darkness * 0.5 + centrality * 0.2
            
            valid_contours.append((cx, cy, score, area))
        
        # Select best contour based on score
        if valid_contours:
            # Sort by score (highest first)
            valid_contours.sort(key=lambda x: x[2], reverse=True)
            best_center = valid_contours[0][:2]
            return best_center
    
    # Fallback: Try to find darkest region in the eye
    # Apply min filter to enhance dark regions
    min_filtered = cv2.erode(blurred, np.ones((3,3), np.uint8))
    
    # Find darkest point
    min_val = np.min(min_filtered)
    if min_val < PUPIL_DARKNESS_THRESHOLD:
        min_locs = np.where(min_filtered == min_val)
        if len(min_locs[0]) > 0:
            # Average the locations if multiple darkest points
            avg_y = int(np.mean(min_locs[0]))
            avg_x = int(np.mean(min_locs[1]))
            
            # Check if within reasonable bounds (not too close to edge)
            edge_margin = min(width, height) // 10
            if (edge_margin <= avg_x < width - edge_margin and 
                edge_margin <= avg_y < height - edge_margin):
                return (avg_x, avg_y)
    
    # Last resort: use center of eye
    return (center_x, center_y)

def extract_eye_features(frame, landmarks):
    """Extract comprehensive eye features for gaze estimation"""
    # Extract left and right eye regions
    left_eye_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) 
                               for i in LEFT_EYE_LANDMARKS])
    right_eye_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) 
                                for i in RIGHT_EYE_LANDMARKS])
    
    # Calculate eye aspect ratios
    left_ear = eye_aspect_ratio(left_eye_points)
    right_ear = eye_aspect_ratio(right_eye_points)
    
    # Check if eyes are open
    eyes_open = left_ear > EYE_AR_THRESHOLD and right_ear > EYE_AR_THRESHOLD
    
    if not eyes_open:
        # Handle blink
        eye_tracker_state['last_blink_time'] = time.time()
        return None
    
    # If recently blinked, skip a few frames
    if time.time() - eye_tracker_state['last_blink_time'] < BLINK_TIMEOUT:
        return None
    
    # Extract eye regions
    left_eye_gray, left_bounds = extract_eye_region(frame, landmarks, LEFT_EYE_LANDMARKS)
    right_eye_gray, right_bounds = extract_eye_region(frame, landmarks, RIGHT_EYE_LANDMARKS)
    
    # Detect pupils
    left_pupil_pos = robust_pupil_detection(left_eye_gray)
    right_pupil_pos = robust_pupil_detection(right_eye_gray)
    
    # If either pupil not detected, return None
    if left_pupil_pos is None or right_pupil_pos is None:
        return None
    
    # Convert pupil positions to absolute frame coordinates
    left_pupil_abs = (left_bounds[0] + left_pupil_pos[0], 
                     left_bounds[1] + left_pupil_pos[1])
    right_pupil_abs = (right_bounds[0] + right_pupil_pos[0], 
                      right_bounds[1] + right_pupil_pos[1])
    
    # Apply temporal smoothing
    if eye_tracker_state['prev_left_pupil'] is not None:
        left_pupil_smoothed = (
            int(SMOOTHING_FACTOR * eye_tracker_state['prev_left_pupil'][0] + 
                (1 - SMOOTHING_FACTOR) * left_pupil_abs[0]),
            int(SMOOTHING_FACTOR * eye_tracker_state['prev_left_pupil'][1] + 
                (1 - SMOOTHING_FACTOR) * left_pupil_abs[1])
        )
        right_pupil_smoothed = (
            int(SMOOTHING_FACTOR * eye_tracker_state['prev_right_pupil'][0] + 
                (1 - SMOOTHING_FACTOR) * right_pupil_abs[0]),
            int(SMOOTHING_FACTOR * eye_tracker_state['prev_right_pupil'][1] + 
                (1 - SMOOTHING_FACTOR) * right_pupil_abs[1])
        )
    else:
        left_pupil_smoothed = left_pupil_abs
        right_pupil_smoothed = right_pupil_abs
    
    # Update previous positions
    eye_tracker_state['prev_left_pupil'] = left_pupil_smoothed
    eye_tracker_state['prev_right_pupil'] = right_pupil_smoothed
    
    # Calculate additional features
    # 1. Pupil positions normalized by eye size
    left_eye_width = left_bounds[2] - left_bounds[0]
    left_eye_height = left_bounds[3] - left_bounds[1]
    right_eye_width = right_bounds[2] - right_bounds[0]
    right_eye_height = right_bounds[3] - right_bounds[1]
    
    left_pupil_rel_x = (left_pupil_smoothed[0] - left_bounds[0]) / left_eye_width
    left_pupil_rel_y = (left_pupil_smoothed[1] - left_bounds[1]) / left_eye_height
    right_pupil_rel_x = (right_pupil_smoothed[0] - right_bounds[0]) / right_eye_width
    right_pupil_rel_y = (right_pupil_smoothed[1] - right_bounds[1]) / right_eye_height
    
    # 2. Angle between pupils
    pupil_vector = (right_pupil_smoothed[0] - left_pupil_smoothed[0],
                   right_pupil_smoothed[1] - left_pupil_smoothed[1])
    pupil_angle = np.arctan2(pupil_vector[1], pupil_vector[0])
    
    # 3. Distance between pupils (normalized by face width)
    face_width = landmarks.part(16).x - landmarks.part(0).x
    pupil_distance = np.sqrt(pupil_vector[0]**2 + pupil_vector[1]**2) / face_width
    
    # Combine all features into a vector
    features = [
        left_pupil_rel_x, left_pupil_rel_y,
        right_pupil_rel_x, right_pupil_rel_y,
        pupil_angle, pupil_distance,
        left_ear, right_ear
    ]
    
    return np.array(features)

def visualize_eye_tracking(frame, landmarks, eye_features=None):
    """Visualize eye tracking on the frame"""
    result_frame = frame.copy()
    
    if landmarks is not None:
        # Draw landmarks for left eye
        left_eye_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) 
                                   for i in LEFT_EYE_LANDMARKS])
        # Draw landmarks for right eye
        right_eye_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) 
                                    for i in RIGHT_EYE_LANDMARKS])
                                    
        # Draw eye outlines
        cv2.polylines(result_frame, [left_eye_points], True, (0, 255, 0), 1)
        cv2.polylines(result_frame, [right_eye_points], True, (0, 255, 0), 1)
        
        # Draw pupils if available
        if eye_tracker_state['prev_left_pupil'] is not None:
            cv2.circle(result_frame, eye_tracker_state['prev_left_pupil'], 3, (0, 0, 255), -1)
        if eye_tracker_state['prev_right_pupil'] is not None:
            cv2.circle(result_frame, eye_tracker_state['prev_right_pupil'], 3, (0, 0, 255), -1)
    
    return result_frame

def process_frame_for_tracking(frame):
    """Process frame to detect face, extract eyes, and get gaze features"""
    if frame is None or frame.size == 0:
        return None, None
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = detector(gray)
    
    if not faces:
        return None, None
    
    # Use first face
    face = faces[0]
    
    # Get facial landmarks
    landmarks = predictor(gray, face)
    
    # Extract eye features
    eye_features = extract_eye_features(frame, landmarks)
    
    return landmarks, eye_features

def train_gaze_model():
    """Train SVR models for gaze prediction using collected calibration data"""
    global gaze_models, eye_tracker_state
    
    if len(calibration_data['points']) < 5:
        print("Not enough calibration data points")
        return False
    
    # Extract x and y coordinates
    screen_x = [p[0] for p in calibration_data['points']]
    screen_y = [p[1] for p in calibration_data['points']]
    
    # Create SVR pipelines with preprocessing
    x_model = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR(kernel='rbf', C=100, gamma='auto'))
    ])
    
    y_model = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR(kernel='rbf', C=100, gamma='auto'))
    ])
    
    # Train models
    x_model.fit(calibration_data['eye_features'], screen_x)
    y_model.fit(calibration_data['eye_features'], screen_y)
    
    # Save models
    gaze_models['x'] = x_model
    gaze_models['y'] = y_model
    
    eye_tracker_state['is_calibrated'] = True
    print("Gaze models trained successfully")
    return True

def predict_gaze(eye_features):
    """Predict gaze position using trained models"""
    if not eye_tracker_state['is_calibrated'] or eye_features is None:
        return None
    
    # Reshape for prediction (sklearn expects 2D array)
    features_2d = eye_features.reshape(1, -1)
    
    # Predict x and y coordinates
    gaze_x = gaze_models['x'].predict(features_2d)[0]
    gaze_y = gaze_models['y'].predict(features_2d)[0]
    
    # Clamp to [0,1] range
    gaze_x = max(0, min(1, gaze_x))
    gaze_y = max(0, min(1, gaze_y))
    
    # Apply temporal smoothing to gaze
    if eye_tracker_state['gaze_history']:
        prev_gaze = eye_tracker_state['gaze_history'][-1]
        smoothed_x = SMOOTHING_FACTOR * prev_gaze[0] + (1 - SMOOTHING_FACTOR) * gaze_x
        smoothed_y = SMOOTHING_FACTOR * prev_gaze[1] + (1 - SMOOTHING_FACTOR) * gaze_y
        gaze_pos = (smoothed_x, smoothed_y)
    else:
        gaze_pos = (gaze_x, gaze_y)
    
    # Add to history
    eye_tracker_state['gaze_history'].append(gaze_pos)
    
    return gaze_pos

def run_eye_tracker():
    """Main function with single-window integrated pipeline"""
    # Initialize eye tracker
    init_eye_tracker()
    
    # Create single window for all steps
    cv2.namedWindow("Eye Tracker", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Eye Tracker", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # Start webcam
    cap = cv2.VideoCapture(0)
    
    # Get screen dimensions
    screen = np.zeros((1080, 1920, 3), dtype=np.uint8)
    screen_h, screen_w = screen.shape[:2]
    
    # State machine variables
    # 0 = face positioning, 1 = calibration, 2 = gaze tracking
    current_state = 0
    face_positioned = False
    calibration_started = False
    calibration_points_grid = []
    current_point_idx = 0
    point_start_time = 0
    point_duration = 2.0
    
    # Main loop
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            break
        
        # Reset screen
        screen.fill(0)
        
        # Process frame to get landmarks and eye features
        landmarks, eye_features = process_frame_for_tracking(frame)
        
        # Handle different states
        if current_state == 0:  # Face positioning
            # Draw face positioning guide (oval)
            center_x, center_y = screen_w // 2, screen_h // 2
            face_guide_width, face_guide_height = 300, 400
            cv2.ellipse(screen, (center_x, center_y), (face_guide_width//2, face_guide_height//2), 
                       0, 0, 360, (0, 255, 0), 2)
            
            # Show webcam feed in the center
            cam_h, cam_w = 480, 640
            frame_resized = cv2.resize(frame, (cam_w, cam_h))
            screen[center_y-cam_h//2:center_y+cam_h//2, 
                   center_x-cam_w//2:center_x+cam_w//2] = frame_resized
            
            # Instructions
            cv2.putText(screen, "Center your face in the green oval", 
                      (center_x - 250, 50), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(screen, "Press SPACE when ready, ESC to exit", 
                      (center_x - 250, 100), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Check if face is positioned correctly
            if landmarks is not None:
                face_center_x = (landmarks.part(30).x)  # Nose tip
                face_center_y = (landmarks.part(30).y)
                
                # Map to screen coordinates
                screen_face_x = center_x + (face_center_x - frame.shape[1]//2)
                screen_face_y = center_y + (face_center_y - frame.shape[0]//2)
                
                # Draw face center
                cv2.circle(screen, (screen_face_x, screen_face_y), 5, (0, 165, 255), -1)
                
                # Check if face is centered
                cam_center_x, cam_center_y = frame.shape[1] // 2, frame.shape[0] // 2
                if (abs(face_center_x - cam_center_x) < FACE_POSITION_THRESHOLD and 
                    abs(face_center_y - cam_center_y) < FACE_POSITION_THRESHOLD):
                    # Face is well positioned
                    cv2.putText(screen, "Face positioned correctly! Press SPACE to continue", 
                              (center_x - 300, center_y + 300), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    face_positioned = True
                else:
                    # Guide the user
                    direction = ""
                    if face_center_x < cam_center_x - FACE_POSITION_THRESHOLD:
                        direction += "Move right. "
                    elif face_center_x > cam_center_x + FACE_POSITION_THRESHOLD:
                        direction += "Move left. "
                    if face_center_y < cam_center_y - FACE_POSITION_THRESHOLD:
                        direction += "Move down. "
                    elif face_center_y > cam_center_y + FACE_POSITION_THRESHOLD:
                        direction += "Move up. "
                    
                    cv2.putText(screen, direction, 
                              (center_x - 200, center_y + 300), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
            else:
                cv2.putText(screen, "Position your face in the frame", 
                          (center_x - 200, center_y + 300), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        elif current_state == 1:  # Calibration
            if not calibration_started:
                # Initialize calibration
                screen.fill(0)
                cv2.putText(screen, "Look at each red dot until it disappears", 
                          (screen_w//4, screen_h//2 - 50), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                cv2.putText(screen, "Keep your head still during calibration", 
                          (screen_w//4, screen_h//2 + 50), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                cv2.putText(screen, "Press SPACE to start", 
                          (screen_w//4, screen_h//2 + 150), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                
                # Show small webcam preview
                preview_h, preview_w = 180, 240
                frame_small = cv2.resize(frame, (preview_w, preview_h))
                screen[20:20+preview_h, screen_w-preview_w-20:screen_w-20] = frame_small
                
                # Check for key press to start calibration
                key = cv2.waitKey(1) & 0xFF
                if key == 32:  # SPACE
                    calibration_started = True
                    
                    # Reset calibration data
                    calibration_data['points'] = []
                    calibration_data['eye_features'] = []
                    eye_tracker_state['is_calibrated'] = False
                    
                    # Create calibration points grid (3x3 grid + center)
                    calibration_points_grid = []
                    for y in [0.1, 0.5, 0.9]:
                        for x in [0.1, 0.5, 0.9]:
                            # Avoid duplicating the center point
                            if not (x == 0.5 and y == 0.5) or not calibration_points_grid:
                                calibration_points_grid.append((x, y))
                    
                    # Start with first point
                    current_point_idx = 0
                    point_start_time = time.time()
                    
                    # Reset eye tracking state for clean calibration
                    eye_tracker_state['prev_left_pupil'] = None
                    eye_tracker_state['prev_right_pupil'] = None
                    eye_tracker_state['gaze_history'].clear()
            else:
                # Running calibration
                current_time = time.time()
                
                # Check if all points have been processed
                if current_point_idx >= len(calibration_points_grid):
                    # Train the model if we have enough points
                    if len(calibration_data['points']) >= 5:
                        train_gaze_model()
                        
                        # Show completion message
                        screen.fill(0)
                        cv2.putText(screen, "Calibration Complete!", 
                                  (screen_w//3, screen_h//2 - 50), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
                        cv2.putText(screen, f"Collected {len(calibration_data['points'])} points", 
                                  (screen_w//3, screen_h//2 + 50), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        cv2.putText(screen, "Press any key to start gaze tracking", 
                                  (screen_w//3, screen_h//2 + 150), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        
                        # Show small webcam preview
                        preview_h, preview_w = 180, 240
                        frame_small = cv2.resize(frame, (preview_w, preview_h))
                        screen[20:20+preview_h, screen_w-preview_w-20:screen_w-20] = frame_small
                        
                        # Wait for key press
                        cv2.imshow("Eye Tracker", screen)
                        cv2.waitKey(0)
                        
                        # Move to gaze tracking state
                        current_state = 2
                        continue
                    else:
                        # Calibration failed - not enough points
                        screen.fill(0)
                        cv2.putText(screen, "Calibration Failed", 
                                  (screen_w//3, screen_h//2 - 50), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                        cv2.putText(screen, "Not enough valid points collected", 
                                  (screen_w//3, screen_h//2 + 50), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        cv2.putText(screen, "Press any key to retry", 
                                  (screen_w//3, screen_h//2 + 150), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        
                        # Wait for key press
                        cv2.imshow("Eye Tracker", screen)
                        cv2.waitKey(0)
                        
                        # Reset calibration
                        calibration_started = False
                        continue
                
                # Get current calibration point
                point_rel = calibration_points_grid[current_point_idx]
                point_x = int(point_rel[0] * screen_w)
                point_y = int(point_rel[1] * screen_h)
                
                # Calculate time elapsed for this point
                point_elapsed = current_time - point_start_time
                
                # Animation for the calibration point
                # Start small, grow, hold, shrink
                if point_elapsed < point_duration * 0.3:
                    # Growing phase (0-30% of duration)
                    size = int(5 + 15 * (point_elapsed / (point_duration * 0.3)))
                elif point_elapsed < point_duration * 0.7:
                    # Steady phase (30-70% of duration)
                    size = 20
                else:
                    # Shrinking phase (70-100% of duration)
                    remaining = point_duration - point_elapsed
                    size = max(5, int(20 * (remaining / (point_duration * 0.3))))
                
                # Collect eye data during the steady phase
                if (eye_features is not None and 
                    point_duration * 0.3 <= point_elapsed <= point_duration * 0.7):
                    
                    # Save calibration data
                    if point_rel not in calibration_data['points']:
                        calibration_data['points'].append(point_rel)
                        calibration_data['eye_features'].append(eye_features)
                        
                        # Visual feedback for data collection
                        cv2.circle(screen, (point_x, point_y), 30, (0, 255, 0), 1)
                
                # Draw the calibration point
                cv2.circle(screen, (point_x, point_y), size, (0, 0, 255), -1)
                
                # Show progress
                progress = (current_point_idx + point_elapsed/point_duration) / len(calibration_points_grid) * 100
                cv2.putText(screen, f"Calibration Progress: {progress:.0f}%", 
                          (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(screen, f"Points: {len(calibration_data['points'])}/{len(calibration_points_grid)}", 
                          (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Show small webcam preview with eye tracking visualization
                preview_h, preview_w = 180, 240
                if landmarks is not None:
                    tracked_frame = visualize_eye_tracking(frame, landmarks, eye_features)
                    frame_small = cv2.resize(tracked_frame, (preview_w, preview_h))
                else:
                    frame_small = cv2.resize(frame, (preview_w, preview_h))
                screen[20:20+preview_h, screen_w-preview_w-20:screen_w-20] = frame_small
                
                # Move to next point when time is up
                if point_elapsed >= point_duration:
                    current_point_idx += 1
                    point_start_time = current_time
        
        elif current_state == 2:  # Gaze tracking
            # Predict gaze position if calibrated
            gaze_position = predict_gaze(eye_features) if eye_features is not None else None
            
            if gaze_position is not None:
                # Convert to screen coordinates
                gaze_x = int(gaze_position[0] * screen_w)
                gaze_y = int(gaze_position[1] * screen_h)
                
                # Draw red circle outline for gaze
                cv2.circle(screen, (gaze_x, gaze_y), 20, (0, 0, 255), 2)
            
            # Show webcam preview with eye tracking visualization
            preview_h, preview_w = 250, 320
            if landmarks is not None:
                tracked_frame = visualize_eye_tracking(frame, landmarks, eye_features)
                frame_small = cv2.resize(tracked_frame, (preview_w, preview_h))
            else:
                frame_small = cv2.resize(frame, (preview_w, preview_h))
            screen[20:20+preview_h, 20:20+preview_w] = frame_small
            
            # Show status and instructions
            cv2.putText(screen, "Eye Tracking Active", 
                      (preview_w + 40, 40), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            if gaze_position is None:
                cv2.putText(screen, "No gaze detected", 
                          (preview_w + 40, 80), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.putText(screen, "Press ESC to exit, R to recalibrate", 
                      (20, screen_h - 20), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            # Check for recalibration request
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                # Reset to calibration state
                current_state = 1
                calibration_started = False
        
        # Display the screen
        cv2.imshow("Eye Tracker", screen)
        
        # Check for exit key
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        
        # Check for state transitions
        if current_state == 0 and face_positioned and key == 32:  # SPACE
            current_state = 1  # Move to calibration
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Run the eye tracker
    run_eye_tracker()
