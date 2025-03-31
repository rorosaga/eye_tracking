import cv2
import dlib
import numpy as np
from collections import deque
import time
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pygaze.display import Display
from pygaze.screen import Screen
from pygaze.eyetracker import EyeTracker
import threading

# Configuration
FACE_DETECTOR_PATH = "../data/shape_predictor_68_face_landmarks.dat"
CALIBRATION_POINTS = 5  # Number of calibration points
HEATMAP_DECAY = 0.995  # Decay factor for heatmap (higher = longer persistence)
HEATMAP_INTENSITY = 0.2  # Intensity of new gaze points
GAZE_HISTORY_LENGTH = 10  # Length of gaze history for smoothing
DISPLAY_HEATMAP = True  # Toggle heatmap display

# Initialize face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(FACE_DETECTOR_PATH)

# Create a named window and start webcam
cv2.namedWindow('Gaze Tracker')
cap = cv2.VideoCapture(0)

# Get initial frame dimensions
ret, frame = cap.read()
if not ret:
    print("Failed to capture initial frame")
    exit()

frame_height, frame_width = frame.shape[:2]

# Tracking variables
calibration_mode = True
calibration_points = []
calibration_eye_positions = []
calibrated = False
smooth_gaze_point = None
last_gaze_update = 0

# Eye tracking variables
left_pupil_history = deque(maxlen=3)
right_pupil_history = deque(maxlen=3)
gaze_point_history = deque(maxlen=GAZE_HISTORY_LENGTH)

# Heatmap creation
heatmap = np.zeros((frame_height, frame_width), dtype=np.float32)
heatmap_colormap = cv2.applyColorMap(np.zeros((frame_height, frame_width), dtype=np.uint8), cv2.COLORMAP_JET)
heatmap_updated = False

# Create a custom colormap with transparency for low values
heatmap_colors = [(0, 0, 0, 0)]  # Start with transparent
heatmap_colors.extend(plt.cm.jet(np.linspace(0.1, 1, 255)))  # Add jet colors
custom_cmap = LinearSegmentedColormap.from_list('custom_jet', heatmap_colors, N=256)

# Data collection for analysis
gaze_points_data = []
fixation_duration = 0
fixation_threshold = 30  # pixels - maximum distance for fixation
last_fixation_point = None
fixation_start_time = 0
fixations = []  # Will store (x, y, duration) tuples

# Eye aspect ratio threshold for blink detection
EYE_AR_THRESHOLD = 0.2

def eye_aspect_ratio(eye_points):
    """Calculate eye aspect ratio for blink detection"""
    # Compute the euclidean distances between vertical eye landmarks
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    
    # Compute the euclidean distance between horizontal eye landmarks
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    
    return ear

def get_calibration_point(index, width, height):
    """Return position for calibration point based on index"""
    points = [
        (int(width * 0.1), int(height * 0.1)),   # Top left
        (int(width * 0.9), int(height * 0.1)),   # Top right
        (int(width * 0.5), int(height * 0.5)),   # Center
        (int(width * 0.1), int(height * 0.9)),   # Bottom left
        (int(width * 0.9), int(height * 0.9)),   # Bottom right
    ]
    
    if index < len(points):
        return points[index]
    else:
        return (int(width * 0.5), int(height * 0.5))  # Default to center

def draw_calibration_point(frame, point, active=True):
    """Draw calibration target point"""
    # Outer circle
    color = (0, 0, 255) if active else (100, 100, 100)
    cv2.circle(frame, point, 20, color, 2)
    
    # Inner circle
    cv2.circle(frame, point, 5, (255, 255, 255), -1)
    
    return frame

def extract_eye(frame, landmarks, eye_points):
    """Extract eye region from facial landmarks"""
    # Get the eye region points
    eye_region = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in eye_points])
    
    # Get min and max for bounding box
    min_x = np.min(eye_region[:, 0])
    max_x = np.max(eye_region[:, 0])
    min_y = np.min(eye_region[:, 1])
    max_y = np.max(eye_region[:, 1])
    
    # Add padding
    padding = 5
    min_x = max(0, min_x - padding)
    max_x = min(frame.shape[1], max_x + padding)
    min_y = max(0, min_y - padding)
    max_y = min(frame.shape[0], max_y + padding)
    
    # Extract the eye image
    eye_frame = frame[min_y:max_y, min_x:max_x]
    
    # Return eye image and position
    return eye_frame, (min_x, min_y, max_x - min_x, max_y - min_y)

def detect_pupil(eye_frame):
    """Detect pupil in the eye frame"""
    if eye_frame is None or eye_frame.size == 0:
        return None, None
    
    # Convert to grayscale if necessary
    if len(eye_frame.shape) == 3:
        eye_gray = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)
    else:
        eye_gray = eye_frame
    
    # Enhance contrast
    eye_gray = cv2.equalizeHist(eye_gray)
    
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(eye_gray, (5, 5), 0)
    
    # Find the darkest region as a simple pupil detector
    _, thresh = cv2.threshold(blur, 45, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour (likely to be the pupil)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate the center of mass
        M = cv2.moments(largest_contour)
        
        if M["m00"] != 0:
            pupil_x = int(M["m10"] / M["m00"])
            pupil_y = int(M["m01"] / M["m00"])
            return pupil_x, pupil_y
    
    # If contour method fails, try direct circle detection
    circles = cv2.HoughCircles(
        blur, 
        cv2.HOUGH_GRADIENT, 
        dp=1.2, 
        minDist=blur.shape[0]//4,
        param1=50, 
        param2=25, 
        minRadius=5, 
        maxRadius=30
    )
    
    if circles is not None:
        # Convert to integers
        circles = np.uint16(np.around(circles))
        
        # Get the first (best) circle
        circle = circles[0, 0]
        return circle[0], circle[1]
    
    # Final fallback - find darkest point
    minVal, _, minLoc, _ = cv2.minMaxLoc(blur)
    return minLoc[0], minLoc[1]

def smooth_gaze(current_gaze, history):
    """Apply smoothing to gaze point"""
    if current_gaze is None:
        return None
    
    # Add to history
    history.append(current_gaze)
    
    if len(history) < 2:
        return current_gaze
    
    # Weighted average with more weight on recent positions
    total_x = 0
    total_y = 0
    total_weight = 0
    
    # Higher weights for more recent positions - fixed to avoid index error
    # Create weights dynamically based on history length
    weights = [2**i for i in range(len(history))]  # Exponential weighting
    
    for i, (x, y) in enumerate(history):
        weight = weights[i]
        total_x += x * weight
        total_y += y * weight
        total_weight += weight
    
    # Return smoothed position
    return (int(total_x / total_weight), int(total_y / total_weight))

def map_eye_to_screen(left_pupil, right_pupil):
    """Map eye positions to screen coordinates using calibration data"""
    global calibrated, smooth_gaze_point, last_gaze_update
    
    # Only proceed if we have calibration data and valid pupil positions
    if not calibrated or len(calibration_eye_positions) < 4 or left_pupil is None or right_pupil is None:
        return None
    
    # Use average of both eye positions for better accuracy
    avg_pupil_x = (left_pupil[0] + right_pupil[0]) / 2
    avg_pupil_y = (left_pupil[1] + right_pupil[1]) / 2
    
    # Simple linear mapping from pupil position to screen position
    # Get min/max values from calibration data
    eye_x_values = [p[0] for p in calibration_eye_positions]
    eye_y_values = [p[1] for p in calibration_eye_positions]
    screen_x_values = [p[0] for p in calibration_points]
    screen_y_values = [p[1] for p in calibration_points]
    
    # Calculate normalized position
    try:
        x_min, x_max = min(eye_x_values), max(eye_x_values)
        y_min, y_max = min(eye_y_values), max(eye_y_values)
        
        # Normalize eye position between 0 and 1
        norm_x = (avg_pupil_x - x_min) / (x_max - x_min) if x_max > x_min else 0.5
        norm_y = (avg_pupil_y - y_min) / (y_max - y_min) if y_max > y_min else 0.5
        
        # Apply bounds
        norm_x = max(0, min(1, norm_x))
        norm_y = max(0, min(1, norm_y))
        
        # Map to screen dimensions
        screen_x = int(norm_x * frame_width)
        screen_y = int(norm_y * frame_height)
        
        # Update last gaze time
        last_gaze_update = time.time()
        
        return (screen_x, screen_y)
    except:
        return None

def update_heatmap(gaze_point):
    """Update the heatmap with a new gaze point"""
    global heatmap, heatmap_colormap, heatmap_updated
    
    if gaze_point is None:
        return
    
    # Create a temporary heatmap for the new point
    temp_heatmap = np.zeros_like(heatmap)
    
    # Draw a gaussian at the gaze point
    x, y = gaze_point
    sigma = 50  # Size of the gaussian blob
    
    # Ensure in bounds
    if 0 <= x < frame_width and 0 <= y < frame_height:
        # Add a circular blob
        cv2.circle(temp_heatmap, (x, y), sigma, 1.0, -1)
        
        # Apply gaussian blur
        temp_heatmap = cv2.GaussianBlur(temp_heatmap, (sigma*2+1, sigma*2+1), sigma/3)
        
        # Normalize
        if np.max(temp_heatmap) > 0:
            temp_heatmap = temp_heatmap / np.max(temp_heatmap)
        
        # Apply decay to existing heatmap and add new points
        heatmap = heatmap * HEATMAP_DECAY + temp_heatmap * HEATMAP_INTENSITY
        
        # Clamp values
        heatmap = np.clip(heatmap, 0, 1)
        
        # Update the colormap
        heatmap_colormap = cv2.applyColorMap(np.uint8(heatmap * 255), cv2.COLORMAP_JET)
        heatmap_updated = True

def save_heatmap():
    """Save the current heatmap as an image"""
    if not os.path.exists("../output"):
        os.makedirs("../output")
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    colorized = cv2.applyColorMap(np.uint8(heatmap * 255), cv2.COLORMAP_JET)
    cv2.imwrite(f"../output/heatmap_{timestamp}.png", colorized)
    print(f"Heatmap saved as ../output/heatmap_{timestamp}.png")

def save_gaze_data():
    """Save gaze data and fixations to CSV files"""
    if not os.path.exists("../output"):
        os.makedirs("../output")
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Save gaze points
    with open(f"../output/gaze_points_{timestamp}.csv", 'w') as f:
        f.write("timestamp,x,y\n")
        for point in gaze_points_data:
            f.write(f"{point[0]},{point[1]},{point[2]}\n")
    
    # Save fixations
    with open(f"../output/fixations_{timestamp}.csv", 'w') as f:
        f.write("x,y,duration_ms\n")
        for fix in fixations:
            f.write(f"{fix[0]},{fix[1]},{fix[2]}\n")
    
    print(f"Gaze data saved to ../output/gaze_points_{timestamp}.csv")
    print(f"Fixation data saved to ../output/fixations_{timestamp}.csv")

def update_fixations(gaze_point):
    """Update fixation tracking"""
    global last_fixation_point, fixation_start_time, fixation_duration, fixations
    
    if gaze_point is None:
        # If gaze point is lost, reset fixation
        if last_fixation_point is not None and fixation_duration > 100:
            # Save the previous fixation if it was long enough
            fixations.append((last_fixation_point[0], last_fixation_point[1], fixation_duration))
        
        last_fixation_point = None
        fixation_duration = 0
        return
    
    current_time = time.time() * 1000  # Convert to milliseconds
    
    if last_fixation_point is None:
        # Start a new fixation
        last_fixation_point = gaze_point
        fixation_start_time = current_time
        fixation_duration = 0
    else:
        # Check if still within fixation threshold
        dist = np.sqrt((gaze_point[0] - last_fixation_point[0])**2 + 
                      (gaze_point[1] - last_fixation_point[1])**2)
        
        if dist < fixation_threshold:
            # Update fixation duration
            fixation_duration = current_time - fixation_start_time
        else:
            # Moved outside threshold, save fixation if it was long enough
            if fixation_duration > 100:  # Only save fixations longer than 100ms
                fixations.append((last_fixation_point[0], last_fixation_point[1], fixation_duration))
            
            # Start new fixation
            last_fixation_point = gaze_point
            fixation_start_time = current_time
            fixation_duration = 0

# Main loop
try:
    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
        
        # Create copy for display
        display_frame = frame.copy()
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = detector(gray)
        
        # Pupil detection variables
        left_pupil = None
        right_pupil = None
        
        # Process faces
        for face in faces:
            # Get landmarks
            landmarks = predictor(gray, face)
            
            # Define eye regions
            left_eye_points = range(36, 42)  # Left eye landmarks (right in mirror image)
            right_eye_points = range(42, 48)  # Right eye landmarks (left in mirror image)
            
            # Get eye landmarks for blink detection
            left_eye_landmarks = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in left_eye_points])
            right_eye_landmarks = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in right_eye_points])
            
            # Check if eyes are open
            left_ear = eye_aspect_ratio(left_eye_landmarks)
            right_ear = eye_aspect_ratio(right_eye_landmarks)
            
            left_eye_open = left_ear > EYE_AR_THRESHOLD
            right_eye_open = right_ear > EYE_AR_THRESHOLD
            
            # Draw eye outlines
            cv2.polylines(display_frame, [left_eye_landmarks], True, (0, 255, 0), 1)
            cv2.polylines(display_frame, [right_eye_landmarks], True, (0, 255, 0), 1)
            
            # Process eyes if open
            if left_eye_open and right_eye_open:
                # Extract eye regions
                left_eye_img, left_eye_rect = extract_eye(frame, landmarks, left_eye_points)
                right_eye_img, right_eye_rect = extract_eye(frame, landmarks, right_eye_points)
                
                # Detect pupils
                if left_eye_img is not None and left_eye_img.size > 0:
                    left_pupil_pos_x, left_pupil_pos_y = detect_pupil(left_eye_img)
                    if left_pupil_pos_x is not None:
                        # Convert to original frame coordinates
                        left_pupil_x = left_eye_rect[0] + left_pupil_pos_x
                        left_pupil_y = left_eye_rect[1] + left_pupil_pos_y
                        left_pupil = (left_pupil_x, left_pupil_y)
                        
                        # Draw pupil
                        cv2.circle(display_frame, left_pupil, 3, (0, 0, 255), -1)
                
                if right_eye_img is not None and right_eye_img.size > 0:
                    right_pupil_pos_x, right_pupil_pos_y = detect_pupil(right_eye_img)
                    if right_pupil_pos_x is not None:
                        # Convert to original frame coordinates
                        right_pupil_x = right_eye_rect[0] + right_pupil_pos_x
                        right_pupil_y = right_eye_rect[1] + right_pupil_pos_y
                        right_pupil = (right_pupil_x, right_pupil_y)
                        
                        # Draw pupil
                        cv2.circle(display_frame, right_pupil, 3, (0, 0, 255), -1)
        
        # Calibration and gaze mapping logic
        if calibration_mode and len(calibration_points) < CALIBRATION_POINTS:
            # Show calibration point
            current_calib_point = get_calibration_point(len(calibration_points), frame_width, frame_height)
            display_frame = draw_calibration_point(display_frame, current_calib_point)
            
            # Display instructions
            cv2.putText(display_frame, "Look at the red circle and press SPACE to calibrate", 
                       (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(display_frame, f"Points: {len(calibration_points)}/{CALIBRATION_POINTS}", 
                       (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # If calibration complete, map gaze to screen
            if calibrated and left_pupil is not None and right_pupil is not None:
                # Map pupil positions to screen coordinates
                gaze_point = map_eye_to_screen(left_pupil, right_pupil)
                
                # Smooth the gaze point
                smooth_gaze_point = smooth_gaze(gaze_point, gaze_point_history)
                
                if smooth_gaze_point is not None:
                    # Record gaze point with timestamp
                    gaze_points_data.append((time.time(), smooth_gaze_point[0], smooth_gaze_point[1]))
                    
                    # Update fixation data
                    update_fixations(smooth_gaze_point)
                    
                    # Update heatmap
                    update_heatmap(smooth_gaze_point)
                    
                    # Draw gaze point
                    cv2.circle(display_frame, smooth_gaze_point, 10, (0, 0, 255), -1)
                    cv2.circle(display_frame, smooth_gaze_point, 4, (255, 255, 255), -1)
        
        # Overlay heatmap if enabled and available
        if DISPLAY_HEATMAP and heatmap_updated and calibrated:
            # Create an alpha mask from the heatmap
            alpha_mask = (heatmap > 0.05).reshape(frame_height, frame_width, 1).astype(np.float32) * 0.7
            
            # Blend heatmap with display frame
            display_frame = display_frame * (1 - alpha_mask) + heatmap_colormap * alpha_mask
            
            # Convert back to uint8
            display_frame = display_frame.astype(np.uint8)
        
        # Display calibration status
        if calibrated:
            cv2.putText(display_frame, "Calibrated - Press 'r' to recalibrate", 
                       (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show fixation info if available
            if last_fixation_point is not None and fixation_duration > 200:
                cv2.putText(display_frame, f"Fixation: {int(fixation_duration)}ms", 
                           (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show the frame
        cv2.imshow('Gaze Tracker', display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('h'):
            # Toggle heatmap display
            DISPLAY_HEATMAP = not DISPLAY_HEATMAP
        elif key == ord('s'):
            # Save heatmap
            save_heatmap()
            save_gaze_data()
        elif key == ord('r'):
            # Reset calibration
            calibration_mode = True
            calibration_points = []
            calibration_eye_positions = []
            calibrated = False
            heatmap = np.zeros((frame_height, frame_width), dtype=np.float32)
            heatmap_updated = False
        elif key == ord(' '):
            # Process calibration point
            if calibration_mode and left_pupil is not None and right_pupil is not None:
                current_point = get_calibration_point(len(calibration_points), frame_width, frame_height)
                
                # Average eye position
                avg_pupil_x = (left_pupil[0] + right_pupil[0]) / 2
                avg_pupil_y = (left_pupil[1] + right_pupil[1]) / 2
                
                # Store calibration data
                calibration_points.append(current_point)
                calibration_eye_positions.append((avg_pupil_x, avg_pupil_y))
                
                # Check if calibration is complete
                if len(calibration_points) >= CALIBRATION_POINTS:
                    calibration_mode = False
                    calibrated = True
                    print("Calibration complete!")

except KeyboardInterrupt:
    print("Stopping...")
finally:
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Save data before exit
    if heatmap_updated:
        save_heatmap()
    if len(gaze_points_data) > 0:
        save_gaze_data()