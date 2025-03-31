import cv2
import dlib
import numpy as np
from collections import deque
import time
import os
import datetime
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Configuration
FACE_DETECTOR_PATH = "../data/shape_predictor_68_face_landmarks.dat"
CALIBRATION_POINTS = 9  # Increased calibration points for better accuracy
HEATMAP_DECAY = 0.995
HEATMAP_INTENSITY = 0.2
GAZE_HISTORY_LENGTH = 10
DISPLAY_HEATMAP = True
RECORDING_DURATION = 180  # 3 minutes in seconds
OVERLAY_OPACITY = 0.2     # Lower opacity for less intrusive overlay
MINIMIZE_DURING_RECORDING = True  # Minimize window during recording

# Initialize face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(FACE_DETECTOR_PATH)

# Create analysis directory
output_dir = "../analysis"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Create a named window and start webcam
cv2.namedWindow('Gaze Analysis', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Gaze Analysis', cv2.WND_PROP_TOPMOST, 1)  # Keep window on top
cap = cv2.VideoCapture(0)

# Get initial frame dimensions
ret, frame = cap.read()
if not ret:
    print("Failed to capture initial frame")
    exit()

frame_height, frame_width = frame.shape[:2]

# Make window small and position in corner
corner_width = int(frame_width * 0.25)  # 25% of original size
corner_height = int(frame_height * 0.25)
cv2.resizeWindow('Gaze Analysis', corner_width, corner_height)

# Tracking variables
calibration_mode = True
calibration_points = []
calibration_eye_positions = []
calibrated = False
smooth_gaze_point = None
recording_start_time = None

# Eye tracking variables
left_pupil_history = deque(maxlen=3)
right_pupil_history = deque(maxlen=3)
gaze_point_history = deque(maxlen=GAZE_HISTORY_LENGTH)

# Heatmap creation
heatmap = np.zeros((frame_height, frame_width), dtype=np.float32)
heatmap_colormap = cv2.applyColorMap(np.zeros((frame_height, frame_width), dtype=np.uint8), cv2.COLORMAP_JET)
heatmap_updated = False

# Data collection for analysis
gaze_points_data = []
fixation_duration = 0
fixation_threshold = 30  # pixels
last_fixation_point = None
fixation_start_time = 0
fixations = []  # (x, y, duration, timestamp)

# Screen regions of interest (example)
# You can define specific screen regions for analysis
screen_regions = {
    "top_banner": (0, 0, frame_width, int(frame_height * 0.1)),
    "left_sidebar": (0, 0, int(frame_width * 0.2), frame_height),
    "main_content": (int(frame_width * 0.2), int(frame_height * 0.1), 
                     int(frame_width * 0.6), int(frame_height * 0.8)),
    "right_sidebar": (int(frame_width * 0.8), 0, int(frame_width * 0.2), frame_height)
}

# Region viewing statistics
region_views = {region: 0 for region in screen_regions}
region_dwell_times = {region: 0 for region in screen_regions}
last_region = None
region_entry_time = 0

# Eye aspect ratio threshold for blink detection
EYE_AR_THRESHOLD = 0.2

def eye_aspect_ratio(eye_points):
    """Calculate eye aspect ratio for blink detection"""
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    return (A + B) / (2.0 * C)

def get_calibration_point(index, width, height):
    """Return position for calibration point using a 3x3 grid for better accuracy"""
    grid_x = index % 3
    grid_y = index // 3
    
    x = int(width * (0.1 + 0.4 * grid_x))
    y = int(height * (0.1 + 0.4 * grid_y))
    
    return (x, y)

def draw_calibration_point(frame, point, active=True):
    """Draw calibration target point with animation"""
    # Get current time for animation
    t = time.time()
    pulse = (np.sin(t * 5) + 1) / 2  # Pulsing animation between 0 and 1
    
    # Outer circle
    color = (0, 0, 255) if active else (100, 100, 100)
    size = int(20 + 5 * pulse)
    cv2.circle(frame, point, size, color, 2)
    
    # Inner circle
    inner_size = int(5 + 2 * pulse)
    cv2.circle(frame, point, inner_size, (255, 255, 255), -1)
    
    return frame

def extract_eye(frame, landmarks, eye_points):
    """Extract eye region from facial landmarks"""
    eye_region = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in eye_points])
    
    min_x = np.min(eye_region[:, 0])
    max_x = np.max(eye_region[:, 0])
    min_y = np.min(eye_region[:, 1])
    max_y = np.max(eye_region[:, 1])
    
    padding = 5
    min_x = max(0, min_x - padding)
    max_x = min(frame.shape[1], max_x + padding)
    min_y = max(0, min_y - padding)
    max_y = min(frame.shape[0], max_y + padding)
    
    eye_frame = frame[min_y:max_y, min_x:max_x]
    
    return eye_frame, (min_x, min_y, max_x - min_x, max_y - min_y)

def detect_pupil(eye_frame):
    """Detect pupil in the eye frame with multiple fallback methods"""
    if eye_frame is None or eye_frame.size == 0:
        return None, None
    
    # Convert to grayscale
    if len(eye_frame.shape) == 3:
        eye_gray = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)
    else:
        eye_gray = eye_frame
    
    # Enhance contrast
    eye_gray = cv2.equalizeHist(eye_gray)
    blur = cv2.GaussianBlur(eye_gray, (5, 5), 0)
    
    # Try different threshold values for more robust detection
    for threshold in [40, 55, 70]:  # Multiple thresholds to handle different lighting
        _, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Filter contours by area and circularity
            filtered_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                if area > 10 and circularity > 0.6:  # Minimum area and roundness
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        filtered_contours.append((contour, cx, cy, area, circularity))
            
            if filtered_contours:
                # Sort by combination of size and circularity
                best_contour = max(filtered_contours, key=lambda x: x[3] * x[4])
                return best_contour[1], best_contour[2]
    
    # Try Hough circles as fallback
    try:
        circles = cv2.HoughCircles(
            blur, 
            cv2.HOUGH_GRADIENT, 
            dp=1.2, 
            minDist=blur.shape[0]//4,
            param1=50, 
            param2=25, 
            minRadius=3, 
            maxRadius=15
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            circle = circles[0, 0]
            return circle[0], circle[1]
    except:
        pass
    
    # Last resort - darkest point
    min_val, _, min_loc, _ = cv2.minMaxLoc(blur)
    return min_loc[0], min_loc[1]

def smooth_gaze(current_gaze, history):
    """Apply smoothing to gaze point with outlier rejection"""
    if current_gaze is None:
        return None
    
    # Check if current point is an outlier compared to history
    if len(history) >= 2:
        prev_x, prev_y = history[-1]
        distance = np.sqrt((current_gaze[0] - prev_x)**2 + (current_gaze[1] - prev_y)**2)
        
        # If point jumps too far, it might be an error
        if distance > 100:  # Threshold for detecting jumps
            # Only accept large jumps if confirmed by consecutive points
            return history[-1]  # Return previous position instead
    
    # Add to history
    history.append(current_gaze)
    
    if len(history) < 2:
        return current_gaze
    
    # Create exponential weights based on history length
    weights = [2**i for i in range(len(history))]
    
    # Apply weighted average
    total_x = 0
    total_y = 0
    total_weight = 0
    
    for i, (x, y) in enumerate(history):
        weight = weights[i]
        total_x += x * weight
        total_y += y * weight
        total_weight += weight
    
    # Return smoothed position
    return (int(total_x / total_weight), int(total_y / total_weight))

def map_eye_to_screen(left_pupil, right_pupil, screen_size):
    """Map eye positions to screen coordinates using calibration data"""
    if not calibrated or len(calibration_eye_positions) < 4 or left_pupil is None or right_pupil is None:
        return None
    
    # Use average of both eyes when available, or individual eye if one is missing
    if left_pupil is not None and right_pupil is not None:
        avg_pupil_x = (left_pupil[0] + right_pupil[0]) / 2
        avg_pupil_y = (left_pupil[1] + right_pupil[1]) / 2
    elif left_pupil is not None:
        avg_pupil_x, avg_pupil_y = left_pupil
    elif right_pupil is not None:
        avg_pupil_x, avg_pupil_y = right_pupil
    else:
        return None
    
    # Get min/max values from calibration data
    eye_x_values = [p[0] for p in calibration_eye_positions]
    eye_y_values = [p[1] for p in calibration_eye_positions]
    screen_x_values = [p[0] for p in calibration_points]
    screen_y_values = [p[1] for p in calibration_points]
    
    try:
        # Use polynomial regression for more accurate mapping
        # For simplicity we'll use linear here, but you can implement polynomial for better results
        x_min, x_max = min(eye_x_values), max(eye_x_values)
        y_min, y_max = min(eye_y_values), max(eye_y_values)
        
        # Apply bounds with a buffer to handle edge cases
        norm_x = (avg_pupil_x - x_min) / (x_max - x_min) if x_max > x_min else 0.5
        norm_y = (avg_pupil_y - y_min) / (y_max - y_min) if y_max > y_min else 0.5
        
        # Smooth the mapping with cubic interpolation - simplified here
        screen_x = int(norm_x * screen_size[0])
        screen_y = int(norm_y * screen_size[1])
        
        return (screen_x, screen_y)
    except:
        return None

def update_region_statistics(gaze_point):
    """Update statistics for screen regions"""
    global last_region, region_entry_time, region_views, region_dwell_times
    
    if gaze_point is None:
        return
    
    current_time = time.time()
    x, y = gaze_point
    
    # Determine which region contains the gaze point
    current_region = None
    for region_name, (rx, ry, rw, rh) in screen_regions.items():
        if rx <= x < rx + rw and ry <= y < ry + rh:
            current_region = region_name
            break
    
    # Update statistics
    if current_region:
        if last_region != current_region:
            # We've entered a new region
            if last_region:
                # Update dwell time for the previous region
                dwell_time = current_time - region_entry_time
                region_dwell_times[last_region] += dwell_time
            
            # Record entry to new region
            region_views[current_region] += 1
            region_entry_time = current_time
            last_region = current_region
    
    # If we're not in any region, reset last_region
    if not current_region and last_region:
        dwell_time = current_time - region_entry_time
        region_dwell_times[last_region] += dwell_time
        last_region = None

def update_heatmap(gaze_point):
    """Update the heatmap with a new gaze point"""
    global heatmap, heatmap_colormap, heatmap_updated
    
    if gaze_point is None:
        return
    
    # Create a temporary heatmap for the new point
    temp_heatmap = np.zeros_like(heatmap)
    
    # Draw a gaussian at the gaze point
    x, y = gaze_point
    sigma = 30  # Size of the gaussian blob (slightly smaller for precision)
    
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

def update_fixations(gaze_point):
    """Update fixation tracking with timestamps"""
    global last_fixation_point, fixation_start_time, fixation_duration, fixations
    
    current_time = time.time()
    
    if gaze_point is None:
        # If gaze point is lost, reset fixation
        if last_fixation_point is not None and fixation_duration > 100:
            # Save the previous fixation with timestamp
            fixations.append((last_fixation_point[0], last_fixation_point[1], 
                             fixation_duration, current_time - (fixation_duration/1000)))
        
        last_fixation_point = None
        fixation_duration = 0
        return
    
    current_time_ms = current_time * 1000  # Convert to milliseconds
    
    if last_fixation_point is None:
        # Start a new fixation
        last_fixation_point = gaze_point
        fixation_start_time = current_time_ms
        fixation_duration = 0
    else:
        # Check if still within fixation threshold
        dist = np.sqrt((gaze_point[0] - last_fixation_point[0])**2 + 
                      (gaze_point[1] - last_fixation_point[1])**2)
        
        if dist < fixation_threshold:
            # Update fixation duration
            fixation_duration = current_time_ms - fixation_start_time
        else:
            # Moved outside threshold, save fixation if it was long enough
            if fixation_duration > 100:  # Only save fixations longer than 100ms
                fixations.append((last_fixation_point[0], last_fixation_point[1], 
                                 fixation_duration, current_time - (fixation_duration/1000)))
            
            # Start new fixation
            last_fixation_point = gaze_point
            fixation_start_time = current_time_ms
            fixation_duration = 0

def save_analysis_results():
    """Save all analysis data including heatmap, fixations, and region statistics"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Create session directory
    session_dir = os.path.join(output_dir, f"session_{timestamp}")
    os.makedirs(session_dir, exist_ok=True)
    
    # Save heatmap
    if heatmap_updated:
        plt.figure(figsize=(12, 8))
        plt.imshow(heatmap, cmap='jet')
        plt.colorbar(label='Gaze intensity')
        plt.title('Gaze Heatmap')
        plt.axis('off')
        plt.savefig(os.path.join(session_dir, "heatmap.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also save the raw heatmap data
        np.save(os.path.join(session_dir, "heatmap_data.npy"), heatmap)
    
    # Save gaze points
    with open(os.path.join(session_dir, "gaze_points.csv"), 'w') as f:
        f.write("timestamp,x,y\n")
        for point in gaze_points_data:
            f.write(f"{point[0]},{point[1]},{point[2]}\n")
    
    # Save fixations
    with open(os.path.join(session_dir, "fixations.csv"), 'w') as f:
        f.write("x,y,duration_ms,timestamp\n")
        for fix in fixations:
            f.write(f"{fix[0]},{fix[1]},{fix[2]},{fix[3]}\n")
    
    # Save region statistics
    with open(os.path.join(session_dir, "region_statistics.csv"), 'w') as f:
        f.write("region,views,dwell_time_seconds\n")
        for region in region_views.keys():
            f.write(f"{region},{region_views[region]},{region_dwell_times[region]:.2f}\n")
    
    # Generate summary report
    with open(os.path.join(session_dir, "summary_report.txt"), 'w') as f:
        f.write("GAZE ANALYSIS SUMMARY\n")
        f.write("=====================\n\n")
        f.write(f"Session Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Recording Duration: {RECORDING_DURATION} seconds\n\n")
        
        f.write("FIXATION ANALYSIS\n")
        f.write("----------------\n")
        total_fixations = len(fixations)
        avg_duration = np.mean([fix[2] for fix in fixations]) if fixations else 0
        f.write(f"Total Fixations: {total_fixations}\n")
        f.write(f"Average Fixation Duration: {avg_duration:.2f} ms\n\n")
        
        f.write("REGION ANALYSIS\n")
        f.write("--------------\n")
        total_views = sum(region_views.values())
        total_dwell = sum(region_dwell_times.values())
        
        for region, views in region_views.items():
            dwell = region_dwell_times[region]
            view_pct = (views / total_views * 100) if total_views > 0 else 0
            dwell_pct = (dwell / total_dwell * 100) if total_dwell > 0 else 0
            
            f.write(f"Region: {region}\n")
            f.write(f"  Views: {views} ({view_pct:.1f}%)\n")
            f.write(f"  Dwell Time: {dwell:.2f} seconds ({dwell_pct:.1f}%)\n\n")
    
    # Create visualizations
    
    # Fixation map
    plt.figure(figsize=(12, 8))
    plt.scatter([f[0] for f in fixations], [f[1] for f in fixations], 
               s=[f[2]/10 for f in fixations], alpha=0.5, c=[f[2] for f in fixations], 
               cmap='viridis')
    plt.colorbar(label='Fixation Duration (ms)')
    plt.xlim(0, frame_width)
    plt.ylim(frame_height, 0)  # Invert Y axis to match image coordinates
    plt.title('Fixation Map')
    plt.savefig(os.path.join(session_dir, "fixation_map.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Region analysis pie charts
    plt.figure(figsize=(16, 6))
    
    plt.subplot(1, 2, 1)
    plt.pie([region_views[r] for r in region_views], labels=[r for r in region_views], 
           autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Views by Region')
    
    plt.subplot(1, 2, 2)
    plt.pie([region_dwell_times[r] for r in region_dwell_times], labels=[r for r in region_dwell_times], 
           autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Dwell Time by Region')
    
    plt.tight_layout()
    plt.savefig(os.path.join(session_dir, "region_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Analysis saved to {session_dir}")
    return session_dir

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
        
        # Check recording time limit
        current_time = time.time()
        if calibrated and recording_start_time is not None:
            if MINIMIZE_DURING_RECORDING:
                # Make window truly minimal during actual recording
                cv2.resizeWindow('Gaze Analysis', 150, 100)
                cv2.moveWindow('Gaze Analysis', 0, 0)  # Move to top-left corner
        
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
            left_eye_points = range(36, 42)
            right_eye_points = range(42, 48)
            
            # Get eye landmarks for blink detection
            left_eye_landmarks = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in left_eye_points])
            right_eye_landmarks = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in right_eye_points])
            
            # Check if eyes are open
            left_ear = eye_aspect_ratio(left_eye_landmarks)
            right_ear = eye_aspect_ratio(right_eye_landmarks)
            
            left_eye_open = left_ear > EYE_AR_THRESHOLD
            right_eye_open = right_ear > EYE_AR_THRESHOLD
            
            # Draw eye outlines (only in calibration mode)
            if calibration_mode:
                cv2.polylines(display_frame, [left_eye_landmarks], True, (0, 255, 0), 1)
                cv2.polylines(display_frame, [right_eye_landmarks], True, (0, 255, 0), 1)
            
            # Process eyes if open
            if left_eye_open:
                left_eye_img, left_eye_rect = extract_eye(frame, landmarks, left_eye_points)
                
                if left_eye_img is not None and left_eye_img.size > 0:
                    left_pupil_pos_x, left_pupil_pos_y = detect_pupil(left_eye_img)
                    if left_pupil_pos_x is not None:
                        left_pupil_x = left_eye_rect[0] + left_pupil_pos_x
                        left_pupil_y = left_eye_rect[1] + left_pupil_pos_y
                        left_pupil = (left_pupil_x, left_pupil_y)
                        
                        # Draw pupil during calibration
                        if calibration_mode:
                            cv2.circle(display_frame, left_pupil, 3, (0, 0, 255), -1)
            
            if right_eye_open:
                right_eye_img, right_eye_rect = extract_eye(frame, landmarks, right_eye_points)
                
                if right_eye_img is not None and right_eye_img.size > 0:
                    right_pupil_pos_x, right_pupil_pos_y = detect_pupil(right_eye_img)
                    if right_pupil_pos_x is not None:
                        right_pupil_x = right_eye_rect[0] + right_pupil_pos_x
                        right_pupil_y = right_eye_rect[1] + right_pupil_pos_y
                        right_pupil = (right_pupil_x, right_pupil_y)
                        
                        # Draw pupil during calibration
                        if calibration_mode:
                            cv2.circle(display_frame, right_pupil, 3, (0, 0, 255), -1)
        
        # Calibration and gaze mapping logic
        if calibration_mode and len(calibration_points) < CALIBRATION_POINTS:
            # Show calibration point
            current_calib_point = get_calibration_point(len(calibration_points), frame_width, frame_height)
            display_frame = draw_calibration_point(display_frame, current_calib_point)
            
            # Display instructions
            cv2.putText(display_frame, "Look at the red circle and press SPACE", 
                       (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(display_frame, f"Calibration: {len(calibration_points)}/{CALIBRATION_POINTS}", 
                       (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Show full screen during calibration
            cv2.setWindowProperty('Gaze Analysis', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            # After calibration, switch to small corner window
            if calibrated:
                # Exit fullscreen after calibration
                cv2.setWindowProperty('Gaze Analysis', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Gaze Analysis', corner_width, corner_height)
                
                # If calibration complete, map gaze to screen
                if left_pupil is not None or right_pupil is not None:
                    # Map pupil positions to screen coordinates
                    gaze_point = map_eye_to_screen(left_pupil, right_pupil, (frame_width, frame_height))
                    
                    # Smooth the gaze point
                    smooth_gaze_point = smooth_gaze(gaze_point, gaze_point_history)
                    
                    if smooth_gaze_point is not None:
                        # Record gaze point with timestamp
                        gaze_points_data.append((current_time, smooth_gaze_point[0], smooth_gaze_point[1]))
                        
                        # Update fixation data
                        update_fixations(smooth_gaze_point)
                        
                        # Update heatmap
                        update_heatmap(smooth_gaze_point)
                        
                        # Update region statistics
                        update_region_statistics(smooth_gaze_point)
                        
                        # Draw gaze point (small indicator in corner window)
                        cv2.circle(display_frame, smooth_gaze_point, 5, (0, 0, 255), -1)
                
                # Display recording progress
                elapsed = current_time - recording_start_time
                remaining = max(0, RECORDING_DURATION - elapsed)
                progress = int((elapsed / RECORDING_DURATION) * 100)
                
                cv2.putText(display_frame, f"Recording: {int(elapsed)}s / {RECORDING_DURATION}s", 
                           (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(display_frame, f"Progress: {progress}%", 
                           (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Draw progress bar
                bar_width = display_frame.shape[1] - 20
                filled_width = int(bar_width * progress / 100)
                cv2.rectangle(display_frame, (10, 50), (10 + bar_width, 60), (0, 100, 0), 1)
                cv2.rectangle(display_frame, (10, 50), (10 + filled_width, 60), (0, 255, 0), -1)
            else:
                # Waiting for calibration to complete
                cv2.putText(display_frame, "Calibration complete! Press ENTER to start recording", 
                           (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, "Recording will run for 3 minutes", 
                           (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show the frame
        cv2.imshow('Gaze Analysis', display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):
            # Process calibration point
            if calibration_mode and (left_pupil is not None or right_pupil is not None):
                current_point = get_calibration_point(len(calibration_points), frame_width, frame_height)
                
                # Use average of available pupils
                if left_pupil is not None and right_pupil is not None:
                    avg_pupil_x = (left_pupil[0] + right_pupil[0]) / 2
                    avg_pupil_y = (left_pupil[1] + right_pupil[1]) / 2
                elif left_pupil is not None:
                    avg_pupil_x, avg_pupil_y = left_pupil
                else:
                    avg_pupil_x, avg_pupil_y = right_pupil
                
                # Store calibration data
                calibration_points.append(current_point)
                calibration_eye_positions.append((avg_pupil_x, avg_pupil_y))
                
                # Check if calibration is complete
                if len(calibration_points) >= CALIBRATION_POINTS:
                    calibration_mode = False
                    calibrated = True
                    print("Calibration complete! Press ENTER to start recording.")
        elif key == 13:  # ENTER key
            # Start recording after calibration
            if calibrated and recording_start_time is None:  # Changed from 0 to None
                recording_start_time = time.time()
                print(f"Recording started for {RECORDING_DURATION} seconds...")

except KeyboardInterrupt:
    print("Stopping...")
finally:
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Save analysis data
    if calibrated and len(gaze_points_data) > 0:
        session_dir = save_analysis_results()
        
        # Open results folder
        try:
            os.startfile(session_dir)  # Windows-specific
        except:
            print(f"Analysis saved to {session_dir}")  