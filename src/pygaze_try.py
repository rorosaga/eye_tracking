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
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import io
from PIL import Image
import random

# Configuration
FACE_DETECTOR_PATH = "../data/shape_predictor_68_face_landmarks.dat"
CALIBRATION_POINTS = 5  # Number of calibration points
HEATMAP_DECAY = 0.995  # Decay factor for heatmap (higher = longer persistence)
HEATMAP_INTENSITY = 0.2  # Intensity of new gaze points
GAZE_HISTORY_LENGTH = 10  # Length of gaze history for smoothing
DISPLAY_HEATMAP = True  # Toggle heatmap display
FLIP_HORIZONTAL = True  # Flip video horizontally to correct mirror effect
FULLSCREEN = True  # Enable maximized window mode (not true fullscreen)
USE_BROWSER = True  # Enable browser integration with Selenium automatically
AUTO_LAUNCH_BROWSER = True  # Automatically launch browser after calibration
BROWSER_URL = "https://www.adidas.es/zapatillas-hombre"  # Default URL to open
ENABLE_LIVE_HEATMAP = False  # Disable live heatmap to avoid blue tint issues

# Selenium/Browser variables
browser = None
browser_position = (0, 0)
browser_size = (0, 0)
scroll_position = 0
webpage_height = 0
webpage_width = 0
web_gaze_points = []  # Store gaze points relative to webpage (x, y, timestamp)
full_webpage_image = None  # Store the full webpage image captured at start

# Initialize face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(FACE_DETECTOR_PATH)

# Create a named window and start webcam
cv2.namedWindow('Gaze Tracker', cv2.WINDOW_NORMAL)
if FULLSCREEN:
    # Set window to be maximized but not true fullscreen
    cv2.setWindowProperty('Gaze Tracker', cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_AUTOSIZE)
    # Get screen resolution for maximizing window
    screen_width = cv2.getWindowImageRect('Gaze Tracker')[2]
    screen_height = cv2.getWindowImageRect('Gaze Tracker')[3]
    # Resize window to match screen size
    cv2.resizeWindow('Gaze Tracker', screen_width, screen_height)
cap = cv2.VideoCapture(0)

# Get initial frame dimensions
ret, frame = cap.read()
if not ret:
    print("Failed to capture initial frame")
    exit()

# Flip the frame horizontally if enabled
if FLIP_HORIZONTAL:
    frame = cv2.flip(frame, 1)  # 1 = horizontal flip

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

def start_browser_tracking(url=BROWSER_URL):
    """Start browser with Selenium for gaze tracking on a webpage"""
    global browser, browser_position, browser_size, USE_BROWSER
    global webpage_height, webpage_width, scroll_position, web_gaze_points
    global full_webpage_image  # Store the full webpage image
    
    # Reset gaze points when starting a new browser session
    web_gaze_points = []
    
    print("Starting browser for eye tracking...")
    
    # Configure Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--disable-infobars")
    chrome_options.add_argument("--disable-extensions")
    
    # Initialize browser
    browser = webdriver.Chrome(options=chrome_options)
    
    # Initialize scroll position
    scroll_position = 0
    
    # Navigate to URL
    browser.get(url)
    
    # Add disclaimer overlay immediately
    add_disclaimer_overlay()
    print("Added disclaimer overlay - page will be captured after accepting cookies")
    
    # Accept cookies using the specific button ID provided by the user
    try:
        print("Looking for the specific cookie consent button...")
        
        # First try the specific button ID the user provided
        try:
            # Look for the specific button by ID
            cookie_button = WebDriverWait(browser, 5).until(
                EC.element_to_be_clickable((By.ID, "glass-gdpr-default-consent-accept-button"))
            )
            
            if cookie_button:
                print("Found cookie button with specific ID")
                cookie_button.click()
                print("Clicked specific cookie button")
                time.sleep(2)  # Wait for dialog to close
        except Exception as e:
            print(f"Specific button not found: {e}, trying alternative methods")
            
            # If specific button fails, try text content
            try:
                cookie_button = WebDriverWait(browser, 3).until(
                    EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Aceptar seguimiento')]"))
                )
                
                if cookie_button:
                    print("Found cookie button by text content")
                    cookie_button.click()
                    print("Clicked cookie button by text")
                    time.sleep(2)
            except:
                # Fall back to previous methods
                cookie_patterns = [
                    "//button[contains(text(), 'Accept')]",
                    "//button[contains(text(), 'Aceptar')]",
                    "//button[contains(@class, 'cookie')]",
                    "//button[contains(@id, 'cookie')]",
                    "//div[contains(@class, 'cookie')]//button",
                    "//div[contains(@id, 'cookie')]//button",
                    "//a[contains(text(), 'Accept')]"
                ]
                
                for pattern in cookie_patterns:
                    try:
                        buttons = WebDriverWait(browser, 2).until(
                            EC.presence_of_all_elements_located((By.XPATH, pattern))
                        )
                        
                        for button in buttons:
                            if button.is_displayed():
                                print(f"Found cookie button with pattern: {pattern}")
                                button.click()
                                print("Clicked cookie button")
                                time.sleep(1)  # Wait for dialog to close
                                break
                        
                        # Break the outer loop if we clicked a button
                        if buttons:
                            break
                    except:
                        continue
    except:
        print("No cookie dialog found or couldn't be clicked.")
    
    # Wait for page to fully load
    time.sleep(2)
    
    # Get browser window position and size
    browser_position = (browser.get_window_position()['x'], browser.get_window_position()['y'])
    browser_size = (browser.get_window_size()['width'], browser.get_window_size()['height'])
    
    # Get actual webpage dimensions
    webpage_height = browser.execute_script("return Math.max(document.body.scrollHeight, document.documentElement.scrollHeight)")
    webpage_width = browser.execute_script("return Math.max(document.body.scrollWidth, document.documentElement.scrollWidth)")
    
    print(f"Browser started at position {browser_position}, size {browser_size}")
    print(f"Webpage dimensions: {webpage_width}x{webpage_height}")
    
    # Update the disclaimer to inform user about the scrolling process
    update_disclaimer_for_scrolling()
    
    # Capture the full webpage at the beginning
    print("Capturing full webpage screenshot immediately...")
    print("Please wait as the browser automatically scrolls to capture the entire page...")
    full_webpage_image = capture_full_webpage()
    
    if full_webpage_image is not None:
        print("Full webpage captured successfully! Eye tracking will map to this image.")
        # Save the initial capture
        try:
            if not os.path.exists("../output"):
                os.makedirs("../output")
                
            full_webpage_image.save("../output/output_image.png")
            print("Saved initial webpage screenshot to ../output/output_image.png")
        except Exception as e:
            print(f"Error saving initial screenshot: {e}")
            try:
                if not os.path.exists("./output"):
                    os.makedirs("./output")
                    
                full_webpage_image.save("./output/output_image.png")
                print("Saved initial webpage screenshot to ./output/output_image.png")
            except Exception as e2:
                print(f"Error saving to alternative location: {e2}")
    else:
        print("WARNING: Failed to capture full webpage. Eye tracking may not be accurate.")
    
    # Set flag to enable browser integration
    USE_BROWSER = True
    
    # Start monitoring scroll position in a background thread
    threading.Thread(target=monitor_scroll_position, daemon=True).start()
    
    # Update the disclaimer to allow the user to close it
    update_disclaimer_after_scrolling()
    
    return browser

def add_disclaimer_overlay():
    """Add a disclaimer overlay to the browser window"""
    global browser
    
    if browser is None:
        return
    
    try:
        # Create an overlay with disclaimer text that blocks interaction
        browser.execute_script("""
            // Create overlay div if it doesn't exist
            var overlay = document.createElement('div');
            overlay.id = 'eye_tracking_disclaimer';
            
            // Style the overlay
            overlay.style.position = 'fixed';
            overlay.style.top = '0';
            overlay.style.left = '0';
            overlay.style.width = '100%';
            overlay.style.height = '100%';
            overlay.style.backgroundColor = 'rgba(0, 0, 0, 0.8)';
            overlay.style.color = 'white';
            overlay.style.zIndex = '10000';
            overlay.style.display = 'flex';
            overlay.style.flexDirection = 'column';
            overlay.style.justifyContent = 'center';
            overlay.style.alignItems = 'center';
            overlay.style.textAlign = 'center';
            overlay.style.fontFamily = 'Arial, sans-serif';
            overlay.style.fontSize = '18px';
            overlay.style.padding = '20px';
            overlay.style.boxSizing = 'border-box';
            
            // Add disclaimer text
            overlay.innerHTML = `
                <h1 style="color: #FF5555; margin-bottom: 20px; font-size: 24px;">Eye Tracking Session in Progress</h1>
                <p style="margin-bottom: 15px; line-height: 1.5;">This program is recording where you look on this webpage.</p>
                <p style="margin-bottom: 15px; line-height: 1.5;">We need to accept cookies first before capturing begins.</p>
                <p style="margin-bottom: 15px; line-height: 1.5;">Please wait while we set up the tracking session.</p>
                <p style="color: #AAFFAA; margin-top: 20px;">Do not close this window.</p>
            `;
            
            // Append to body
            document.body.appendChild(overlay);
        """)
        
        print("Added initial disclaimer overlay to browser")
    except Exception as e:
        print(f"Error adding disclaimer overlay: {e}")

def update_disclaimer_for_scrolling():
    """Update the disclaimer to show scrolling status"""
    global browser
    
    if browser is None:
        return
    
    try:
        browser.execute_script("""
            var overlay = document.getElementById('eye_tracking_disclaimer');
            if (overlay) {
                overlay.innerHTML = `
                    <h1 style="color: #FF5555; margin-bottom: 20px; font-size: 24px;">Capturing Website</h1>
                    <p style="margin-bottom: 15px; line-height: 1.5;">The browser is automatically scrolling to capture the entire webpage.</p>
                    <p style="margin-bottom: 15px; line-height: 1.5;">This will take a few moments. Please wait...</p>
                    <div style="width: 60px; height: 60px; margin: 20px auto; border: 5px solid #f3f3f3; border-top: 5px solid #FF5555; border-radius: 50%; animation: spin 1s linear infinite;"></div>
                    <style>
                        @keyframes spin {
                            0% { transform: rotate(0deg); }
                            100% { transform: rotate(360deg); }
                        }
                    </style>
                `;
            }
        """)
        print("Updated disclaimer for scrolling phase")
    except Exception as e:
        print(f"Error updating disclaimer: {e}")

def update_disclaimer_after_scrolling():
    """Update the disclaimer after scrolling to allow closing"""
    global browser
    
    if browser is None:
        return
    
    try:
        browser.execute_script("""
            var overlay = document.getElementById('eye_tracking_disclaimer');
            if (overlay) {
                overlay.innerHTML = `
                    <h1 style="color: #FF5555; margin-bottom: 20px; font-size: 24px;">Eye Tracking Session in Progress</h1>
                    <p style="margin-bottom: 15px; line-height: 1.5;">This program is recording where you look on this webpage.</p>
                    <p style="margin-bottom: 15px; line-height: 1.5;">A heatmap of your eye movements will be generated.</p>
                    <p style="margin-bottom: 15px; line-height: 1.5;">The website has been captured successfully.</p>
                    <p style="margin-bottom: 15px; line-height: 1.5;">You can view the website through this overlay but cannot interact with it.</p>
                    <button id="close_disclaimer" style="padding: 10px 20px; background-color: #FF5555; color: white; border: none; border-radius: 5px; font-size: 16px; cursor: pointer; margin-top: 20px;">Close This Message</button>
                    <p style="color: #AAFFAA; margin-top: 20px;">Close the eye tracker window when finished.</p>
                `;
                
                // Add event listener for the close button
                document.getElementById('close_disclaimer').addEventListener('click', function() {
                    document.getElementById('eye_tracking_disclaimer').remove();
                });
            }
        """)
        print("Updated disclaimer with close button")
    except Exception as e:
        print(f"Error updating disclaimer after scrolling: {e}")

def remove_disclaimer_overlay():
    """Remove the disclaimer overlay from the browser window"""
    global browser
    
    if browser is None:
        return
    
    try:
        browser.execute_script("""
            var overlay = document.getElementById('eye_tracking_disclaimer');
            if (overlay) {
                overlay.remove();
            }
        """)
        print("Removed disclaimer overlay from browser")
    except Exception as e:
        print(f"Error removing disclaimer overlay: {e}")

def monitor_scroll_position():
    """Monitor scroll position in browser"""
    global scroll_position, browser
    
    while browser is not None:
        try:
            scroll_position = browser.execute_script("return window.pageYOffset")
            time.sleep(0.1)  # Check 10 times per second
        except:
            # Browser probably closed
            break

def overlay_heatmap_on_browser():
    """Overlays a semi-transparent heatmap on the browser in real-time"""
    global browser, web_gaze_points, webpage_height, webpage_width
    
    # Wait a bit for the browser to be fully initialized
    time.sleep(1)
    
    # Track when we last updated the heatmap
    last_update_time = 0
    update_interval = 0.3  # Update every 300ms for better performance
    
    # Store a reference to the base64 heatmap to avoid unnecessary updates
    last_heatmap_data = None
    
    while browser is not None:
        try:
            current_time = time.time()
            
            # Only update if we have gaze data and enough time has passed
            if (web_gaze_points and 
                len(web_gaze_points) > 5 and 
                current_time - last_update_time > update_interval):
                
                # Create a simplified heatmap for injection
                heatmap_canvas = np.zeros((webpage_height, webpage_width), dtype=np.float32)
                
                # Get viewport dimensions
                viewport_height = browser.execute_script("return window.innerHeight")
                viewport_width = browser.execute_script("return window.innerWidth")
                current_scroll = scroll_position if scroll_position is not None else 0
                
                # Focus on points that would be visible in the current viewport
                # and recent points (last 50 for better performance)
                visible_points = []
                for x, y, ts in web_gaze_points[-200:]:  # Consider last 200 points max
                    # Check if the point is in or near the current viewport
                    if 0 <= x < viewport_width and current_scroll - 100 <= y <= current_scroll + viewport_height + 100:
                        visible_points.append((x, y, ts))
                
                # Use more recent points with higher intensity
                for i, (x, y, _) in enumerate(visible_points[-50:]):
                    if 0 <= x < webpage_width and 0 <= y < webpage_height:
                        # More recent points are brighter
                        intensity = 0.3 + (i / 50) * 0.7
                        # Draw a simple blob, bigger for more recent points
                        size = 30 + int((i / 50) * 20)
                        cv2.circle(heatmap_canvas, (int(x), int(y)), size, intensity, -1)
                
                # Apply gaussian blur for smoother heatmap
                heatmap_canvas = cv2.GaussianBlur(heatmap_canvas, (31, 31), 15)
                
                # Normalize and convert to colors
                heatmap_canvas = np.clip(heatmap_canvas, 0, 1)
                heatmap_bytes = cv2.imencode('.png', 
                                           cv2.applyColorMap(np.uint8(heatmap_canvas * 255), 
                                                           cv2.COLORMAP_JET))[1].tobytes()
                
                # Convert to base64 for inline display
                import base64
                heatmap_b64 = base64.b64encode(heatmap_bytes).decode('utf-8')
                
                # Only update if the heatmap has changed
                if heatmap_b64 != last_heatmap_data:
                    # Inject overlay div if it doesn't exist yet, or update it
                    browser.execute_script("""
                        var heatmapOverlay = document.getElementById('gaze_heatmap_overlay');
                        if (!heatmapOverlay) {
                            // Create overlay if it doesn't exist
                            heatmapOverlay = document.createElement('div');
                            heatmapOverlay.id = 'gaze_heatmap_overlay';
                            heatmapOverlay.style.position = 'fixed';  // Fixed to viewport instead of absolute
                            heatmapOverlay.style.top = '0';
                            heatmapOverlay.style.left = '0';
                            heatmapOverlay.style.width = '100%';
                            heatmapOverlay.style.height = '100%';  // Cover viewport only
                            heatmapOverlay.style.pointerEvents = 'none';
                            heatmapOverlay.style.zIndex = '9999';
                            heatmapOverlay.style.opacity = '0.6';
                            
                            // Create image element
                            var img = document.createElement('img');
                            img.id = 'gaze_heatmap_img';
                            img.style.width = '100%';
                            img.style.height = '100%';
                            img.style.objectFit = 'cover';
                            heatmapOverlay.appendChild(img);
                            
                            document.body.appendChild(heatmapOverlay);
                        }
                        
                        // Update the image source
                        document.getElementById('gaze_heatmap_img').src = 'data:image/png;base64,""" + heatmap_b64 + """';
                    """)
                    
                    # Store for comparison
                    last_heatmap_data = heatmap_b64
                    
                # Update timestamp
                last_update_time = current_time
            
            # Sleep a short time to avoid high CPU usage
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error updating browser heatmap: {e}")
            # Don't break on errors, just try again
            time.sleep(0.5)

def map_gaze_to_webpage(gaze_point):
    """Map gaze point on screen to position on webpage"""
    global browser_position, browser_size, scroll_position
    
    if gaze_point is None or browser is None or scroll_position is None:
        return None
    
    # Check if gaze is within browser window
    x, y = gaze_point
    browser_x, browser_y = browser_position
    browser_w, browser_h = browser_size
    
    # Adjust for window decoration (approximately)
    browser_header_height = 80  # Approximate height of browser header
    
    # X-axis correction factor to fix leftward shift (adjust as needed)
    x_correction = 25  # Pixels to shift right - adjust this value based on testing
    
    # Check if gaze is in browser content area
    if (browser_x <= x <= browser_x + browser_w and 
        browser_y + browser_header_height <= y <= browser_y + browser_h):
        
        # Calculate position relative to webpage (including scroll)
        web_x = x - browser_x + x_correction  # Add correction to fix leftward shift
        web_y = y - (browser_y + browser_header_height) + scroll_position
        
        return (web_x, web_y)
    
    return None

def capture_full_webpage():
    """Capture full webpage screenshot"""
    global browser, webpage_height, webpage_width
    
    if browser is None:
        print("Browser not active, can't capture webpage")
        return None
    
    print("Capturing full webpage screenshot...")
    print(f"Website dimensions: {webpage_width}x{webpage_height}")
    
    try:
        # Temporarily hide the disclaimer overlay for clean screenshots
        browser.execute_script("""
            var overlay = document.getElementById('eye_tracking_disclaimer');
            if (overlay) {
                overlay.style.display = 'none';
            }
        """)
        
        # Sanity check webpage dimensions - use defaults if problematic
        if webpage_height <= 0 or webpage_height > 50000:
            print(f"Invalid webpage height detected: {webpage_height}, using fallback value")
            webpage_height = browser.execute_script("return window.innerHeight * 5")  # Rough estimate
        
        if webpage_width <= 0 or webpage_width > 10000:
            print(f"Invalid webpage width detected: {webpage_width}, using fallback value")
            webpage_width = browser.execute_script("return window.innerWidth")
    
        # Create a canvas with the full webpage dimensions
        full_screenshot = Image.new('RGB', (webpage_width, webpage_height))
        
        # Get original scroll position to restore later
        original_scroll = browser.execute_script("return window.pageYOffset")
        
        # Remove any blue tint or overlays that might be present
        browser.execute_script("""
            var overlay = document.getElementById('gaze_heatmap_overlay');
            if (overlay) overlay.remove();
            
            // Remove any style elements that might cause blue tint
            var styles = document.querySelectorAll('style');
            for (var i = 0; i < styles.length; i++) {
                if (styles[i].innerHTML.indexOf('blue') !== -1 || 
                    styles[i].innerHTML.indexOf('rgba(0,0,255') !== -1 ||
                    styles[i].innerHTML.indexOf('overlay') !== -1) {
                    styles[i].remove();
                }
            }
        """)
        
        # Wait a moment for the page to settle
        time.sleep(0.5)
        
        # Scroll through the page and take screenshots
        viewport_height = browser.execute_script("return window.innerHeight")
        
        print(f"Capturing screenshot in chunks (viewport height: {viewport_height}px)")
        
        # Use larger steps for very tall pages but ensure some overlap for proper stitching
        step_size = max(viewport_height - 100, viewport_height // 2)
        max_scroll_attempts = min(30, webpage_height // step_size + 2)  # Limit max scroll attempts
        
        for i, scroll_y in enumerate(range(0, webpage_height, step_size)):
            if i >= max_scroll_attempts:
                print(f"Reached maximum scroll attempts ({max_scroll_attempts}), stopping capture")
                break
                
            # Scroll to position
            browser.execute_script(f"window.scrollTo(0, {scroll_y})")
            time.sleep(0.3)  # Wait longer for rendering and any lazy-loaded content
            
            # Get screenshot
            try:
                screenshot = browser.get_screenshot_as_png()
                screenshot = Image.open(io.BytesIO(screenshot))
                
                # Determine vertical position in the final image
                current_scroll = browser.execute_script("return window.pageYOffset")
                
                print(f"  - Captured section at scroll position {current_scroll}")
                
                # Paste into the full screenshot
                full_screenshot.paste(screenshot, (0, current_scroll))
            except Exception as section_error:
                print(f"Error capturing section at scroll position {scroll_y}: {section_error}")
                # Continue to next section rather than aborting
                continue
        
        try:
            # Restore original scroll position
            browser.execute_script(f"window.scrollTo(0, {original_scroll})")
        except:
            # Not critical if this fails
            pass
            
        # Restore the disclaimer overlay with updated content
        browser.execute_script("""
            var overlay = document.getElementById('eye_tracking_disclaimer');
            if (overlay) {
                overlay.style.display = 'flex';
            }
        """)
        
        print("Full webpage capture complete")
        
        return full_screenshot
        
    except Exception as e:
        print(f"Error capturing full webpage: {e}")
        import traceback
        traceback.print_exc()
        
        # Make sure to restore the disclaimer overlay even if there's an error
        try:
            browser.execute_script("""
                var overlay = document.getElementById('eye_tracking_disclaimer');
                if (overlay) {
                    overlay.style.display = 'flex';
                }
            """)
        except:
            pass
            
        return None

def generate_webpage_heatmap():
    """Generate heatmap overlay for the full webpage"""
    global web_gaze_points, webpage_height, webpage_width, full_webpage_image
    
    if not web_gaze_points:
        print("No gaze data collected for webpage")
        return None
    
    if full_webpage_image is None:
        print("No webpage image was captured, cannot generate heatmap")
        return None
    
    # Get image dimensions from the captured image
    img_width, img_height = full_webpage_image.size
    
    # Create heatmap canvas matching the full webpage image
    heatmap_canvas = np.zeros((img_height, img_width), dtype=np.float32)
    
    print(f"Generating heatmap with dimensions {img_width}x{img_height} from {len(web_gaze_points)} gaze points")
    
    # Add each gaze point to the heatmap
    for x, y, _ in web_gaze_points:
        # Make sure coordinates are within bounds
        if 0 <= x < img_width and 0 <= y < img_height:
            # Create a gaussian blob around the gaze point
            sigma = 50  # Size of gaussian blob
            y_pos = min(max(0, int(y)), img_height-1)
            x_pos = min(max(0, int(x)), img_width-1)
            
            # Debug print for coordinates
            if len(web_gaze_points) < 10 or random.random() < 0.01:  # Print first few or random 1%
                print(f"Mapped gaze point at x={x_pos}, y={y_pos}")
            
            # We'll use a simpler approach since the webpage can be very large
            # Create a smaller gaussian and paste it onto the large canvas
            size = sigma * 4 + 1
            small_gaussian = np.zeros((size, size), dtype=np.float32)
            center = (size // 2, size // 2)
            cv2.circle(small_gaussian, center, sigma, 1.0, -1)
            small_gaussian = cv2.GaussianBlur(small_gaussian, (size, size), sigma/2)
            
            # Normalize the small gaussian
            if np.max(small_gaussian) > 0:
                small_gaussian = small_gaussian / np.max(small_gaussian)
            
            # Determine paste region
            y_start = max(0, y_pos - size//2)
            y_end = min(img_height, y_pos + size//2 + 1)
            x_start = max(0, x_pos - size//2)
            x_end = min(img_width, x_pos + size//2 + 1)
            
            # Determine the section of the small gaussian to use
            small_y_start = max(0, size//2 - y_pos)
            small_y_end = small_y_start + (y_end - y_start)
            small_x_start = max(0, size//2 - x_pos)
            small_x_end = small_x_start + (x_end - x_start)
            
            # Add to heatmap
            try:
                heatmap_slice = small_gaussian[small_y_start:small_y_end, small_x_start:small_x_end]
                heatmap_canvas[y_start:y_end, x_start:x_end] += heatmap_slice * HEATMAP_INTENSITY
            except:
                # Skip this point if there's an error (boundary issues)
                pass
    
    # Normalize the heatmap
    heatmap_canvas = np.clip(heatmap_canvas, 0, 1)
    
    # Convert to colormap
    heatmap_colored = cv2.applyColorMap(np.uint8(heatmap_canvas * 255), cv2.COLORMAP_JET)
    
    return heatmap_colored

def save_webpage_heatmap():
    """Save full webpage with heatmap overlay"""
    global full_webpage_image, web_gaze_points
    
    if full_webpage_image is None:
        print("No webpage image captured, cannot save heatmap")
        return
    
    if not web_gaze_points or len(web_gaze_points) < 5:
        print("Not enough gaze data for heatmap, only saving webpage")
        try:
            if not os.path.exists("../output"):
                os.makedirs("../output")
            full_webpage_image.save("../output/output_image.png")
            print("Saved webpage image only (no heatmap) to ../output/output_image.png")
        except Exception as e:
            print(f"Error saving webpage image: {e}")
        return
    
    try:
        # Create output directory if needed
        if not os.path.exists("../output"):
            os.makedirs("../output")
        
        # Generate heatmap
        print("Generating heatmap overlay on captured webpage...")
        heatmap_overlay = generate_webpage_heatmap()
        
        if heatmap_overlay is None:
            print("Failed to generate heatmap, saving webpage only")
            full_webpage_image.save("../output/output_image.png")
            return
        
        # Convert PIL Image to numpy array
        webpage_np = np.array(full_webpage_image)
        
        # Make sure dimensions match
        if webpage_np.shape[:2] != heatmap_overlay.shape[:2]:
            heatmap_overlay = cv2.resize(
                heatmap_overlay, 
                (webpage_np.shape[1], webpage_np.shape[0]), 
                interpolation=cv2.INTER_AREA
            )
        
        # Create alpha channel based on heatmap intensity
        heatmap_gray = cv2.cvtColor(heatmap_overlay, cv2.COLOR_BGR2GRAY)
        alpha = (heatmap_gray > 10).astype(np.float32) * 0.7
        alpha = alpha.reshape(alpha.shape[0], alpha.shape[1], 1)
        
        # Blend images
        blended = webpage_np * (1 - alpha) + heatmap_overlay * alpha
        
        # Save both the original webpage and the heatmap overlay
        full_webpage_image.save("../output/output_image.png")
        print("Saved webpage image to ../output/output_image.png")
        
        cv2.imwrite("../output/output_image_heatmap.png", blended.astype(np.uint8))
        print("Saved heatmap overlay to ../output/output_image_heatmap.png")
        
        # Save raw gaze data
        with open("../output/webpage_gaze.csv", 'w') as f:
            f.write("x,y,timestamp\n")
            for x, y, ts in web_gaze_points:
                f.write(f"{x},{y},{ts}\n")
        print("Saved gaze data to ../output/webpage_gaze.csv")
    
    except Exception as e:
        print(f"Error saving heatmap: {e}")
        import traceback
        traceback.print_exc()

# Main loop
try:
    print("Starting eye tracking with automatic browser launch")
    print("Please complete calibration by looking at the red circles and pressing SPACE")
    
    # Check if we automatically enable browser mode after calibration
    if AUTO_LAUNCH_BROWSER:
        print("Browser will launch automatically after calibration")
        print("Controls: 'h' to toggle heatmap, 'b' to toggle browser, 's' to save data, 'q' to quit")
    
    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
        
        # Flip frame horizontally if enabled
        if FLIP_HORIZONTAL:
            frame = cv2.flip(frame, 1)  # 1 = horizontal flip
        
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
                    current_time = time.time()
                    gaze_points_data.append((current_time, smooth_gaze_point[0], smooth_gaze_point[1]))
                    
                    # Map to webpage coordinates if browser is active
                    if USE_BROWSER and browser is not None:
                        web_point = map_gaze_to_webpage(smooth_gaze_point)
                        if web_point is not None:
                            web_gaze_points.append((web_point[0], web_point[1], current_time))
                    
                    # Update fixation data
                    update_fixations(smooth_gaze_point)
                    
                    # Update heatmap
                    update_heatmap(smooth_gaze_point)
                    
                    # Draw gaze point
                    cv2.circle(display_frame, smooth_gaze_point, 10, (0, 0, 255), -1)
                    cv2.circle(display_frame, smooth_gaze_point, 4, (255, 255, 255), -1)
                    
                    # Display webpage tracking info if active
                    if USE_BROWSER and browser is not None:
                        web_point = map_gaze_to_webpage(smooth_gaze_point)
                        if web_point is not None:
                            cv2.putText(display_frame, f"Web: {int(web_point[0])}, {int(web_point[1])}", 
                                      (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        
                        # Fix: Check if scroll_position is None before converting to int
                        if scroll_position is not None:
                            cv2.putText(display_frame, f"Scroll: {int(scroll_position)}", 
                                      (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
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
        
        # Handle key presses and window events
        key = cv2.waitKey(1) & 0xFF
        
        # Check if window was closed
        if cv2.getWindowProperty('Gaze Tracker', cv2.WND_PROP_VISIBLE) < 1:
            break
        
        if key == ord('q'):
            break
        elif key == ord('h'):
            # Toggle heatmap display
            DISPLAY_HEATMAP = not DISPLAY_HEATMAP
        elif key == ord('f'):
            # Toggle fullscreen
            FULLSCREEN = not FULLSCREEN
            if FULLSCREEN:
                # Set window to be maximized but not true fullscreen
                cv2.setWindowProperty('Gaze Tracker', cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_AUTOSIZE)
                # Get current screen resolution
                screen_width = cv2.getWindowImageRect('Gaze Tracker')[2]
                screen_height = cv2.getWindowImageRect('Gaze Tracker')[3]
                # Resize window to match screen size
                cv2.resizeWindow('Gaze Tracker', screen_width, screen_height)
            else:
                # Return to normal window size
                cv2.setWindowProperty('Gaze Tracker', cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Gaze Tracker', 1024, 768)  # Default size
        elif key == ord('b'):
            # Toggle browser mode
            if not USE_BROWSER:
                threading.Thread(target=start_browser_tracking, daemon=True).start()
            else:
                if browser is not None:
                    print("Closing browser and saving webpage heatmap...")
                    save_webpage_heatmap()
                    browser.quit()
                    browser = None
                    USE_BROWSER = False
                    print("Browser closed")
        elif key == ord('s'):
            # Save heatmap
            save_heatmap()
            save_gaze_data()
            if USE_BROWSER and browser is not None:
                save_webpage_heatmap()
        elif key == ord('r'):
            # Reset calibration
            calibration_mode = True
            calibration_points = []
            calibration_eye_positions = []
            calibrated = False
            heatmap = np.zeros((frame_height, frame_width), dtype=np.float32)
            heatmap_updated = False
            
            # Close browser if open
            if browser is not None:
                browser.quit()
                browser = None
                USE_BROWSER = False
                
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
                    
                    # If browser mode is enabled, start it now after calibration
                    if AUTO_LAUNCH_BROWSER and browser is None:
                        threading.Thread(target=start_browser_tracking, daemon=True).start()

except KeyboardInterrupt:
    print("Stopping...")
finally:
    # Cleanup
    if browser is not None:
        try:
            print("Saving webpage heatmap before exit...")
            save_webpage_heatmap()
            print("Browser cleanup...")
            browser.quit()
            print("Browser closed")
        except Exception as exit_error:
            print(f"Error during browser cleanup: {exit_error}")
            import traceback
            traceback.print_exc()
    
    print("Releasing camera...")
    cap.release()
    print("Closing windows...")
    cv2.destroyAllWindows()
    
    # Save data before exit
    print("Saving eye tracking data...")
    if heatmap_updated:
        save_heatmap()
    if len(gaze_points_data) > 0:
        save_gaze_data()
    
    print("Cleanup complete. Exiting.")