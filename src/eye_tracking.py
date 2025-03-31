import cv2
import numpy as np
import os
import time
from gaze_tracking import GazeTracking
from datetime import datetime

class AttentionAnalyzer:
    """
    A simplified eye tracking solution for analyzing customer attention
    on products, advertisements, or interfaces.
    """

    def __init__(self):
        self.gaze = GazeTracking()
        self.webcam = cv2.VideoCapture(0)
        
        # Get screen dimensions
        screen_info = self._get_screen_dimensions()
        self.screen_width, self.screen_height = screen_info
        
        # Create directory for saving results
        self.results_dir = "attention_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize attention heat map
        self.heatmap = np.zeros((self.screen_height, self.screen_width), dtype=np.float32)
        self.attention_data = []
        
        # Session info
        self.session_start_time = None
        self.image_path = None
        self.recording = False

    def _get_screen_dimensions(self):
        """Get screen dimensions using opencv window"""
        # Create a fullscreen window to get dimensions
        cv2.namedWindow("temp", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("temp", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        screen = cv2.resize(np.zeros((100, 100, 3), dtype=np.uint8), (0, 0), fx=100, fy=100)
        cv2.imshow("temp", screen)
        cv2.waitKey(1)
        height, width, _ = screen.shape
        cv2.destroyWindow("temp")
        return width, height

    def load_image(self, image_path):
        """Load an image for attention analysis"""
        if not os.path.exists(image_path):
            print(f"Error: Image file {image_path} not found")
            return False
            
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.image = cv2.resize(self.image, (self.screen_width, self.screen_height))
        
        # Reset heatmap
        self.heatmap = np.zeros((self.screen_height, self.screen_width), dtype=np.float32)
        self.attention_data = []
        
        return True

    def start_recording(self):
        """Start recording attention data"""
        if self.image_path is None:
            print("Error: Load an image first before recording")
            return False
            
        self.recording = True
        self.session_start_time = datetime.now()
        print(f"Started recording attention data at {self.session_start_time}")
        return True

    def stop_recording(self):
        """Stop recording and save results"""
        if not self.recording:
            return
            
        self.recording = False
        session_duration = (datetime.now() - self.session_start_time).total_seconds()
        print(f"Recorded {len(self.attention_data)} gaze points over {session_duration:.1f} seconds")
        
        # Generate filename based on time
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_name = os.path.splitext(os.path.basename(self.image_path))[0]
        results_filename = f"{self.results_dir}/{image_name}_attention_{timestamp}"
        
        # Save heatmap visualization
        self._save_heatmap(f"{results_filename}_heatmap.jpg")
        
        # Save raw data
        self._save_data(f"{results_filename}_data.csv")
        
        print(f"Results saved to {results_filename}_heatmap.jpg and {results_filename}_data.csv")

    def _update_heatmap(self, x, y):
        """Update the attention heatmap"""
        if 0 <= x < self.screen_width and 0 <= y < self.screen_height:
            # Add Gaussian blob around the gaze point
            sigma = 50  # Standard deviation of the Gaussian blob
            mask_size = int(sigma * 6)  # Size of the Gaussian mask
            
            # Create x and y ranges for the mask
            x_min = max(0, x - mask_size // 2)
            x_max = min(self.screen_width, x + mask_size // 2)
            y_min = max(0, y - mask_size // 2)
            y_max = min(self.screen_height, y + mask_size // 2)
            
            # Create meshgrid for the mask region
            x_range = np.arange(x_min, x_max)
            y_range = np.arange(y_min, y_max)
            xx, yy = np.meshgrid(x_range, y_range)
            
            # Calculate Gaussian values
            gaussian = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
            
            # Update the heatmap
            self.heatmap[y_min:y_max, x_min:x_max] += gaussian

    def _save_heatmap(self, filename):
        """Save the attention heatmap overlaid on the image"""
        # Normalize heatmap
        if np.max(self.heatmap) > 0:
            norm_heatmap = self.heatmap / np.max(self.heatmap)
        else:
            norm_heatmap = self.heatmap
            
        # Convert to color heatmap
        heatmap_img = cv2.applyColorMap((norm_heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Blend with original image
        result = cv2.addWeighted(self.image, 0.7, heatmap_img, 0.3, 0)
        
        # Add legend
        legend_height = 30
        legend_width = self.screen_width
        legend = np.zeros((legend_height, legend_width, 3), dtype=np.uint8)
        
        # Create gradient
        for i in range(legend_width):
            intensity = i / (legend_width - 1)
            color = cv2.applyColorMap(np.array([[intensity * 255]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0]
            legend[:, i] = color
            
        # Add labels
        cv2.putText(legend, "Low Attention", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(legend, "High Attention", (legend_width - 150, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Combine with result
        result_with_legend = np.vstack((result, legend))
        
        # Save
        cv2.imwrite(filename, result_with_legend)

    def _save_data(self, filename):
        """Save the raw attention data as CSV"""
        with open(filename, 'w') as f:
            f.write("timestamp,x,y,duration\n")
            for point in self.attention_data:
                f.write(f"{point['timestamp']},{point['x']},{point['y']},{point['duration']}\n")

    def run_analysis(self):
        """Run the attention analysis and display real-time visualization"""
        if self.image_path is None:
            print("Error: Load an image first before analysis")
            return
            
        print("\nAttention Analysis Tool")
        print("=======================")
        print("Controls:")
        print("  R - Start/Stop recording")
        print("  S - Save current results")
        print("  Q - Quit")
        print("\nDisplay the image to analyze and begin tracking eye movements...")
        
        # Setup display window
        cv2.namedWindow("Attention Analysis", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Attention Analysis", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        last_gaze_time = time.time()
        last_gaze_position = None
        
        while True:
            # Get a new frame from the webcam
            success, frame = self.webcam.read()
            if not success:
                print("Failed to capture webcam frame")
                break
                
            # Process the frame with gaze tracking
            self.gaze.refresh(frame)
            
            # Create visualization frame
            result_frame = self.image.copy()
            
            # Process gaze data
            if self.gaze.pupils_located and not self.gaze.is_blinking():
                h_ratio = self.gaze.horizontal_ratio()
                v_ratio = self.gaze.vertical_ratio()
                
                if h_ratio is not None and v_ratio is not None:
                    # Simple mapping of gaze ratio to screen coordinates
                    # This is an approximation that doesn't require complex calibration
                    x = int((1.0 - h_ratio) * self.screen_width)  # Invert h_ratio for correct mapping
                    y = int(v_ratio * self.screen_height)
                    
                    # Calculate time spent at this gaze position
                    current_time = time.time()
                    if last_gaze_position:
                        duration = current_time - last_gaze_time
                        # Only record if there's a significant change in position
                        distance = np.sqrt((x - last_gaze_position[0])**2 + (y - last_gaze_position[1])**2)
                        if distance < 100:  # If the eye hasn't moved much, consider it the same fixation
                            # Update the heatmap more for longer fixations
                            self._update_heatmap(last_gaze_position[0], last_gaze_position[1])
                            
                            # Record data if we're recording
                            if self.recording:
                                timestamp = (datetime.now() - self.session_start_time).total_seconds()
                                self.attention_data.append({
                                    'timestamp': timestamp,
                                    'x': last_gaze_position[0],
                                    'y': last_gaze_position[1],
                                    'duration': duration
                                })
                    
                    # Draw gaze indicator
                    cv2.circle(result_frame, (x, y), 20, (0, 0, 255), 2)
                    cv2.line(result_frame, (x-25, y), (x+25, y), (0, 0, 255), 2)
                    cv2.line(result_frame, (x, y-25), (x, y+25), (0, 0, 255), 2)
                    
                    last_gaze_position = (x, y)
                    last_gaze_time = current_time
            
            # Create a blended visualization
            if np.max(self.heatmap) > 0:
                norm_heatmap = self.heatmap / np.max(self.heatmap)
                heatmap_img = cv2.applyColorMap((norm_heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
                result_frame = cv2.addWeighted(result_frame, 0.7, heatmap_img, 0.3, 0)
            
            # Display status
            status_text = "RECORDING" if self.recording else "NOT RECORDING"
            cv2.putText(result_frame, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if self.recording else (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow("Attention Analysis", result_frame)
            cv2.imshow("Eye Tracking", self.gaze.annotated_frame())
            
            # Handle key presses
            key = cv2.waitKey(1)
            if key == ord('q'):
                # Quit
                break
            elif key == ord('r'):
                # Toggle recording
                if self.recording:
                    self.stop_recording()
                else:
                    self.start_recording()
            elif key == ord('s'):
                # Save current results
                if not self.recording:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    image_name = os.path.splitext(os.path.basename(self.image_path))[0]
                    results_filename = f"{self.results_dir}/{image_name}_snapshot_{timestamp}"
                    self._save_heatmap(f"{results_filename}_heatmap.jpg")
                    print(f"Snapshot saved to {results_filename}_heatmap.jpg")
        
        # Clean up
        self.webcam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Attention Analysis System")
    parser.add_argument("--image", type=str, help="Path to the image for attention analysis")
    args = parser.parse_args()
    
    analyzer = AttentionAnalyzer()
    
    # Load the image if provided, otherwise ask for it
    if args.image:
        image_loaded = analyzer.load_image(args.image)
    else:
        image_path = input("Enter the path to the image for attention analysis: ")
        image_loaded = analyzer.load_image(image_path)
    
    if image_loaded:
        analyzer.run_analysis()
    else:
        print("Failed to load image. Exiting.")
