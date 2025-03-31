import cv2
import numpy as np

class MotionDetector:
    """
    General-purpose motion detection utility using frame differencing.
    
    This class handles frame differencing, threshold application, noise removal,
    and contour detection to identify motion in video streams.
    
    Features:
    - Frame differencing with adjustable parameters
    - Support for periodic reference frame updates
    - Optional artificial motion simulation for testing
    - Morphological operations for noise reduction
    - Contour detection and centroid extraction
    """
    
    def __init__(self, diff_thresh_value=25, min_area=30, frame_diff_interval=1):
        """
        Initialize motion detector with parameters.
        
        Args:
            diff_thresh_value (int): Threshold for frame differencing (10-50)
            min_area (int): Minimum contour area to consider (10-200)
            frame_diff_interval (int): Calculate difference every N frames
        """
        self.diff_thresh_value = diff_thresh_value
        self.min_area = min_area
        self.frame_diff_interval = frame_diff_interval
        
        # State variables
        self.prev_gray = None
        self.reference_gray = None
        self.frame_count = 0
        self.demo_frame_count = 0
        self.enable_motion_simulation = False
    
    def initialize(self, first_frame):
        """
        Initialize the detector with the first frame.
        
        Args:
            first_frame (numpy.ndarray): First frame of the video
        """
        self.prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        self.reference_gray = self.prev_gray.copy()
        self.frame_count = 0
        self.demo_frame_count = 0
    
    def set_motion_simulation(self, enable=True):
        """
        Enable or disable artificial motion simulation.
        
        Args:
            enable (bool): Whether to enable motion simulation
        """
        self.enable_motion_simulation = enable
    
    def detect_motion(self, frame):
        """
        Detect motion in the current frame.
        
        Args:
            frame (numpy.ndarray): Current frame
            
        Returns:
            tuple: (diff, diff_mask, centers) - difference image, binary mask, and detected centroids
        """
        if self.prev_gray is None:
            self.initialize(frame)
            return None, None, []
        
        # Convert to grayscale
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Update frame counter
        self.frame_count += 1
        
        # Short-term differencing for new motion detection
        short_term_diff = cv2.absdiff(self.prev_gray, curr_gray)
        
        # Long-term differencing with reference frame
        if self.frame_count % self.frame_diff_interval == 0:
            # Update reference frame every N frames but preserve some history
            # Mix current reference with new frame (80% old, 20% new) to avoid sudden changes
            if self.frame_count > self.frame_diff_interval:
                alpha = 0.8  # Weight for old reference frame
                self.reference_gray = cv2.addWeighted(self.reference_gray, alpha, curr_gray, 1.0 - alpha, 0)
            else:
                # First update just takes the current frame
                self.reference_gray = curr_gray.copy()
        
        # Calculate difference using reference frame
        long_term_diff = cv2.absdiff(self.reference_gray, curr_gray)
        
        # Combine short-term and long-term differences to catch both new and sustained motion
        # For each pixel, take the maximum difference value
        diff = cv2.max(short_term_diff, long_term_diff)
        
        # Add artificial motion if enabled
        if self.enable_motion_simulation:
            diff = self._add_simulated_motion(diff)
        
        # Apply threshold
        _, diff_mask = cv2.threshold(diff, self.diff_thresh_value, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to remove noise and fill small holes
        kernel = np.ones((3, 3), np.uint8)
        diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_OPEN, kernel)  # Remove small noise
        diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_CLOSE, kernel)  # Fill small holes
        
        # Find contours of motion regions
        contours, _ = cv2.findContours(
            diff_mask.copy(), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Extract centers from contours
        centers = []
        for cnt in contours:
            if cv2.contourArea(cnt) < self.min_area:
                continue
                
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centers.append((cx, cy))
        
        # Special handling for sudden changes - if a lot of motion is detected suddenly
        # This happens when a new insect enters the frame or something changes rapidly
        if len(centers) > 0 and self.frame_count > 10:
            # If we suddenly have more centers than before and they're in new areas
            # Make sure we're tracking them by incorporating them into reference
            if len(centers) >= 3:  # Arbitrary threshold for "significant" motion
                # Update reference frame more aggressively in regions with motion
                motion_mask = np.zeros_like(curr_gray)
                # Draw filled contours on the mask
                for cnt in contours:
                    if cv2.contourArea(cnt) >= self.min_area:
                        cv2.drawContours(motion_mask, [cnt], 0, 255, -1)
                        
                # Dilate the motion areas slightly to include surrounding pixels
                dilate_kernel = np.ones((5, 5), np.uint8)
                motion_mask = cv2.dilate(motion_mask, dilate_kernel, iterations=1)
                
                # Where motion is detected, update reference frame more aggressively
                # This helps adapt to new objects that enter the frame
                ref_update_mask = (motion_mask > 0).astype(np.uint8)
                inverse_mask = 1 - ref_update_mask
                
                # Update reference frame only in motion areas with more weight on current frame
                self.reference_gray = cv2.addWeighted(
                    self.reference_gray * inverse_mask, 1.0,  # Keep non-motion areas unchanged
                    curr_gray * ref_update_mask, 1.0,  # Use current frame for motion areas
                    0
                )
        
        # Update previous frame
        self.prev_gray = curr_gray.copy()
        
        return diff, diff_mask, centers
    
    def _add_simulated_motion(self, diff):
        """
        Add artificial motion patterns to the difference image for demonstration.
        
        Args:
            diff (numpy.ndarray): Original difference image
            
        Returns:
            numpy.ndarray: Difference image with artificial motion added
        """
        self.demo_frame_count += 1
        
        # Every 30 frames, create a new simulated insect
        if self.demo_frame_count % 30 == 0:
            # Create a new trajectory starting point at the edge of the frame
            side = np.random.randint(0, 4)  # 0: top, 1: right, 2: bottom, 3: left
            
            h, w = diff.shape
            margin = 50  # Margin from edge
            
            if side == 0:  # Top
                self.sim_x = np.random.randint(margin, w - margin)
                self.sim_y = margin
                self.sim_vx = np.random.randn() * 2  # Random x velocity
                self.sim_vy = np.random.uniform(1, 3)  # Downward velocity
            elif side == 1:  # Right
                self.sim_x = w - margin
                self.sim_y = np.random.randint(margin, h - margin)
                self.sim_vx = -np.random.uniform(1, 3)  # Leftward velocity
                self.sim_vy = np.random.randn() * 2  # Random y velocity
            elif side == 2:  # Bottom
                self.sim_x = np.random.randint(margin, w - margin)
                self.sim_y = h - margin
                self.sim_vx = np.random.randn() * 2  # Random x velocity
                self.sim_vy = -np.random.uniform(1, 3)  # Upward velocity
            else:  # Left
                self.sim_x = margin
                self.sim_y = np.random.randint(margin, h - margin)
                self.sim_vx = np.random.uniform(1, 3)  # Rightward velocity
                self.sim_vy = np.random.randn() * 2  # Random y velocity
                
            self.sim_active = True
            self.sim_size = np.random.randint(10, 30)  # Random size of simulated insect
            self.sim_intensity = np.random.randint(80, 255)  # Random brightness
        
        # Every frame, update the simulated insect position
        if hasattr(self, 'sim_active') and self.sim_active:
            # Check if enough natural motion is present
            natural_motion_pixels = np.sum(diff > self.diff_thresh_value)
            
            # Only add simulation if natural motion is limited
            if natural_motion_pixels < 100:  # If minimal natural motion
                # Update position with noise (simulate erratic insect movement)
                # Add some random acceleration
                self.sim_vx += np.random.randn() * 0.5
                self.sim_vy += np.random.randn() * 0.5
                
                # Add some damping to avoid excessive speeds
                self.sim_vx *= 0.9
                self.sim_vy *= 0.9
                
                # Update position
                self.sim_x += self.sim_vx
                self.sim_y += self.sim_vy
                
                h, w = diff.shape
                
                # Bounce off edges with random direction change
                if self.sim_x < 0:
                    self.sim_x = 0
                    self.sim_vx = np.random.uniform(0.5, 2.0)  # Bounce right
                elif self.sim_x >= w:
                    self.sim_x = w - 1
                    self.sim_vx = -np.random.uniform(0.5, 2.0)  # Bounce left
                
                if self.sim_y < 0:
                    self.sim_y = 0
                    self.sim_vy = np.random.uniform(0.5, 2.0)  # Bounce down
                elif self.sim_y >= h:
                    self.sim_y = h - 1
                    self.sim_vy = -np.random.uniform(0.5, 2.0)  # Bounce up
                
                # Draw the simulated insect with a realistic pattern
                # Use an irregular blob shape for more realism
                for theta in np.linspace(0, 2*np.pi, 12):  # 12 points around a circle
                    # Randomize the radius to create an irregular shape
                    radius = self.sim_size * (0.7 + 0.3 * np.random.random())
                    
                    # Calculate point coordinates
                    pt_x = int(self.sim_x + radius * np.cos(theta))
                    pt_y = int(self.sim_y + radius * np.sin(theta))
                    
                    # Make sure the point is within bounds
                    if 0 <= pt_y < h and 0 <= pt_x < w:
                        # Create a small blob at this point
                        blob_size = int(radius * 0.3)
                        x_min = max(0, pt_x - blob_size)
                        x_max = min(w, pt_x + blob_size)
                        y_min = max(0, pt_y - blob_size)
                        y_max = min(h, pt_y + blob_size)
                        
                        # Add intensity to form a blob
                        intensity = self.sim_intensity * (0.8 + 0.2 * np.random.random())
                        diff[y_min:y_max, x_min:x_max] = intensity
                
                # With a small probability, change direction randomly to simulate erratic movement
                if np.random.random() < 0.1:
                    angle = np.random.uniform(0, 2*np.pi)
                    speed = np.sqrt(self.sim_vx**2 + self.sim_vy**2)
                    self.sim_vx = speed * np.cos(angle)
                    self.sim_vy = speed * np.sin(angle)
                
                # Occasionally let the insect disappear (beyond frame edge)
                if self.demo_frame_count % 300 == 0:
                    self.sim_active = False
        
        return diff
    
    def visualize(self, frame, canvas, w, debug_width, debug_height):
        """
        Add motion detection visualization to debug canvas.
        
        Args:
            frame (numpy.ndarray): Current frame
            canvas (numpy.ndarray): Canvas to draw on
            w (int): Width of main frame
            debug_width (int): Width of debug area
            debug_height (int): Height of debug area
            
        Returns:
            tuple: (canvas, y_offset) - updated canvas and vertical position for next visualization
        """
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None or self.reference_gray is None:
            return canvas, 0
        
        # Compute both short-term and long-term frame differences
        short_term_diff = cv2.absdiff(self.prev_gray, curr_gray)
        long_term_diff = cv2.absdiff(self.reference_gray, curr_gray)
        
        # Combine differences to get composite diff (same as in detect_motion)
        diff = cv2.max(short_term_diff, long_term_diff)
        
        # Apply simulated motion if enabled
        if self.enable_motion_simulation:
            diff = self._add_simulated_motion(diff)
        
        # Apply threshold
        _, diff_mask = cv2.threshold(diff, self.diff_thresh_value, 255, cv2.THRESH_BINARY)
        
        # Resize for debug area
        short_diff_small = cv2.resize(short_term_diff, (debug_width, debug_height))
        long_diff_small = cv2.resize(long_term_diff, (debug_width, debug_height))
        diff_small = cv2.resize(diff, (debug_width, debug_height))
        diff_mask_small = cv2.resize(diff_mask, (debug_width, debug_height))
        
        # Normalize and enhance difference visualizations for better visibility
        short_diff_norm = cv2.normalize(short_diff_small, None, 0, 255, cv2.NORM_MINMAX)
        long_diff_norm = cv2.normalize(long_diff_small, None, 0, 255, cv2.NORM_MINMAX)
        diff_norm = cv2.normalize(diff_small, None, 0, 255, cv2.NORM_MINMAX)
        
        # Apply contrast enhancement
        short_diff_enhanced = cv2.convertScaleAbs(short_diff_norm, alpha=3.0, beta=0)
        long_diff_enhanced = cv2.convertScaleAbs(long_diff_norm, alpha=3.0, beta=0)
        diff_enhanced = cv2.convertScaleAbs(diff_norm, alpha=3.0, beta=0)
        
        # Convert to color using different colormaps for distinction
        short_diff_color = cv2.applyColorMap(short_diff_enhanced, cv2.COLORMAP_COOL)  # Cool colormap
        long_diff_color = cv2.applyColorMap(long_diff_enhanced, cv2.COLORMAP_HOT)     # Hot colormap
        diff_color = cv2.applyColorMap(diff_enhanced, cv2.COLORMAP_JET)               # Jet colormap
        
        # Color for the mask - bright green
        diff_mask_color = np.zeros_like(diff_color)
        diff_mask_color[diff_mask_small > 0] = [0, 255, 0]
        
        # Calculate vertical positions for all visualizations
        y_offset = 10
        viz_spacing = 60  # Space between visualizations
        
        # Reduced height for each panel to fit more panels
        panel_height = int(debug_height * 0.8)
        
        # 1. Draw short-term difference
        cv2.rectangle(canvas, (w+5, y_offset-5), (w+debug_width-5, y_offset+panel_height+5), (100, 100, 100), 1)
        canvas[y_offset:y_offset+panel_height, w:w+debug_width] = short_diff_color
        label_bg_size = cv2.getTextSize("Short-term Motion", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(canvas, (w+8, y_offset-20), (w+10+label_bg_size[0], y_offset-5), (40, 40, 40), -1)
        cv2.putText(canvas, "Short-term Motion", 
                  (w + 10, y_offset - 7), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 2. Draw long-term difference with reference frame
        y_offset += panel_height + viz_spacing
        cv2.rectangle(canvas, (w+5, y_offset-5), (w+debug_width-5, y_offset+panel_height+5), (100, 100, 100), 1)
        canvas[y_offset:y_offset+panel_height, w:w+debug_width] = long_diff_color
        label_bg_size = cv2.getTextSize(f"Long-term Motion (ref every {self.frame_diff_interval} frames)", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(canvas, (w+8, y_offset-20), (w+10+label_bg_size[0], y_offset-5), (40, 40, 40), -1)
        cv2.putText(canvas, f"Long-term Motion (ref every {self.frame_diff_interval} frames)", 
                  (w + 10, y_offset - 7), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 3. Draw combined difference (final result before thresholding)
        y_offset += panel_height + viz_spacing
        cv2.rectangle(canvas, (w+5, y_offset-5), (w+debug_width-5, y_offset+panel_height+5), (100, 100, 100), 1)
        canvas[y_offset:y_offset+panel_height, w:w+debug_width] = diff_color
        label_bg_size = cv2.getTextSize("Combined Motion", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(canvas, (w+8, y_offset-20), (w+10+label_bg_size[0], y_offset-5), (40, 40, 40), -1)
        cv2.putText(canvas, "Combined Motion", 
                  (w + 10, y_offset - 7), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 4. Draw thresholded mask
        y_offset += panel_height + viz_spacing
        cv2.rectangle(canvas, (w+5, y_offset-5), (w+debug_width-5, y_offset+panel_height+5), (100, 100, 100), 1)
        # Create a copy of the mask to draw centers on
        diff_mask_with_centers = diff_mask_color.copy()
        
        # Find contours to draw centers on visualization
        contours, _ = cv2.findContours(
            diff_mask_small.copy(), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Extract and draw centers
        for cnt in contours:
            if cv2.contourArea(cnt) < (self.min_area * diff_mask_small.shape[0] / diff_mask.shape[0]):
                continue
                
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                # Draw red circle at center
                cv2.circle(diff_mask_with_centers, (cx, cy), 5, (0, 0, 255), -1)
                # Add arrow to indicate detected motion center
                cv2.drawMarker(diff_mask_with_centers, (cx, cy), (255, 0, 255), 
                             markerType=cv2.MARKER_DIAMOND, markerSize=8, thickness=2)
        
        canvas[y_offset:y_offset+panel_height, w:w+debug_width] = diff_mask_with_centers
        label_bg_size = cv2.getTextSize("Threshold Mask", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(canvas, (w+8, y_offset-20), (w+10+label_bg_size[0], y_offset-5), (40, 40, 40), -1)
        cv2.putText(canvas, "Threshold Mask", 
                  (w + 10, y_offset - 7), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add count of pixels exceeding threshold
        active_pixels = np.sum(diff_mask > 0)
        cv2.putText(canvas, f"Active pixels: {active_pixels}", 
                  (w + 10, y_offset + panel_height + 15), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Add colorbar and threshold indicator
        cbar_h = 15
        cbar_w = debug_width - 20
        cbar_x = w + 10
        cbar_y = y_offset + panel_height + 30
        
        # Draw colorbar background
        cv2.rectangle(canvas, (cbar_x, cbar_y), (cbar_x + cbar_w, cbar_y + cbar_h), (50, 50, 50), -1)
        
        # Draw colorbar gradient
        for i in range(cbar_w):
            val = int(255 * i / cbar_w)
            color = cv2.applyColorMap(np.array([[val]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0]
            cv2.line(canvas, (cbar_x + i, cbar_y), (cbar_x + i, cbar_y + cbar_h), color.tolist(), 1)
        
        # Add threshold line
        thresh_pos = int(self.diff_thresh_value * cbar_w / 255) + cbar_x
        cv2.line(canvas, (thresh_pos, cbar_y - 3), (thresh_pos, cbar_y + cbar_h + 3), (0, 255, 255), 2)
        
        # Add labels for colorbar
        cv2.putText(canvas, "0", (cbar_x, cbar_y + cbar_h + 15), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(canvas, "255", (cbar_x + cbar_w - 20, cbar_y + cbar_h + 15), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(canvas, f"Threshold: {self.diff_thresh_value}", (cbar_x, cbar_y + cbar_h + 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Return updated canvas and the new vertical position for next elements
        return canvas, cbar_y + cbar_h + 40 