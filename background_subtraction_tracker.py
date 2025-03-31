import cv2
import numpy as np

class BackgroundSubtractionTracker:
    """
    Tracks small fast-moving insects using background subtraction and contour detection.
    
    This tracker uses the following steps:
    1. Background subtraction to isolate moving objects
    2. Morphological operations to clean up the foreground mask
    3. Contour detection to find foreground regions
    4. Centroid calculation and trajectory building
    """
    
    def __init__(self, max_targets=10):
        """
        Initialize tracker with parameters.
        
        Args:
            max_targets (int): Maximum number of targets to track
        """
        self.max_targets = max_targets
        
        # Background subtraction parameters
        self.history = 500
        self.var_threshold = 16
        self.detect_shadows = True
        self.min_area = 50  # Minimum contour area
        
        # Create background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.history,
            varThreshold=self.var_threshold,
            detectShadows=self.detect_shadows
        )
        
        # Initialize variables
        self.trajectories = []
        self.active_trajectories = []
        
    def process_frame(self, frame):
        """
        Process a new frame and update tracked positions.
        
        Args:
            frame (numpy.ndarray): New frame to process
            
        Returns:
            list: List of current positions of tracked targets
        """
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Remove shadows (gray areas) by thresholding
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=1)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(
            fg_mask.copy(), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Extract centroids from contours
        centers = []
        for cnt in contours:
            if cv2.contourArea(cnt) < self.min_area:
                continue
                
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centers.append((cx, cy))
        
        # Current positions to return
        current_positions = []
        
        # Limit number of centers to max_targets
        centers = centers[:self.max_targets]
        
        # Associate detections with trajectories
        if len(self.active_trajectories) == 0 and centers:
            # First detections, initialize trajectories
            self.active_trajectories = [[center] for center in centers]
            current_positions = centers
        else:
            # Match centers to existing trajectories using nearest neighbor
            if centers:
                used_centers = set()
                
                for traj_idx, trajectory in enumerate(self.active_trajectories):
                    if not trajectory:  # Skip empty trajectories
                        continue
                    
                    last_point = np.array(trajectory[-1])
                    best_idx = -1
                    min_dist = float('inf')
                    
                    # Find closest center to continue this trajectory
                    for i, center in enumerate(centers):
                        if i in used_centers:
                            continue
                        
                        dist = np.linalg.norm(np.array(last_point) - np.array(center))
                        if dist < min_dist and dist < 50:  # Add distance threshold
                            min_dist = dist
                            best_idx = i
                    
                    if best_idx != -1:
                        # Add this center to trajectory
                        self.active_trajectories[traj_idx].append(centers[best_idx])
                        current_positions.append(centers[best_idx])
                        used_centers.add(best_idx)
                
                # Create new trajectories for unassigned centers if needed
                if len(self.active_trajectories) < self.max_targets:
                    for i, center in enumerate(centers):
                        if i in used_centers:
                            continue
                        if len(self.active_trajectories) >= self.max_targets:
                            break
                        
                        self.active_trajectories.append([center])
                        current_positions.append(center)
        
        # Handle disappeared targets
        # If a trajectory hasn't been updated for a while, move it to completed
        i = 0
        while i < len(self.active_trajectories):
            if not self.active_trajectories[i]:
                # Remove empty trajectories
                self.active_trajectories.pop(i)
            else:
                i += 1
        
        # Ensure we don't have more than max_targets active trajectories
        if len(self.active_trajectories) > self.max_targets:
            # Sort by trajectory length (longer trajectories are more valuable)
            sorted_idx = sorted(range(len(self.active_trajectories)), 
                               key=lambda i: len(self.active_trajectories[i]))
            
            # Move excess trajectories to completed list
            for i in sorted_idx[:len(self.active_trajectories) - self.max_targets]:
                if len(self.active_trajectories[i]) > 5:  # Only save if it's substantial
                    self.trajectories.append(self.active_trajectories[i])
            
            # Keep only the max_targets longest trajectories
            self.active_trajectories = [self.active_trajectories[i] for i in 
                                       sorted_idx[-(self.max_targets):]]
        
        return current_positions
    
    def get_all_trajectories(self):
        """
        Get all trajectories including active and completed ones.
        
        Returns:
            list: List of all trajectories
        """
        result = self.trajectories.copy()
        for traj in self.active_trajectories:
            if len(traj) > 5:  # Only include trajectories with sufficient points
                result.append(traj)
        return result
    
    def visualize(self, frame, positions=None):
        """
        Visualize tracking results on frame with additional debug information.
        
        Args:
            frame (numpy.ndarray): Frame to draw on
            positions (list, optional): Current positions to highlight
            
        Returns:
            numpy.ndarray: Visualization frame
        """
        # Create a copy for visualization
        vis_frame = frame.copy()
        
        # Add visualization components
        h, w = frame.shape[:2]
        debug_width = 320  # Width of each debug visualization
        debug_height = int(debug_width * h / w)  # Keep aspect ratio
        
        # Create a canvas with room for debug visualizations
        canvas = np.zeros((h, w + debug_width, 3), dtype=np.uint8)
        canvas[:, :w] = vis_frame
        
        # Fill debug area with black background
        canvas[:, w:] = (20, 20, 20)  # Dark gray background
        
        # 1. Apply background subtraction to get foreground mask
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Save the original mask (with shadows) for visualization
        orig_mask = fg_mask.copy()
        
        # Remove shadows (gray areas) by thresholding
        _, fg_mask_binary = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        fg_mask_cleaned = cv2.morphologyEx(fg_mask_binary, cv2.MORPH_OPEN, kernel)
        fg_mask_cleaned = cv2.dilate(fg_mask_cleaned, kernel, iterations=1)
        
        # Find contours in the mask for visualization
        contours, _ = cv2.findContours(
            fg_mask_cleaned.copy(), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Resize for debug area
        orig_mask_small = cv2.resize(orig_mask, (debug_width, debug_height))
        fg_mask_binary_small = cv2.resize(fg_mask_binary, (debug_width, debug_height))
        fg_mask_cleaned_small = cv2.resize(fg_mask_cleaned, (debug_width, debug_height))
        
        # Draw original foreground mask (including shadows)
        y_offset = 10
        # Apply colormap to the original mask to better visualize shadow areas (gray values)
        orig_mask_color = cv2.applyColorMap(orig_mask_small, cv2.COLORMAP_JET)
        canvas[y_offset:y_offset+debug_height, w:w+debug_width] = orig_mask_color
        
        # Add label
        cv2.putText(canvas, "Original Mask (with shadows)", 
                  (w + 10, y_offset - 5), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw binary foreground mask (after thresholding)
        y_offset = y_offset + debug_height + 20
        fg_mask_binary_color = cv2.cvtColor(fg_mask_binary_small, cv2.COLOR_GRAY2BGR)
        canvas[y_offset:y_offset+debug_height, w:w+debug_width] = fg_mask_binary_color
        
        # Add label
        cv2.putText(canvas, "Binary Mask (no shadows)", 
                  (w + 10, y_offset - 5), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw cleaned foreground mask (after morphology)
        y_offset = y_offset + debug_height + 20
        fg_mask_cleaned_color = cv2.cvtColor(fg_mask_cleaned_small, cv2.COLOR_GRAY2BGR)
        canvas[y_offset:y_offset+debug_height, w:w+debug_width] = fg_mask_cleaned_color
        
        # Add label
        cv2.putText(canvas, "Cleaned Mask", 
                  (w + 10, y_offset - 5), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add legend for MOG2 mask colors
        legend_y = y_offset + debug_height + 10
        cv2.putText(canvas, "MOG2 Mask Legend:", 
                  (w + 10, legend_y), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw color samples for legend
        box_size = 15
        # Shadow pixel (gray in MOG2, ~127)
        cv2.rectangle(canvas, 
                    (w + 10, legend_y + 10), 
                    (w + 10 + box_size, legend_y + 10 + box_size), 
                    (127, 127, 127), -1)
        cv2.putText(canvas, "Shadow", 
                  (w + 35, legend_y + 22), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Foreground pixel (white in MOG2, 255)
        cv2.rectangle(canvas, 
                    (w + 120, legend_y + 10), 
                    (w + 120 + box_size, legend_y + 10 + box_size), 
                    (255, 255, 255), -1)
        cv2.putText(canvas, "Foreground", 
                  (w + 145, legend_y + 22), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Background pixel (black in MOG2, 0)
        cv2.rectangle(canvas, 
                    (w + 230, legend_y + 10), 
                    (w + 230 + box_size, legend_y + 10 + box_size), 
                    (0, 0, 0), -1)
        cv2.putText(canvas, "Background", 
                  (w + 255, legend_y + 22), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Draw contours on main image
        contour_img = vis_frame.copy()
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                # Draw small contours in red (below threshold)
                cv2.drawContours(canvas, [cnt], 0, (0, 0, 255), 1)
            else:
                # Draw valid contours in green (above threshold)
                cv2.drawContours(canvas, [cnt], 0, (0, 255, 0), 2)
                
                # Draw the area size
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(canvas, f"{area:.0f}", 
                             (cx, cy), 
                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Draw active trajectories
        for trajectory in self.active_trajectories:
            if len(trajectory) < 2:
                continue
                
            # Draw trajectory line
            for i in range(1, len(trajectory)):
                cv2.line(
                    canvas,
                    trajectory[i-1],
                    trajectory[i],
                    (0, 255, 0),
                    2
                )
            
            # Draw circle at the latest position
            cv2.circle(canvas, trajectory[-1], 5, (0, 0, 255), -1)
        
        # Draw current positions if provided
        if positions:
            for pos in positions:
                cv2.circle(canvas, pos, 7, (255, 0, 0), 2)
        
        # Add parameter information
        info_y = legend_y + 40
        cv2.putText(canvas, "Parameters:", (w + 10, info_y), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        param_info = [
            f"history: {self.history}",
            f"var_threshold: {self.var_threshold}",
            f"detect_shadows: {self.detect_shadows}",
            f"min_area: {self.min_area}"
        ]
        
        for i, info in enumerate(param_info):
            cv2.putText(canvas, info, (w + 10, info_y + 20 + i * 20), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Add count of active trajectories
        active_count = sum(1 for traj in self.active_trajectories if len(traj) > 0)
        cv2.putText(canvas, f"Active trajectories: {active_count}", 
                  (w + 10, info_y + 20 + len(param_info) * 20), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Add frame count to top-left corner
        if hasattr(self, 'frame_count'):
            self.frame_count += 1
        else:
            self.frame_count = 1
            
        cv2.putText(canvas, f"Frame: {self.frame_count}", 
                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return canvas 