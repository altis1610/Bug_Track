import cv2
import numpy as np

class DenseOpticalFlowTracker:
    """
    Tracks small fast-moving insects using dense optical flow (Farneback method).
    
    This tracker uses the following steps:
    1. Calculate dense optical flow between consecutive frames
    2. Filter motion by magnitude threshold
    3. Extract motion regions and their centroids
    4. Associate detections with existing tracks
    """
    
    def __init__(self, max_targets=10):
        """
        Initialize tracker with parameters.
        
        Args:
            max_targets (int): Maximum number of targets to track
        """
        self.max_targets = max_targets
        
        # Motion parameters
        self.mag_thresh = 2.0  # Motion magnitude threshold
        self.min_area = 50     # Minimum area for object detection
        
        # Farneback optical flow parameters
        self.flow_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        
        # Initialize variables
        self.prev_gray = None
        self.trajectories = []
        self.active_trajectories = []
        
    def initialize(self, first_frame):
        """
        Initialize the tracker with the first frame.
        
        Args:
            first_frame (numpy.ndarray): First frame of the video
        """
        self.prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
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
        if self.prev_gray is None:
            self.initialize(frame)
            return []
            
        # Convert to grayscale
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, 
            curr_gray, 
            None,
            **self.flow_params
        )
        
        # Calculate magnitude and angle
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Create a mask of significant motion
        motion_mask = np.uint8((mag > self.mag_thresh) * 255)
        
        # Apply morphology operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
        motion_mask = cv2.dilate(motion_mask, kernel, iterations=1)
        
        # Find contours of motion regions
        contours, _ = cv2.findContours(
            motion_mask, 
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
        
        # Update previous frame
        self.prev_gray = curr_gray.copy()
        
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
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Add visualization components
        h, w = frame.shape[:2]
        debug_width = 320  # Width of each debug visualization
        debug_height = int(debug_width * h / w)  # Keep aspect ratio
        
        # Create a canvas with room for debug visualizations
        canvas = np.zeros((h, w + debug_width, 3), dtype=np.uint8)
        canvas[:, :w] = vis_frame
        
        # Fill debug area with black background
        canvas[:, w:] = (20, 20, 20)  # Dark gray background
        
        # 1. Draw optical flow magnitude visualization if available
        if self.prev_gray is not None:
            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, 
                curr_gray, 
                None,
                **self.flow_params
            )
            
            # Calculate magnitude and angle of flow vectors
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            # Create motion mask based on magnitude threshold
            motion_mask = np.uint8((mag > self.mag_thresh) * 255)
            
            # Normalize magnitude for visualization
            max_mag = np.max(mag) if np.max(mag) > 0 else 1
            normalized_mag = np.uint8(np.minimum(mag * 255 / max_mag, 255))
            
            # Resize for debug area
            mag_small = cv2.resize(normalized_mag, (debug_width, debug_height))
            motion_mask_small = cv2.resize(motion_mask, (debug_width, debug_height))
            
            # Colorize magnitude
            mag_color = cv2.applyColorMap(mag_small, cv2.COLORMAP_JET)
            mask_color = cv2.cvtColor(motion_mask_small, cv2.COLOR_GRAY2BGR)
            
            # Draw in debug area
            y_offset = 10
            canvas[y_offset:y_offset+debug_height, w:w+debug_width] = mag_color
            
            # Add label
            cv2.putText(canvas, "Flow Magnitude", 
                       (w + 10, y_offset - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add colorbar for magnitude
            cbar_h = 15
            cbar_w = debug_width - 20
            cbar_x = w + 10
            cbar_y = y_offset + debug_height + 5
            
            # Draw colorbar background
            cv2.rectangle(canvas, (cbar_x, cbar_y), (cbar_x + cbar_w, cbar_y + cbar_h), (50, 50, 50), -1)
            
            # Draw colorbar gradient
            for i in range(cbar_w):
                val = int(255 * i / cbar_w)
                color = cv2.applyColorMap(np.array([[val]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0]
                cv2.line(canvas, (cbar_x + i, cbar_y), (cbar_x + i, cbar_y + cbar_h), color.tolist(), 1)
            
            # Add threshold line on colorbar (scaled to max magnitude)
            thresh_val = min(self.mag_thresh * 255 / max_mag, 255)
            thresh_pos = int(thresh_val * cbar_w / 255) + cbar_x
            cv2.line(canvas, (thresh_pos, cbar_y - 3), (thresh_pos, cbar_y + cbar_h + 3), (0, 255, 255), 2)
            
            # Add labels for colorbar
            cv2.putText(canvas, "0", (cbar_x, cbar_y + cbar_h + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(canvas, f"Max: {max_mag:.1f}", (cbar_x + cbar_w - 60, cbar_y + cbar_h + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(canvas, f"Threshold: {self.mag_thresh:.1f}", (cbar_x, cbar_y + cbar_h + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # 2. Draw thresholded motion mask
            y_offset = 2 * y_offset + debug_height + 40
            canvas[y_offset:y_offset+debug_height, w:w+debug_width] = mask_color
            
            # Add label
            cv2.putText(canvas, "Motion Mask", 
                       (w + 10, y_offset - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 3. Draw flow visualization on main image (sparse, for better visibility)
            step = 16  # Only show flow every step pixels
            for y in range(0, h, step):
                for x in range(0, w, step):
                    if mag[y, x] > self.mag_thresh:
                        # Get flow at this point
                        fx, fy = flow[y, x]
                        # Draw a line showing flow direction and magnitude
                        cv2.line(canvas, 
                                (x, y), 
                                (int(x + fx), int(y + fy)), 
                                (0, 255, 255), 
                                1)
                        # Draw a circle at the end point
                        cv2.circle(canvas, (int(x + fx), int(y + fy)), 1, (0, 0, 255), -1)
        
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
        info_y = y_offset + debug_height + 20
        cv2.putText(canvas, "Parameters:", (w + 10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        param_info = [
            f"mag_thresh: {self.mag_thresh:.1f}",
            f"min_area: {self.min_area}",
            f"pyr_scale: {self.flow_params['pyr_scale']}",
            f"levels: {self.flow_params['levels']}",
            f"winsize: {self.flow_params['winsize']}",
            f"iterations: {self.flow_params['iterations']}",
            f"poly_n: {self.flow_params['poly_n']}",
            f"poly_sigma: {self.flow_params['poly_sigma']}"
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