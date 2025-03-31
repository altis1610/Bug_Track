import cv2
import numpy as np
from motion_detection import MotionDetector

class SparseOpticalFlowTracker:
    """
    Tracks small fast-moving insects using sparse optical flow with frame differencing.
    
    This tracker uses the following steps:
    1. Frame differencing to identify motion regions
    2. Feature point extraction within motion regions
    3. Optical flow tracking of feature points between frames
    4. Trajectory building from tracked points
    """
    
    def __init__(self, max_targets=10):
        """
        Initialize tracker with parameters.
        
        Args:
            max_targets (int): Maximum number of targets to track
        """
        self.max_targets = max_targets
        
        # Parameters for feature detection
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )
        
        # Parameters for Lucas-Kanade optical flow
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Frame differencing parameters
        self.diff_thresh_value = 10  # Reduced from 25 to 10
        self.frame_diff_interval = 5  # Calculate difference every 5 frames
        
        # Motion detection parameters
        self.min_movement_threshold = 3  # Minimum movement in pixels to consider a point moving
        self.stagnant_frames_limit = 5  # Number of frames a point can remain static before it's considered non-insect
        
        # Initialize motion detector
        self.motion_detector = MotionDetector(
            diff_thresh_value=self.diff_thresh_value,
            min_area=30,
            frame_diff_interval=self.frame_diff_interval
        )
        self.motion_detector.set_motion_simulation(True)
        
        # Initialize variables
        self.prev_gray = None
        self.p0 = None
        self.trajectories = []
        self.active_trajectories = []
        self.frame_count = 0
        
        # Track motion for each trajectory
        self.stagnant_frames_counter = []  # Count frames where point didn't move significantly
        self.refresh_features_interval = 15  # Refresh features periodically
        
    def initialize(self, first_frame):
        """
        Initialize the tracker with the first frame.
        
        Args:
            first_frame (numpy.ndarray): First frame of the video
        """
        self.prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        self.motion_detector.initialize(first_frame)
        self.p0 = cv2.goodFeaturesToTrack(self.prev_gray, mask=None, **self.feature_params)
        self.trajectories = []
        self.active_trajectories = []
        self.stagnant_frames_counter = []
        self.frame_count = 0
        
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
        
        # Use motion detector to get motion regions
        diff, diff_mask, new_motion_centers = self.motion_detector.detect_motion(frame)
        
        # Force refresh features periodically or when very few points remain
        should_refresh = (self.frame_count % self.refresh_features_interval == 0) or \
                          (self.p0 is None or len(self.p0) < 5)
                       
        # Extract new features in motion areas if needed
        if should_refresh:
            self.p0 = cv2.goodFeaturesToTrack(curr_gray, mask=diff_mask, **self.feature_params)
            if self.p0 is None:
                self.prev_gray = curr_gray.copy()
                return []
        
        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, curr_gray, self.p0, None, **self.lk_params
        )
        
        if p1 is None:
            self.prev_gray = curr_gray.copy()
            return []
        
        # Filter good points
        good_new = p1[st == 1]
        good_old = self.p0[st == 1]
        
        # Calculate movement distances for each point
        movements = []
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            # Calculate distance moved
            dist = np.linalg.norm(new - old)
            movements.append(dist)
        
        # Current positions to return
        current_positions = []
        
        # Reset stagnant counter array if number of points changed
        if len(self.stagnant_frames_counter) != len(good_new):
            self.stagnant_frames_counter = [0] * len(good_new)
        
        # Update stagnant counters based on movement
        for i, dist in enumerate(movements):
            if dist < self.min_movement_threshold:
                self.stagnant_frames_counter[i] += 1
            else:
                # Reset counter if point moved significantly
                self.stagnant_frames_counter[i] = 0
        
        # Create a mask of points that haven't been static for too long
        valid_motion_mask = np.array([
            counter < self.stagnant_frames_limit for counter in self.stagnant_frames_counter
        ])
        
        # Filter points that have been static for too long
        if valid_motion_mask.size > 0:
            # Only keep points with sufficient movement
            good_new_filtered = good_new[valid_motion_mask]
            movements_filtered = [movements[i] for i, valid in enumerate(valid_motion_mask) if valid]
        else:
            good_new_filtered = good_new
            movements_filtered = movements
            
        # If we still have a shortage of moving points, try to add new feature points from motion areas
        if len(good_new_filtered) < self.max_targets and new_motion_centers:
            # Convert motion centers to proper format for tracking
            new_points = np.array([[center[0], center[1]] for center in new_motion_centers], dtype=np.float32).reshape(-1, 1, 2)
            
            # Limit to a reasonable number of new points
            max_new = min(len(new_points), self.max_targets - len(good_new_filtered))
            if max_new > 0 and len(new_points) > 0:
                # Add these new points to be tracked
                if self.p0 is None:
                    self.p0 = new_points[:max_new]
                else:
                    # Add only points that are not too close to existing points
                    for np_idx in range(min(len(new_points), max_new)):
                        np_x, np_y = new_points[np_idx][0]
                        too_close = False
                        for gp in good_new_filtered:
                            gp_x, gp_y = gp
                            if np.linalg.norm(np.array([np_x, np_y]) - np.array([gp_x, gp_y])) < 20:
                                too_close = True
                                break
                        if not too_close:
                            p_to_add = new_points[np_idx].reshape(1, 1, 2)
                            if len(good_new_filtered) == 0:
                                good_new_filtered = p_to_add.reshape(-1, 2)
                            else:
                                good_new_filtered = np.append(good_new_filtered, p_to_add.reshape(-1, 2), axis=0)
        
        # Sort points by movement amount (descending) to prioritize moving points
        if len(good_new_filtered) > 0 and len(movements_filtered) > 0:
            # Create indices sorted by movement
            movement_indices = np.argsort(movements_filtered)[::-1]
            good_new_filtered = good_new_filtered[movement_indices]
        
        # Update trajectories
        if len(self.active_trajectories) == 0 and len(good_new_filtered) > 0:
            # First time or reset, initialize trajectories
            self.active_trajectories = [[] for _ in range(min(len(good_new_filtered), self.max_targets))]
            
            # Take up to max_targets points, prioritizing those with movement
            for i in range(min(len(good_new_filtered), self.max_targets)):
                point = good_new_filtered[i].flatten()
                self.active_trajectories[i].append((int(point[0]), int(point[1])))
                current_positions.append((int(point[0]), int(point[1])))
        else:
            # Link points to existing trajectories (simple nearest neighbor matching)
            used_points = set()
            
            for traj_idx, trajectory in enumerate(self.active_trajectories):
                if not trajectory:  # Skip empty trajectories
                    continue
                
                last_point = np.array(trajectory[-1])
                best_idx = -1
                min_dist = float('inf')
                
                # Find closest new point to continue this trajectory
                for i, point in enumerate(good_new_filtered):
                    if i in used_points:
                        continue
                    
                    dist = np.linalg.norm(last_point - point)
                    # Stricter distance threshold for matching
                    if dist < min_dist and dist < 50:
                        min_dist = dist
                        best_idx = i
                
                if best_idx != -1:
                    # Add this point to trajectory
                    point = good_new_filtered[best_idx].flatten()
                    
                    # Only continue trajectory if point has moved enough recently
                    # or if trajectory is new (less than 3 points)
                    if len(trajectory) < 3 or min_dist >= self.min_movement_threshold:
                        self.active_trajectories[traj_idx].append((int(point[0]), int(point[1])))
                        current_positions.append((int(point[0]), int(point[1])))
                        used_points.add(best_idx)
                    else:
                        # Mark trajectory for potential removal - point isn't moving enough
                        if len(trajectory) > 5:  # Only save if it has at least 5 points
                            self.trajectories.append(trajectory)
                        # Clear this trajectory
                        self.active_trajectories[traj_idx] = []
            
            # Add new trajectories if needed and we haven't reached max_targets
            active_count = sum(1 for traj in self.active_trajectories if len(traj) > 0)
            if active_count < self.max_targets:
                for i, point in enumerate(good_new_filtered):
                    if i in used_points:
                        continue
                    if active_count >= self.max_targets:
                        break
                    
                    point_tuple = (int(point[0]), int(point[1]))
                    
                    # Only start new trajectory if in motion area
                    point_in_motion = False
                    if diff_mask is not None:
                        h, w = diff_mask.shape
                        if 0 <= point_tuple[1] < h and 0 <= point_tuple[0] < w:
                            if diff_mask[point_tuple[1], point_tuple[0]] > 0:
                                point_in_motion = True
                    
                    if point_in_motion:
                        self.active_trajectories.append([point_tuple])
                        current_positions.append(point_tuple)
                        active_count += 1
        
        # Clean up finished trajectories
        active_count = 0
        for i, trajectory in enumerate(self.active_trajectories):
            if len(trajectory) > 0:
                active_count += 1
        
        # If we have too many active trajectories, save completed ones
        while active_count > self.max_targets:
            # Find shortest one to remove
            min_len = float('inf')
            min_idx = -1
            for i, trajectory in enumerate(self.active_trajectories):
                if len(trajectory) > 0 and len(trajectory) < min_len:
                    min_len = len(trajectory)
                    min_idx = i
            
            if min_idx != -1:
                # Save this trajectory if it's long enough
                if len(self.active_trajectories[min_idx]) > 5:  # Only save if it has at least 5 points
                    self.trajectories.append(self.active_trajectories[min_idx])
                
                # Remove from active
                self.active_trajectories[min_idx] = []
                active_count -= 1
        
        # Update for next frame
        self.prev_gray = curr_gray.copy()
        
        # Use filtered points for next tracking iteration
        if len(good_new_filtered) > 0:
            self.p0 = good_new_filtered.reshape(-1, 1, 2)
        elif len(good_new) > 0:
            # Fallback to all points if no moving points found
            self.p0 = good_new.reshape(-1, 1, 2)
        
        self.frame_count += 1
        
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
        canvas[:, w:] = (30, 30, 30)  # Dark gray background
        
        # Add motion detection visualization (frame difference and threshold mask)
        canvas, y_offset = self.motion_detector.visualize(frame, canvas, w, debug_width, debug_height)
        
        # Draw features and tracking results on main frame
        if self.p0 is not None:
            # Draw all feature points
            for i, pt in enumerate(self.p0):
                x, y = pt[0]
                
                # Determine if this point is moving or stagnant
                is_stagnant = False
                stagnant_counter = 0
                
                if i < len(self.stagnant_frames_counter):
                    stagnant_counter = self.stagnant_frames_counter[i]
                    is_stagnant = stagnant_counter >= self.stagnant_frames_limit
                
                # Color based on movement status
                if is_stagnant:
                    # Red for static points (not considered insects)
                    color = (0, 0, 255)
                    size = 2
                else:
                    # Yellow for active points, brighter for more movement
                    movement_factor = 1.0 - (stagnant_counter / max(self.stagnant_frames_limit, 1))
                    color = (0, 255 * movement_factor, 255)
                    size = 3
                
                cv2.circle(canvas, (int(x), int(y)), size, color, -1)
        
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
        
        # Add parameter information in a box
        info_y = y_offset + 20
        
        # Add a semi-transparent background box for parameters
        param_bg_height = 170  # Height of parameter box (increased)
        cv2.rectangle(canvas, (w+5, info_y-5), (w+debug_width-5, info_y+param_bg_height), (40, 40, 40), -1)
        cv2.rectangle(canvas, (w+5, info_y-5), (w+debug_width-5, info_y+param_bg_height), (100, 100, 100), 1)
        
        cv2.putText(canvas, "Parameters:", (w + 10, info_y + 15), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        param_info = [
            f"maxCorners: {self.feature_params['maxCorners']}",
            f"qualityLevel: {self.feature_params['qualityLevel']}",
            f"minDistance: {self.feature_params['minDistance']}",
            f"blockSize: {self.feature_params['blockSize']}",
            f"winSize: {self.lk_params['winSize']}",
            f"maxLevel: {self.lk_params['maxLevel']}",
            f"Min Movement: {self.min_movement_threshold} px",
            f"Stagnant Limit: {self.stagnant_frames_limit} frames",
            f"Refresh Interval: {self.refresh_features_interval} frames"
        ]
        
        for i, info in enumerate(param_info):
            cv2.putText(canvas, info, (w + 15, info_y + 40 + i * 15), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Add count of active trajectories
        active_count = sum(1 for traj in self.active_trajectories if len(traj) > 0)
        cv2.putText(canvas, f"Active trajectories: {active_count}", 
                  (w + 10, info_y + 40 + len(param_info) * 15 + 10), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Add frame count to top-left corner
        cv2.putText(canvas, f"Frame: {self.frame_count}", 
                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Add legend for point colors
        legend_y = info_y + param_bg_height + 10
        cv2.putText(canvas, "Point Legend:", (w + 10, legend_y), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Moving point sample
        cv2.circle(canvas, (w + 30, legend_y + 20), 3, (0, 255, 255), -1)
        cv2.putText(canvas, "Moving Point", (w + 40, legend_y + 25), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Static point sample
        cv2.circle(canvas, (w + 30, legend_y + 40), 2, (0, 0, 255), -1)
        cv2.putText(canvas, "Static Point (ignored)", (w + 40, legend_y + 45), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Current position sample
        cv2.circle(canvas, (w + 30, legend_y + 60), 5, (0, 0, 255), -1)
        cv2.putText(canvas, "Trajectory End", (w + 40, legend_y + 65), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return canvas 