import cv2
import numpy as np
from motion_detection import MotionDetector

class FeatureParticleTracker:
    """
    Tracks small fast-moving insects using feature matching combined with particle filtering.
    
    This tracker uses the following steps:
    1. Frame differencing to roughly locate motion areas
    2. ORB feature extraction and matching between frames
    3. Particle filter initialization for each potential target
    4. Particle filter prediction and update using feature matches and motion information
    5. Trajectory building from filtered positions
    """
    
    def __init__(self, max_targets=10):
        """
        Initialize tracker with parameters.
        
        Args:
            max_targets (int): Maximum number of targets to track
        """
        self.max_targets = max_targets
        
        # Parameters for frame differencing
        self.diff_thresh_value = 25
        self.min_area = 30
        
        # Initialize motion detector
        self.motion_detector = MotionDetector(
            diff_thresh_value=self.diff_thresh_value,
            min_area=self.min_area,
            frame_diff_interval=1  # Use every frame for feature matching
        )
        
        # Parameters for ORB feature extraction
        self.nfeatures = 500
        self.feature_match_thresh = 30  # Distance threshold for feature matching
        
        # Parameters for particle filter
        self.num_particles = 100
        self.particle_std = 5.0  # Standard deviation for particle distribution
        self.process_std = 2.0   # Process noise in motion model
        self.measurement_std = 10.0  # Measurement noise
        
        # Initialize ORB detector and matcher
        self.orb = cv2.ORB_create(nfeatures=self.nfeatures)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Initialize variables
        self.prev_gray = None
        self.prev_kp = None
        self.prev_des = None
        self.trajectories = []
        self.particle_filters = {}  # Dict of active particle filters
        self.next_target_id = 0
        self.frame_count = 0
        
    def initialize(self, first_frame):
        """
        Initialize the tracker with the first frame.
        
        Args:
            first_frame (numpy.ndarray): First frame of the video
        """
        self.prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        self.motion_detector.initialize(first_frame)
        self.prev_kp, self.prev_des = self.orb.detectAndCompute(self.prev_gray, None)
        self.trajectories = []
        self.particle_filters = {}
        self.next_target_id = 0
        self.frame_count = 0
        
    def initialize_particles(self, center):
        """
        Initialize particles around a given center point.
        
        Args:
            center (tuple): Center point (x, y)
            
        Returns:
            tuple: (particles, weights)
        """
        particles = np.empty((self.num_particles, 4))  # x, y, vx, vy
        particles[:, 0] = center[0] + np.random.randn(self.num_particles) * self.particle_std
        particles[:, 1] = center[1] + np.random.randn(self.num_particles) * self.particle_std
        particles[:, 2] = np.random.randn(self.num_particles) * 2  # Initial velocity
        particles[:, 3] = np.random.randn(self.num_particles) * 2  # Initial velocity
        weights = np.ones(self.num_particles) / self.num_particles
        return particles, weights
    
    def predict_particles(self, particles):
        """
        Predict new particle positions using constant velocity model with noise.
        
        Args:
            particles (numpy.ndarray): Current particles
            
        Returns:
            numpy.ndarray: Predicted particles
        """
        # Apply motion model: x' = x + vx, y' = y + vy
        particles[:, 0] += particles[:, 2]
        particles[:, 1] += particles[:, 3]
        
        # Add process noise
        particles[:, 0] += np.random.randn(self.num_particles) * self.process_std
        particles[:, 1] += np.random.randn(self.num_particles) * self.process_std
        particles[:, 2] += np.random.randn(self.num_particles) * self.process_std * 0.5
        particles[:, 3] += np.random.randn(self.num_particles) * self.process_std * 0.5
        
        return particles
    
    def update_weights(self, particles, weights, observation):
        """
        Update particle weights based on observation.
        
        Args:
            particles (numpy.ndarray): Current particles
            weights (numpy.ndarray): Current weights
            observation (tuple): Observed position (x, y)
            
        Returns:
            numpy.ndarray: Updated weights
        """
        # Calculate likelihood using Gaussian model
        dx = particles[:, 0] - observation[0]
        dy = particles[:, 1] - observation[1]
        dist = np.sqrt(dx**2 + dy**2)
        
        # Likelihood is proportional to exp(-dist²/(2*σ²))
        likelihood = np.exp(-0.5 * (dist**2) / (self.measurement_std**2))
        
        # Update weights
        weights = weights * likelihood
        
        # Normalize weights
        weights_sum = np.sum(weights)
        if weights_sum > 0:
            weights = weights / weights_sum
        else:
            weights = np.ones(self.num_particles) / self.num_particles
            
        return weights
    
    def resample_particles(self, particles, weights):
        """
        Resample particles according to weights.
        
        Args:
            particles (numpy.ndarray): Current particles
            weights (numpy.ndarray): Current weights
            
        Returns:
            tuple: (resampled_particles, equal_weights)
        """
        # Resample using systematic resampling
        indices = np.zeros(self.num_particles, dtype=np.int32)
        
        # Generate random starting point
        u0 = np.random.uniform(0, 1.0/self.num_particles)
        
        # Generate cumulative sum of weights
        cumulative_sum = np.cumsum(weights)
        
        # Systematic resampling
        for i in range(self.num_particles):
            u = u0 + i / self.num_particles
            while u > cumulative_sum[indices[i]]:
                indices[i] += 1
                if indices[i] >= self.num_particles - 1:
                    break
        
        # Create new particles from selected indices
        resampled_particles = particles[indices]
        
        # Reset weights to uniform
        equal_weights = np.ones(self.num_particles) / self.num_particles
        
        return resampled_particles, equal_weights
    
    def get_state_estimate(self, particles, weights):
        """
        Calculate weighted mean of particles to get position estimate.
        
        Args:
            particles (numpy.ndarray): Current particles
            weights (numpy.ndarray): Current weights
            
        Returns:
            tuple: (x, y) estimated position
        """
        x = int(np.sum(particles[:, 0] * weights))
        y = int(np.sum(particles[:, 1] * weights))
        return (x, y)
        
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
        _, diff_mask, candidate_centers = self.motion_detector.detect_motion(frame)
        
        # Detect ORB features in current frame
        curr_kp, curr_des = self.orb.detectAndCompute(curr_gray, mask=diff_mask)
        
        # Match features between frames if possible
        matches = []
        if curr_des is not None and self.prev_des is not None:
            matches = self.bf.match(self.prev_des, curr_des)
            
            # Filter matches by distance
            matches = [m for m in matches if m.distance < self.feature_match_thresh]
        
        # Process matched features to compute general motion
        matched_points = []
        if matches:
            for m in matches:
                prev_idx = m.queryIdx
                curr_idx = m.trainIdx
                prev_pt = self.prev_kp[prev_idx].pt
                curr_pt = curr_kp[curr_idx].pt
                matched_points.append((prev_pt, curr_pt))
        
        # Current positions to return
        current_positions = []
        
        # For each candidate center, check if it belongs to an existing target
        for center in candidate_centers:
            # Check if it's close to any existing particle filter's estimate
            assigned = False
            for target_id, (particles, weights) in self.particle_filters.items():
                est_position = self.get_state_estimate(particles, weights)
                if np.linalg.norm(np.array(center) - np.array(est_position)) < 20:
                    assigned = True
                    break
                    
            if not assigned and len(self.particle_filters) < self.max_targets:
                # Initialize new particle filter for this target
                self.particle_filters[self.next_target_id] = self.initialize_particles(center)
                
                # Initialize trajectory
                if len(self.particle_filters) <= self.max_targets:
                    self.next_target_id += 1
        
        # Update each active particle filter
        target_ids_to_remove = []
        for target_id, (particles, weights) in self.particle_filters.items():
            # Predict
            particles = self.predict_particles(particles)
            
            # Find closest candidate center as observation
            est_position = self.get_state_estimate(particles, weights)
            closest_center = None
            min_dist = float('inf')
            
            for center in candidate_centers:
                dist = np.linalg.norm(np.array(center) - np.array(est_position))
                if dist < min_dist and dist < 50:  # Limit observation distance
                    min_dist = dist
                    closest_center = center
            
            # Update weights based on observation if available
            if closest_center is not None:
                weights = self.update_weights(particles, weights, closest_center)
                
                # Resample particles
                particles, weights = self.resample_particles(particles, weights)
                
                # Get refined position estimate
                position = self.get_state_estimate(particles, weights)
                current_positions.append(position)
                
                # Update particle filter
                self.particle_filters[target_id] = (particles, weights)
            else:
                # No observation - target may have disappeared
                target_ids_to_remove.append(target_id)
        
        # Remove targets without observations
        for target_id in target_ids_to_remove:
            particles, weights = self.particle_filters[target_id]
            
            # Get last known position
            position = self.get_state_estimate(particles, weights)
            
            # Create trajectory and add to completed list
            trajectory = [position]  # This is simplified - real impl would store trajectory
            if len(trajectory) > 5:
                self.trajectories.append(trajectory)
                
            del self.particle_filters[target_id]
        
        # Update for next frame
        self.prev_gray = curr_gray.copy()
        self.prev_kp = curr_kp
        self.prev_des = curr_des
        
        return current_positions
    
    def get_all_trajectories(self):
        """
        Get all trajectories including active and completed ones.
        
        Returns:
            list: List of all trajectories
        """
        result = self.trajectories.copy()
        # Also include trajectories from active particle filters
        for target_id, (particles, weights) in self.particle_filters.items():
            # In a real implementation, we would store the trajectory for each filter
            # Here we just use the current estimate as a placeholder
            position = self.get_state_estimate(particles, weights)
            result.append([position])
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
        canvas[:, w:] = (30, 30, 30)  # Dark gray background
        
        # Add motion detection visualization
        canvas, y_offset = self.motion_detector.visualize(frame, canvas, w, debug_width, debug_height)
        
        # 3. Draw feature matches if available
        if self.prev_kp is not None and self.prev_des is not None:
            # Detect ORB features in current frame
            _, diff_mask, _ = self.motion_detector.detect_motion(frame)
            curr_kp, curr_des = self.orb.detectAndCompute(curr_gray, mask=diff_mask)
            
            # Draw keypoints on main image
            if curr_kp is not None:
                # Draw all detected keypoints in yellow
                for kp in curr_kp:
                    x, y = map(int, kp.pt)
                    cv2.circle(canvas, (x, y), 3, (0, 255, 255), 1)
            
            # Match features between frames if possible
            matched_points = []
            if curr_des is not None and self.prev_des is not None:
                matches = self.bf.match(self.prev_des, curr_des)
                
                # Filter matches by distance
                good_matches = [m for m in matches if m.distance < self.feature_match_thresh]
                
                # Draw matches in main view
                for m in good_matches:
                    prev_idx = m.queryIdx
                    curr_idx = m.trainIdx
                    
                    # Get coordinates of matched keypoints
                    prev_x, prev_y = map(int, self.prev_kp[prev_idx].pt)
                    curr_x, curr_y = map(int, curr_kp[curr_idx].pt)
                    
                    # Draw line connecting matches
                    cv2.line(canvas, (prev_x, prev_y), (curr_x, curr_y), (0, 165, 255), 1)
                    
                    # Save matched points
                    matched_points.append(((prev_x, prev_y), (curr_x, curr_y)))
        
        # Draw particle distributions for each target
        for target_id, (particles, weights) in self.particle_filters.items():
            # Get estimated position
            est_position = self.get_state_estimate(particles, weights)
            
            # Draw target ID near estimated position
            cv2.putText(canvas, f"ID: {target_id}", 
                      (est_position[0] + 10, est_position[1] - 10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Draw particles with opacity based on weight
            # First, normalize weights for visualization
            max_weight = np.max(weights)
            norm_weights = weights / max_weight if max_weight > 0 else weights
            
            # Draw particles, showing position and velocity
            for i in range(self.num_particles):
                # Get particle state
                x, y = int(particles[i, 0]), int(particles[i, 1])
                vx, vy = particles[i, 2], particles[i, 3]
                
                # Check if particle is within frame
                if 0 <= x < w and 0 <= y < h:
                    # Particle color based on weight (brighter = higher weight)
                    intensity = int(norm_weights[i] * 255)
                    color = (0, intensity, intensity)  # Yellow/green tint
                    
                    # Draw particle position
                    cv2.circle(canvas, (x, y), 1, color, -1)
                    
                    # Draw velocity vector for some particles (to avoid clutter)
                    if i % 20 == 0:  # Draw velocity for 5% of particles
                        # Draw velocity vector
                        end_x, end_y = int(x + vx), int(y + vy)
                        cv2.arrowedLine(canvas, (x, y), (end_x, end_y), color, 1, tipLength=0.3)
            
            # Draw circle at estimated position
            cv2.circle(canvas, est_position, 5, (0, 0, 255), -1)
            
            # Draw a larger circle to indicate search radius
            cv2.circle(canvas, est_position, 50, (255, 0, 127), 1)
        
        # Add info about feature matching and particle filter
        y_offset = y_offset + debug_height + 20
        cv2.putText(canvas, "Parameters:", (w + 10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        param_info = [
            f"diff_thresh: {self.diff_thresh_value}",
            f"min_area: {self.min_area}",
            f"nfeatures: {self.nfeatures}",
            f"feature_match_thresh: {self.feature_match_thresh}",
            f"num_particles: {self.num_particles}",
            f"process_std: {self.process_std}",
            f"measurement_std: {self.measurement_std}"
        ]
        
        for i, info in enumerate(param_info):
            cv2.putText(canvas, info, (w + 10, y_offset + 20 + i * 20), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
        # Show number of active particle filters
        cv2.putText(canvas, f"Active targets: {len(self.particle_filters)}", 
                   (w + 10, y_offset + 20 + len(param_info) * 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Add frame count to top-left corner
        if hasattr(self, 'frame_count'):
            self.frame_count += 1
        else:
            self.frame_count = 1
            
        cv2.putText(canvas, f"Frame: {self.frame_count}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return canvas 