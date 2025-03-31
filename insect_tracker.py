import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Import our motion detector and tracking methods
from motion_detection import MotionDetector
from sparse_optical_flow_tracker import SparseOpticalFlowTracker
from dense_optical_flow_tracker import DenseOpticalFlowTracker
from background_subtraction_tracker import BackgroundSubtractionTracker
from feature_particle_tracker import FeatureParticleTracker

def get_tracker(method, max_targets, params=None):
    """
    Create and return the requested tracker instance with customized parameters.
    
    Args:
        method (str): Tracking method to use
        max_targets (int): Maximum number of targets to track
        params (dict): Custom parameters for the tracker
        
    Returns:
        object: Tracker instance
    """

    if method == 'sparse':
        tracker = SparseOpticalFlowTracker(max_targets=max_targets)
        if params:
            # Update parameters if provided
            if 'maxCorners' in params:
                tracker.feature_params['maxCorners'] = params['maxCorners']
            if 'qualityLevel' in params:
                tracker.feature_params['qualityLevel'] = params['qualityLevel']
            if 'minDistance' in params:
                tracker.feature_params['minDistance'] = params['minDistance']
            if 'blockSize' in params:
                tracker.feature_params['blockSize'] = params['blockSize']
            if 'winSize' in params:
                tracker.lk_params['winSize'] = params['winSize']
            if 'maxLevel' in params:
                tracker.lk_params['maxLevel'] = params['maxLevel']
            
            # Motion detector parameters
            motion_detector_updated = False
            if 'diff_thresh_value' in params:
                tracker.diff_thresh_value = params['diff_thresh_value']
                motion_detector_updated = True
            if 'frame_diff_interval' in params:
                tracker.frame_diff_interval = params['frame_diff_interval']
                motion_detector_updated = True
            if 'enable_motion_simulation' in params:
                motion_detector_updated = True
            
            # Motion detection thresholds
            if 'min_movement_threshold' in params:
                tracker.min_movement_threshold = params['min_movement_threshold']
            if 'stagnant_frames_limit' in params:
                tracker.stagnant_frames_limit = params['stagnant_frames_limit']
            if 'refresh_features_interval' in params:
                tracker.refresh_features_interval = params['refresh_features_interval']
            
            # Update motion detector if any of its parameters changed
            if motion_detector_updated:
                min_area = params.get('min_area', 30)
                tracker.motion_detector = MotionDetector(
                    diff_thresh_value=tracker.diff_thresh_value,
                    min_area=min_area,
                    frame_diff_interval=tracker.frame_diff_interval
                )
                if 'enable_motion_simulation' in params:
                    tracker.motion_detector.set_motion_simulation(params['enable_motion_simulation'])
                else:
                    tracker.motion_detector.set_motion_simulation(True)
        return tracker

    elif method == 'dense':
        tracker = DenseOpticalFlowTracker(max_targets=max_targets)
        if params:
            # Update parameters if provided
            if 'mag_thresh' in params:
                tracker.mag_thresh = params['mag_thresh']
            if 'min_area' in params:
                tracker.min_area = params['min_area']
            if 'pyr_scale' in params:
                tracker.flow_params['pyr_scale'] = params['pyr_scale']
            if 'levels' in params:
                tracker.flow_params['levels'] = params['levels']
            if 'winsize' in params:
                tracker.flow_params['winsize'] = params['winsize']
            if 'iterations' in params:
                tracker.flow_params['iterations'] = params['iterations']
            if 'poly_n' in params:
                tracker.flow_params['poly_n'] = params['poly_n']
            if 'poly_sigma' in params:
                tracker.flow_params['poly_sigma'] = params['poly_sigma']
        return tracker
    
    elif method == 'background':
        tracker = BackgroundSubtractionTracker(max_targets=max_targets)
        if params:
            # Update parameters if provided
            if 'history' in params:
                tracker.history = params['history']
                # Recreate background subtractor with new parameters
                tracker.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                    history=tracker.history,
                    varThreshold=tracker.var_threshold,
                    detectShadows=tracker.detect_shadows
                )
            if 'var_threshold' in params:
                tracker.var_threshold = params['var_threshold']
                # Recreate background subtractor with new parameters
                tracker.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                    history=tracker.history,
                    varThreshold=tracker.var_threshold,
                    detectShadows=tracker.detect_shadows
                )
            if 'detect_shadows' in params:
                tracker.detect_shadows = params['detect_shadows']
                # Recreate background subtractor with new parameters
                tracker.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                    history=tracker.history,
                    varThreshold=tracker.var_threshold,
                    detectShadows=tracker.detect_shadows
                )
            if 'min_area' in params:
                tracker.min_area = params['min_area']
        return tracker
    
    elif method == 'particle':
        tracker = FeatureParticleTracker(max_targets=max_targets)
        if params:
            # Update parameters if provided
            # Motion detector parameters
            motion_detector_updated = False
            if 'diff_thresh_value' in params:
                tracker.diff_thresh_value = params['diff_thresh_value']
                motion_detector_updated = True
            if 'min_area' in params:
                tracker.min_area = params['min_area']
                motion_detector_updated = True
            if 'frame_diff_interval' in params:
                motion_detector_updated = True
            if 'enable_motion_simulation' in params:
                motion_detector_updated = True
                
            # Update motion detector if any of its parameters changed
            if motion_detector_updated:
                frame_diff_interval = params.get('frame_diff_interval', 1)
                tracker.motion_detector = MotionDetector(
                    diff_thresh_value=tracker.diff_thresh_value,
                    min_area=tracker.min_area,
                    frame_diff_interval=frame_diff_interval
                )
                if 'enable_motion_simulation' in params:
                    tracker.motion_detector.set_motion_simulation(params['enable_motion_simulation'])
            
            # ORB feature parameters
            if 'nfeatures' in params:
                tracker.nfeatures = params['nfeatures']
                # Recreate ORB detector with new parameters
                tracker.orb = cv2.ORB_create(nfeatures=tracker.nfeatures)
            if 'feature_match_thresh' in params:
                tracker.feature_match_thresh = params['feature_match_thresh']
            
            # Particle filter parameters
            if 'num_particles' in params:
                tracker.num_particles = params['num_particles']
            if 'particle_std' in params:
                tracker.particle_std = params['particle_std']
            if 'process_std' in params:
                tracker.process_std = params['process_std']
            if 'measurement_std' in params:
                tracker.measurement_std = params['measurement_std']
        return tracker
    
    else:
        raise ValueError(f"Unknown tracking method: {method}")

def process_video(video_path, method, max_targets, output_path=None, show_video=True, save_trajectory=True, params=None):
    """
    Process a video using the specified tracker.
    
    Args:
        video_path (str): Path to the input video
        method (str): Tracking method to use
        max_targets (int): Maximum number of targets to track
        output_path (str, optional): Path to save the output video
        show_video (bool): Whether to display the video during processing
        save_trajectory (bool): Whether to save the trajectory image
        params (dict): Custom parameters for the tracker
        
    Returns:
        list: List of trajectories for each tracked insect
    """
    # Get the appropriate tracker
    tracker = get_tracker(method, max_targets, params)
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create output video writer if needed
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Process each frame
    positions_history = []
    
    print(f"Processing video using {method} tracker with max {max_targets} targets...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process the frame with the tracker
        positions = tracker.process_frame(frame)
        positions_history.append(positions)
        
        # Visualize the results
        vis_frame = tracker.visualize(frame, positions)
        
        # Add tracker type to the frame
        cv2.putText(
            vis_frame,
            f"Method: {method} | Max targets: {max_targets}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )
        
        # Write to output video if needed
        if out:
            out.write(vis_frame)
        
        # Show the frame if requested
        if show_video:
            cv2.imshow('Insect Tracking', vis_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                break
    
    # Get all trajectories
    trajectories = tracker.get_all_trajectories()
    
    # Clean up
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    
    # Generate and save the trajectory visualization
    if save_trajectory:
        trajectory_path = os.path.splitext(output_path)[0] + '_trajectory.png' if output_path else 'trajectory.png'
        generate_trajectory_image(trajectories, frame_width, frame_height, trajectory_path)
    
    return trajectories

def generate_trajectory_image(trajectories, width, height, output_path):
    """
    Generate an image showing all trajectories.
    
    Args:
        trajectories (list): List of trajectories
        width (int): Image width
        height (int): Image height
        output_path (str): Path to save the image
    """
    # Create a black image
    trajectory_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create colormap for trajectories
    colors = plt.cm.jet(np.linspace(0, 1, len(trajectories)))
    
    # Draw each trajectory with its own color
    for i, trajectory in enumerate(trajectories):
        if len(trajectory) < 2:
            continue
            
        color = (
            int(colors[i][2] * 255),  # B
            int(colors[i][1] * 255),  # G
            int(colors[i][0] * 255)   # R
        )
        
        for j in range(1, len(trajectory)):
            cv2.line(
                trajectory_img,
                trajectory[j-1],
                trajectory[j],
                color,
                2
            )
            
        # Mark the start and end of each trajectory
        cv2.circle(trajectory_img, trajectory[0], 5, (0, 255, 0), -1)       # Start point (green)
        cv2.circle(trajectory_img, trajectory[-1], 5, (0, 0, 255), -1)      # End point (red)
    
    # Save the trajectory image
    cv2.imwrite(output_path, trajectory_img)
    print(f"Saved trajectory visualization to {output_path}")
    
    # Also display using matplotlib for better visualization
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(trajectory_img, cv2.COLOR_BGR2RGB))
    plt.title('Insect Trajectories')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path.replace('.png', '_plt.png'), dpi=300)
    plt.close()

# Example usage
if __name__ == '__main__':
    # Configuration variables (modify these as needed)
    video_path = '/Users/altis/Downloads/IMG_0316.MOV'  # Path to your video
    method = 'dense'              # Options: 'sparse', 'dense', 'background', 'particle'
    max_targets = 5                # Maximum number of insects to track
    output_path = None     # Path to save processed video, None to skip saving
    show_video = True              # Whether to display the video during processing
    save_trajectory = True         # Whether to save trajectory visualization
    
    # Custom parameters for each method (all optional)
    # You can add or remove parameters as needed

    # For sparse optical flow method
    sparse_params = {
        'maxCorners': 100,         # Maximum number of corners to detect (10-300)
        'qualityLevel': 0.3,       # Minimum quality of corner (0.01-0.5)
        'minDistance': 7,          # Minimum distance between corners (3-20)
        'blockSize': 7,            # Size of window for corner detection (3-15)
        'winSize': (15, 15),       # Size of search window for optical flow (5-31, odd numbers)
        'maxLevel': 2,             # Maximum pyramid level for optical flow (0-5)
        'diff_thresh_value': 10,   # Threshold for frame differencing (lowered from 25 to 10)
        'frame_diff_interval': 5,  # Calculate frame difference every N frames
        'enable_motion_simulation': True,  # Enable artifical motion simulation
        'min_movement_threshold': 3,  # Minimum movement in pixels to consider point moving
        'stagnant_frames_limit': 5,  # Max frames a point can be static before rejection
        'refresh_features_interval': 15  # Refresh features every N frames
    }
    
    # For dense optical flow method
    dense_params = {
        'mag_thresh': 1,         # Motion magnitude threshold (0.5-10.0)
        'min_area': 50,            # Minimum contour area (10-200)
        'pyr_scale': 0.5,          # Image scale (<1) to build pyramids (0.1-0.9)
        'levels': 3,               # Number of pyramid layers (1-10)
        'winsize': 15,             # Window size for flow calculation (5-31, odd numbers)
        'iterations': 3,           # Number of iterations (1-10)
        'poly_n': 5,               # Size of pixel neighborhood (3-9)
        'poly_sigma': 1.2          # Gaussian sigma (0.5-2.0)
    }
    
    # For background subtraction method
    background_params = {
        'history': 500,            # Number of frames for background model (50-1000)
        'var_threshold': 16,       # Threshold for background/foreground decision (4-64)
        'detect_shadows': True,    # Whether to detect shadows (True/False)
        'min_area':20             # Minimum contour area (10-200)
    }
    
    # For feature matching with particle filter method
    particle_params = {
        'diff_thresh_value': 25,   # Threshold for frame differencing (10-50)
        'min_area': 30,            # Minimum contour area (10-200)
        'frame_diff_interval': 1,  # Use every frame for feature matching
        'enable_motion_simulation': True,  # Enable artificial motion simulation
        'nfeatures': 500,          # Number of features to extract (100-2000)
        'feature_match_thresh': 30,# Threshold for feature matching (10-60)
        'num_particles': 100,      # Number of particles (50-500)
        'particle_std': 5.0,       # Standard deviation for particle distribution (1.0-20.0)
        'process_std': 2.0,        # Process noise (0.5-10.0)
        'measurement_std': 10.0    # Measurement noise (1.0-30.0)
    }
    
    # Select which parameter set to use based on chosen method
    params = None
    if method == 'sparse':
        params = sparse_params
    elif method == 'dense':
        params = dense_params
    elif method == 'background':
        params = background_params
    elif method == 'particle':
        params = particle_params
    
    # Process the video
    trajectories = process_video(
        video_path,
        method,
        max_targets,
        output_path,
        show_video,
        save_trajectory,
        params
    )
    
    # Print summary
    print(f"Tracking completed. Found {len(trajectories)} trajectories.")
    for i, trajectory in enumerate(trajectories):
        print(f"Trajectory {i+1}: {len(trajectory)} points") 