#!/usr/bin/env python3
"""
Tune parameters for a specific insect tracking method.
This script allows testing different parameter combinations for a chosen method.
"""

import os
import time
import cv2
import numpy as np
from insect_tracker import process_video

def test_parameters(video_path, method, max_targets, parameter_sets, output_dir=None):
    """
    Test multiple parameter sets on the same video with the specified tracking method.
    
    Args:
        video_path (str): Path to the input video
        method (str): Tracking method to use ('sparse', 'dense', 'background', 'particle')
        max_targets (int): Maximum number of targets to track
        parameter_sets (list): List of parameter dictionaries to test
        output_dir (str, optional): Directory to save results, defaults to 'tuning_results'
    
    Returns:
        list: Results for each parameter set
    """
    # Create output directory if needed
    if output_dir is None:
        output_dir = 'tuning_results'
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    method_dir = os.path.join(output_dir, f"{video_name}_{method}")
    os.makedirs(method_dir, exist_ok=True)
    
    results = []
    
    # Test each parameter set
    for i, params in enumerate(parameter_sets):
        set_name = f"set_{i+1}"
        print(f"\n{'='*50}")
        print(f"Testing {method.upper()} with parameter set {i+1}")
        print(f"Parameters: {params}")
        print(f"{'='*50}")
        
        # Output paths
        output_video = os.path.join(method_dir, f"{set_name}.mp4")
        
        # Run tracking
        start_time = time.time()
        trajectories = process_video(
            video_path,
            method,
            max_targets,
            output_video,
            False,  # Don't show video during batch processing
            True,   # Save trajectory
            params
        )
        elapsed_time = time.time() - start_time
        
        # Record results
        result = {
            'set_name': set_name,
            'params': params,
            'trajectories': trajectories,
            'trajectory_count': len(trajectories),
            'output_video': output_video,
            'trajectory_image': os.path.splitext(output_video)[0] + '_trajectory.png',
            'processing_time': elapsed_time
        }
        
        results.append(result)
        
        print(f"Parameter set {i+1} completed in {elapsed_time:.2f} seconds.")
        print(f"Found {len(trajectories)} trajectories.")
    
    # Compare results
    print("\n\nParameter Tuning Results:")
    print(f"{'Set':<8} | {'Trajectories':<12} | {'Time (s)':<10} | {'Parameters'}")
    print(f"{'-'*8} | {'-'*12} | {'-'*10} | {'-'*50}")
    
    for i, result in enumerate(results):
        # Format parameters as a compact string
        param_str = ', '.join([f"{k}:{v}" for k, v in result['params'].items()])
        
        print(f"Set {i+1:<5} | {result['trajectory_count']:<12} | {result['processing_time']:.2f} | {param_str[:50]}...")
    
    return results

# Example usage
if __name__ == '__main__':
    # Configuration
    video_path = 'your_video.mp4'  # Path to your insect video
    method = 'sparse'              # Method to tune ('sparse', 'dense', 'background', 'particle')
    max_targets = 5                # Maximum number of insects to track
    
    # Define parameter sets to test - modify these based on your video characteristics
    if method == 'sparse':
        # Testing different settings for sparse optical flow
        parameter_sets = [
            # Default parameters
            {
                'maxCorners': 100,
                'qualityLevel': 0.3, 
                'minDistance': 7,
                'blockSize': 7,
                'winSize': (15, 15),
                'maxLevel': 2,
                'diff_thresh_value': 25
            },
            # For very small insects
            {
                'maxCorners': 200,
                'qualityLevel': 0.1,
                'minDistance': 3,
                'blockSize': 3,
                'winSize': (11, 11),
                'maxLevel': 3,
                'diff_thresh_value': 15
            },
            # For faster moving insects
            {
                'maxCorners': 100,
                'qualityLevel': 0.3,
                'minDistance': 7,
                'blockSize': 7,
                'winSize': (21, 21),
                'maxLevel': 4,
                'diff_thresh_value': 25
            },
            # For noisy videos
            {
                'maxCorners': 50,
                'qualityLevel': 0.4,
                'minDistance': 10,
                'blockSize': 9,
                'winSize': (15, 15),
                'maxLevel': 2,
                'diff_thresh_value': 35
            }
        ]
    elif method == 'dense':
        # Testing different settings for dense optical flow
        parameter_sets = [
            # Default parameters
            {
                'mag_thresh': 2.0,
                'min_area': 50,
                'pyr_scale': 0.5,
                'levels': 3,
                'winsize': 15,
                'iterations': 3,
                'poly_n': 5,
                'poly_sigma': 1.2
            },
            # For subtle motion
            {
                'mag_thresh': 1.0,
                'min_area': 20,
                'pyr_scale': 0.5,
                'levels': 5,
                'winsize': 15,
                'iterations': 5,
                'poly_n': 5,
                'poly_sigma': 1.2
            },
            # For faster motion
            {
                'mag_thresh': 3.0,
                'min_area': 50,
                'pyr_scale': 0.5,
                'levels': 4,
                'winsize': 21,
                'iterations': 3,
                'poly_n': 7,
                'poly_sigma': 1.5
            },
            # For noisy videos
            {
                'mag_thresh': 4.0,
                'min_area': 80,
                'pyr_scale': 0.5,
                'levels': 3,
                'winsize': 15,
                'iterations': 2,
                'poly_n': 7,
                'poly_sigma': 1.8
            }
        ]
    elif method == 'background':
        # Testing different settings for background subtraction
        parameter_sets = [
            # Default parameters
            {
                'history': 500,
                'var_threshold': 16,
                'detect_shadows': True,
                'min_area': 50
            },
            # Fast adaptation to background changes
            {
                'history': 100,
                'var_threshold': 16,
                'detect_shadows': False,
                'min_area': 50
            },
            # More sensitive detection
            {
                'history': 500,
                'var_threshold': 8,
                'detect_shadows': True,
                'min_area': 30
            },
            # Less sensitive (noisy videos)
            {
                'history': 500,
                'var_threshold': 32,
                'detect_shadows': False,
                'min_area': 100
            }
        ]
    elif method == 'particle':
        # Testing different settings for particle filter
        parameter_sets = [
            # Default parameters
            {
                'diff_thresh_value': 25,
                'min_area': 30,
                'nfeatures': 500,
                'feature_match_thresh': 30,
                'num_particles': 100,
                'particle_std': 5.0,
                'process_std': 2.0,
                'measurement_std': 10.0
            },
            # For erratic motion
            {
                'diff_thresh_value': 20,
                'min_area': 30,
                'nfeatures': 800,
                'feature_match_thresh': 35,
                'num_particles': 200,
                'particle_std': 8.0,
                'process_std': 5.0,
                'measurement_std': 15.0
            },
            # For slower, more predictable motion
            {
                'diff_thresh_value': 25,
                'min_area': 50,
                'nfeatures': 400,
                'feature_match_thresh': 25,
                'num_particles': 80,
                'particle_std': 3.0,
                'process_std': 1.0,
                'measurement_std': 5.0
            },
            # For very small insects
            {
                'diff_thresh_value': 15,
                'min_area': 20,
                'nfeatures': 1000,
                'feature_match_thresh': 40,
                'num_particles': 150,
                'particle_std': 4.0,
                'process_std': 2.5,
                'measurement_std': 12.0
            }
        ]
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Run parameter testing
    results = test_parameters(video_path, method, max_targets, parameter_sets)
    
    # Display recommendation
    best_result = max(results, key=lambda x: x['trajectory_count'])
    print(f"\nBest parameter set: Set {results.index(best_result) + 1}")
    print(f"Found {best_result['trajectory_count']} trajectories in {best_result['processing_time']:.2f} seconds")
    print(f"Recommended parameters:")
    for k, v in best_result['params'].items():
        print(f"  {k}: {v}")
    
    # After finding the best parameters, you can use them in your main tracking
    print("\nTo use these parameters in your tracking code:")
    print(f"params = {best_result['params']}")
    print("trajectories = process_video(video_path, method, max_targets, output_path, show_video, save_trajectory, params)") 