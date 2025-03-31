#!/usr/bin/env python3
"""
Compare all four insect tracking methods on the same video.
This script processes a video with all four methods and saves outputs for comparison.
"""

import os
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Import insect tracker module
from insect_tracker import process_video

# Available tracking methods
METHODS = ['sparse', 'dense', 'background', 'particle']

def run_all_methods(video_path, max_targets=5, display=False, params_dict=None):
    """
    Run all four tracking methods on the same video.
    """
    # Create output directory if needed
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = f"results_{video_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    # Process with each method
    for method in METHODS:
        print(f"\n{'='*50}")
        print(f"Processing with {method.upper()} method")
        print(f"{'='*50}")
        
        # Output paths
        output_video = os.path.join(output_dir, f"{video_name}_{method}.mp4")
        
        # Get parameters for this method if provided
        params = None
        if params_dict and method in params_dict:
            params = params_dict[method]
        
        # Run tracking
        start_time = time.time()
        trajectories = process_video(
            video_path,
            method,
            max_targets,
            output_video,
            display,
            True,
            params
        )
        elapsed_time = time.time() - start_time
        
        # Record results
        results[method] = {
            'trajectories': trajectories,
            'output_video': output_video,
            'trajectory_image': os.path.splitext(output_video)[0] + '_trajectory.png',
            'trajectory_plt': os.path.splitext(output_video)[0] + '_trajectory_plt.png',
            'processing_time': elapsed_time
        }
    
    # Generate comparison visualization
    create_comparison_image(results, output_dir, video_name)
    
    # Print performance summary
    print("\n\nPerformance Summary:")
    print(f"{'Method':<12} | {'Processing Time (s)':<20} | {'Trajectories Found':<20}")
    print(f"{'-'*12} | {'-'*20} | {'-'*20}")
    for method in METHODS:
        print(f"{method:<12} | {results[method]['processing_time']:.2f} | {len(results[method]['trajectories'])}")
    
    return results

def create_comparison_image(results, output_dir, video_name):
    """
    Create a comparison image showing trajectories from all methods.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, method in enumerate(METHODS):
        # Load trajectory image
        img_path = results[method]['trajectory_plt']
        if os.path.exists(img_path):
            img = mpimg.imread(img_path)
            axes[i].imshow(img)
            axes[i].set_title(f"{method.capitalize()} Method\n{len(results[method]['trajectories'])} trajectories, {results[method]['processing_time']:.1f}s")
            axes[i].axis('off')
        else:
            axes[i].text(0.5, 0.5, f"No trajectory image for {method}", 
                        ha='center', va='center')
            axes[i].axis('off')
    
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, f"{video_name}_comparison.png")
    plt.savefig(comparison_path, dpi=300)
    plt.close()
    
    print(f"\nSaved method comparison to {comparison_path}")

# Example usage - edit these variables to match your needs
if __name__ == '__main__':
    # Configuration
    video_path = 'your_video.mp4'   # Path to your insect video
    max_targets = 5                 # Maximum number of insects to track
    display_video = True            # Whether to display the video during processing
    
    # Custom parameters for each method (optional)
    params_dict = {
        'sparse': {
            'maxCorners': 100,
            'qualityLevel': 0.3,
            'minDistance': 7,
            'blockSize': 7,
            'winSize': (15, 15),
            'maxLevel': 2,
            'diff_thresh_value': 25
        },
        'dense': {
            'mag_thresh': 2.0,
            'min_area': 50,
        },
        'background': {
            'history': 500,
            'var_threshold': 16,
            'detect_shadows': True,
        },
        'particle': {
            'diff_thresh_value': 25,
            'min_area': 30,
            'nfeatures': 500,
            'num_particles': 100,
        }
    }
    
    # Run all methods
    results = run_all_methods(video_path, max_targets, display_video, params_dict) 