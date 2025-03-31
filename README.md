# Insect Tracker

A computer vision system for tracking small, fast-moving insects in videos. The system implements four different tracking methods, each using traditional computer vision techniques without deep learning.

## Features

- Four tracking methods:
  1. **Sparse Optical Flow**: Uses frame differencing and Lucas-Kanade optical flow for feature point tracking
  2. **Dense Optical Flow**: Uses Farneback dense optical flow for full-frame motion analysis
  3. **Background Subtraction**: Uses MOG2 background subtraction for moving object detection
  4. **Feature Matching with Particle Filter**: Combines ORB feature matching and particle filtering for robust tracking

- Shared motion detection module for consistent frame differencing
- Customizable number of targets to track
- Real-time visualization
- Output video generation with tracking visualization
- Trajectory visualization with color-coded paths

## Code Structure

- `insect_tracker.py`: Main module for processing videos with different tracking methods
- `motion_detection.py`: Common motion detection utilities using frame differencing
- `sparse_optical_flow_tracker.py`: Implements sparse optical flow tracking
- `dense_optical_flow_tracker.py`: Implements dense optical flow tracking
- `background_subtraction_tracker.py`: Implements background subtraction tracking
- `feature_particle_tracker.py`: Implements feature matching with particle filter
- `compare_methods.py`: Utility script for comparing all tracking methods on the same video
- `tune_parameters.py`: Tool for tuning tracking parameters

## Requirements

- Python 3.6+
- OpenCV (`opencv-python`)
- NumPy
- Matplotlib

Install requirements with:

```bash
pip install opencv-python numpy matplotlib
```

## Usage

Basic usage:

```bash
python insect_tracker.py /path/to/your/video.mp4
```

### Command Line Arguments

| Argument | Description |
|----------|-------------|
| `video_path` | Path to the input video file (required) |
| `--method` | Tracking method to use: 'sparse', 'dense', 'background', or 'particle' (default: 'sparse') |
| `--max-targets` | Maximum number of targets to track (default: 5) |
| `--output` | Path to save the output video (default: None, doesn't save) |
| `--no-display` | Do not display video during processing |
| `--no-trajectory` | Do not save trajectory visualization |

### Examples

Track using sparse optical flow with up to 3 targets:

```bash
python insect_tracker.py video.mp4 --method sparse --max-targets 3
```

Track using background subtraction and save the output:

```bash
python insect_tracker.py video.mp4 --method background --output output.mp4
```

Track using particle filter without displaying the video:

```bash
python insect_tracker.py video.mp4 --method particle --no-display
```

## Output

- **Video**: Annotated video with tracking visualization if `--output` is specified
- **Trajectory Image**: PNG image showing the trajectories of all tracked insects
- **Console Output**: Summary of tracking results including number of trajectories found

## Component Details

### Motion Detection Module

The `MotionDetector` class in `motion_detection.py` provides common motion detection functionality:

- Frame differencing with configurable interval between reference frames
- Short-term and long-term motion detection for improved sensitivity
- Adaptive reference frame updates for handling new objects entering the frame
- Threshold-based motion detection
- Morphological operations for noise removal
- Contour detection and extraction of motion centers
- Optional artificial motion simulation for testing
- Visualization utilities for debugging

### Tracking Methods

#### 1. Sparse Optical Flow

Best for: Videos with clear, distinct features on insects, moderate motion

This method:
- Uses the motion detector to identify areas with motion
- Extracts good features to track in motion areas
- Tracks these features using Lucas-Kanade optical flow
- Builds trajectories by connecting tracked points over time
- **Actively detects new targets entering the frame**
- **Distinguishes between moving and static points**
- **Focuses on truly moving objects while ignoring static background features**
- **Periodically refreshes feature points to maintain tracking accuracy**
- **Filters out points that remain static for too long (non-insect objects)**
- Can simulate artificial insect movement for testing

Parameters include:
- Standard optical flow parameters (maxCorners, qualityLevel, etc.)
- `min_movement_threshold`: Minimum pixel movement required to consider a point in motion
- `stagnant_frames_limit`: Maximum frames a point can remain static before being discarded
- `refresh_features_interval`: How frequently to refresh feature points

#### 2. Dense Optical Flow

Best for: Videos with subtle motion, small insects with minimal texture

This method:
- Calculates dense optical flow for every pixel using Farneback's method
- Filters motion by magnitude threshold
- Extracts contours and centroids from significant motion areas
- Builds trajectories by associating centroids between frames

#### 3. Background Subtraction

Best for: Videos with stable backgrounds, good contrast between insects and background

This method:
- Uses MOG2 background subtraction to isolate moving objects
- Performs morphological operations to clean up the foreground mask
- Extracts contours and centroids from the foreground
- Builds trajectories by tracking centroids over time

#### 4. Feature Matching with Particle Filter

Best for: Complex scenes, fast and erratic insect movement, partial occlusions

This method:
- Uses the motion detector to roughly locate motion areas
- Extracts ORB features and matches them between frames
- Initializes particle filters for each potential target
- Uses particle filter prediction and update steps for robust tracking
- Builds trajectories from filtered positions