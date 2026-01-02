#!/usr/bin/env python3
"""
Visualize tracking data overlaid on original video.

Draws tracked landmarks with different shapes for different body parts
and shows skeleton connections. Helps debug tracking and flip detection.

Can process a single camera or all takes in a session directory.

Usage:
  visualize_tracking.py <session_path>

Example:
  # Process all takes in a session
  visualize_tracking.py session/20251231

  # Process specific take
  visualize_tracking.py session/20251231/take1
"""

import argparse
import cv2
import h5py
import numpy as np
from pathlib import Path
import sys

# MediaPipe landmark groups with shapes and colors
LANDMARK_GROUPS = {
    'face': {
        'indices': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'shape': 'circle',
        'color': (255, 200, 0),  # Cyan
        'name': 'Face'
    },
    'left_arm': {
        'indices': [11, 13, 15, 17, 19, 21],
        'shape': 'square',
        'color': (0, 255, 0),  # Green
        'name': 'Left Arm'
    },
    'right_arm': {
        'indices': [12, 14, 16, 18, 20, 22],
        'shape': 'square',
        'color': (0, 0, 255),  # Red
        'name': 'Right Arm'
    },
    'left_leg': {
        'indices': [23, 25, 27, 29, 31],
        'shape': 'triangle',
        'color': (0, 255, 255),  # Yellow
        'name': 'Left Leg'
    },
    'right_leg': {
        'indices': [24, 26, 28, 30, 32],
        'shape': 'triangle',
        'color': (255, 0, 255),  # Magenta
        'name': 'Right Leg'
    }
}

# Skeleton connections
SKELETON_CONNECTIONS = [
    # Face
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
    (2, 7), (5, 8), (0, 9), (0, 10),

    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),

    # Left arm
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),

    # Right arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),

    # Left leg
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),

    # Right leg
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]


def load_tracking_data(h5_file):
    """Load tracking data from analysis.h5 file."""
    with h5py.File(h5_file, 'r') as f:
        # Load tracks dataset: (n_tracks, 2, n_nodes, n_frames)
        tracks = f['tracks'][:]

    # Reshape to (n_frames, n_nodes, 2)
    points = tracks[0].transpose(2, 1, 0)

    return points


def draw_landmark(frame, x, y, shape, color, size=8):
    """Draw a landmark with specified shape."""
    x, y = int(x), int(y)

    if shape == 'circle':
        cv2.circle(frame, (x, y), size, color, -1)
        cv2.circle(frame, (x, y), size, (255, 255, 255), 1)  # White outline

    elif shape == 'square':
        half = size
        cv2.rectangle(frame, (x-half, y-half), (x+half, y+half), color, -1)
        cv2.rectangle(frame, (x-half, y-half), (x+half, y+half), (255, 255, 255), 1)

    elif shape == 'triangle':
        half = size
        pts = np.array([
            [x, y-half],
            [x-half, y+half],
            [x+half, y+half]
        ], np.int32)
        cv2.fillPoly(frame, [pts], color)
        cv2.polylines(frame, [pts], True, (255, 255, 255), 1)


def draw_legend(frame, groups):
    """Draw legend showing landmark groups."""
    legend_x = 10
    legend_y = 30
    line_height = 30

    # Draw semi-transparent background
    cv2.rectangle(frame, (5, 5), (200, 5 + len(groups) * line_height + 10),
                 (0, 0, 0), -1)
    cv2.rectangle(frame, (5, 5), (200, 5 + len(groups) * line_height + 10),
                 (255, 255, 255), 1)

    for i, (group_name, group_info) in enumerate(groups.items()):
        y = legend_y + i * line_height

        # Draw example shape
        shape_x = legend_x + 15
        draw_landmark(frame, shape_x, y, group_info['shape'],
                     group_info['color'], size=6)

        # Draw label
        cv2.putText(frame, group_info['name'], (legend_x + 35, y + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def draw_skeleton(frame, points):
    """Draw skeleton connections between landmarks."""
    for start_idx, end_idx in SKELETON_CONNECTIONS:
        start = points[start_idx]
        end = points[end_idx]

        if not (np.isnan(start).any() or np.isnan(end).any()):
            pt1 = (int(start[0]), int(start[1]))
            pt2 = (int(end[0]), int(end[1]))
            cv2.line(frame, pt1, pt2, (100, 100, 100), 2)


def draw_frame_info(frame, frame_idx, total_frames):
    """Draw frame counter."""
    text = f"Frame: {frame_idx}/{total_frames}"
    cv2.putText(frame, text, (frame.shape[1] - 200, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def visualize_tracking(video_path, h5_path, output_path=None):
    """
    Create visualization video with tracking overlay.

    Args:
        video_path: Path to original video
        h5_path: Path to analysis.h5 file
        output_path: Output video path (default: input_video_tracked.mp4)
    """
    # Load tracking data
    print(f"Loading tracking data from: {h5_path}")
    tracking_points = load_tracking_data(h5_path)
    n_frames, n_nodes, _ = tracking_points.shape
    print(f"  Loaded {n_frames} frames with {n_nodes} landmarks")

    # Open video
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Error: Could not open video")
        return False

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  Video: {total_frames} frames, {fps:.2f} fps, {width}x{height}")

    # Setup output
    if output_path is None:
        video_path_obj = Path(video_path)
        output_path = video_path_obj.parent / f"{video_path_obj.stem}_tracked.mp4"

    print(f"Creating output: {output_path}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Process frames
    frame_idx = 0
    print(f"\nProcessing frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < len(tracking_points):
            points = tracking_points[frame_idx]

            # Draw skeleton connections first (behind landmarks)
            draw_skeleton(frame, points)

            # Draw landmarks by group
            for group_name, group_info in LANDMARK_GROUPS.items():
                for idx in group_info['indices']:
                    if idx < len(points):
                        pt = points[idx]
                        if not np.isnan(pt).any():
                            draw_landmark(frame, pt[0], pt[1],
                                        group_info['shape'],
                                        group_info['color'])

        # Draw legend and frame info
        draw_legend(frame, LANDMARK_GROUPS)
        draw_frame_info(frame, frame_idx, min(total_frames, len(tracking_points)))

        out.write(frame)
        frame_idx += 1

        if frame_idx % 100 == 0:
            print(f"  Processed {frame_idx}/{total_frames} frames...")

    cap.release()
    out.release()

    print(f"\n✓ Visualization complete!")
    print(f"Output: {output_path}")

    return True


def find_camera_pairs(session_path):
    """
    Find all camera video/analysis pairs in a session.

    Returns: List of (video_path, analysis_path) tuples
    """
    session_path = Path(session_path)
    pairs = []

    # Find all take directories (exclude calibration)
    take_dirs = [d for d in session_path.iterdir()
                 if d.is_dir() and d.name != 'calibration']

    if not take_dirs:
        # Maybe the path is already a take directory
        if session_path.name != 'calibration':
            take_dirs = [session_path]

    # For each take, find camera directories
    for take_dir in sorted(take_dirs):
        camera_dirs = [d for d in take_dir.iterdir() if d.is_dir()]

        for camera_dir in sorted(camera_dirs):
            # Find video file (*-view-recording.mp4)
            video_files = list(camera_dir.glob('*-view-recording.mp4'))

            # Find analysis file (*_analysis.h5)
            analysis_files = list(camera_dir.glob('*_analysis.h5'))

            if video_files and analysis_files:
                pairs.append((video_files[0], analysis_files[0]))
            elif video_files:
                print(f"Warning: Found video but no analysis file in {camera_dir}")
            elif analysis_files:
                print(f"Warning: Found analysis but no video file in {camera_dir}")

    return pairs


def process_session(session_path):
    """
    Process all takes in a session directory.

    Args:
        session_path: Path to session or take directory

    Returns:
        True if all successful, False otherwise
    """
    session_path = Path(session_path)

    if not session_path.exists():
        print(f"Error: Path not found: {session_path}")
        return False

    # Find all camera pairs
    pairs = find_camera_pairs(session_path)

    if not pairs:
        print(f"Error: No camera video/analysis pairs found in {session_path}")
        return False

    print(f"{'='*70}")
    print(f"Visualizing tracking for: {session_path}")
    print(f"{'='*70}")
    print(f"Found {len(pairs)} camera(s) to process\n")

    # Process each pair
    success_count = 0
    for i, (video_path, analysis_path) in enumerate(pairs):
        if i > 0:
            print(f"\n{'='*70}\n")

        print(f"[{i+1}/{len(pairs)}] Processing: {video_path.parent.name}")

        if visualize_tracking(video_path, analysis_path):
            success_count += 1
        else:
            print(f"  Failed to process {video_path}")

    print(f"\n{'='*70}")
    print(f"✓ Batch visualization complete!")
    print(f"Successfully processed: {success_count}/{len(pairs)} camera(s)")
    print(f"{'='*70}")

    return success_count == len(pairs)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize tracking data on video for all takes in a session',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('session_path',
                       help='Path to session or take directory')

    args = parser.parse_args()

    success = process_session(args.session_path)

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
