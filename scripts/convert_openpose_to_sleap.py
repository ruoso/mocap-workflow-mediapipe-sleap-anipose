#!/usr/bin/env python3
"""
Convert OpenPose BODY_25 JSON output to SLEAP analysis.h5 format.
This allows using OpenPose for tracking with sleap-anipose triangulation.
"""

import argparse
import json
import h5py
import numpy as np
from pathlib import Path
import sys

# OpenPose BODY_25 keypoint names (25 keypoints)
BODY_25_KEYPOINTS = [
    "Nose",           # 0
    "Neck",           # 1
    "RShoulder",      # 2
    "RElbow",         # 3
    "RWrist",         # 4
    "LShoulder",      # 5
    "LElbow",         # 6
    "LWrist",         # 7
    "MidHip",         # 8
    "RHip",           # 9
    "RKnee",          # 10
    "RAnkle",         # 11
    "LHip",           # 12
    "LKnee",          # 13
    "LAnkle",         # 14
    "REye",           # 15
    "LEye",           # 16
    "REar",           # 17
    "LEar",           # 18
    "LBigToe",        # 19
    "LSmallToe",      # 20
    "LHeel",          # 21
    "RBigToe",        # 22
    "RSmallToe",      # 23
    "RHeel",          # 24
]


def load_openpose_json(json_file):
    """Load keypoints from OpenPose JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)

    people = data.get('people', [])
    if not people:
        return None

    # Take first person detected (you may want to modify this for multi-person)
    person = people[0]
    keypoints = np.array(person['pose_keypoints_2d']).reshape(-1, 3)  # [25, 3] (x, y, confidence)

    return keypoints


def convert_openpose_to_sleap(openpose_dir, output_file):
    """
    Convert OpenPose JSON outputs to SLEAP analysis.h5 format.

    Args:
        openpose_dir: Directory containing OpenPose JSON files
        output_file: Output SLEAP analysis.h5 file path
    """
    openpose_path = Path(openpose_dir)

    # Find all JSON files (OpenPose names them as: video_name_000000000000_keypoints.json)
    json_files = sorted(openpose_path.glob("*_keypoints.json"))

    if not json_files:
        print(f"Error: No OpenPose JSON files found in {openpose_dir}")
        return False

    print(f"Found {len(json_files)} JSON files")

    # Parse frame data
    frames_data = []
    valid_frames = []

    for idx, json_file in enumerate(json_files):
        keypoints = load_openpose_json(json_file)
        if keypoints is not None:
            frames_data.append(keypoints)
            valid_frames.append(idx)

    if not frames_data:
        print("Error: No valid keypoints found in JSON files")
        return False

    print(f"Loaded {len(frames_data)} frames with valid detections")

    # Convert to numpy arrays
    # Shape: [n_frames, n_keypoints, 3] where 3 is (x, y, confidence)
    all_keypoints = np.array(frames_data)
    n_frames, n_nodes, _ = all_keypoints.shape

    # Separate coordinates and scores
    points = all_keypoints[:, :, :2]  # [n_frames, n_nodes, 2] (x, y)
    point_scores = all_keypoints[:, :, 2]  # [n_frames, n_nodes] (confidence)

    # Create SLEAP-compatible HDF5 file
    print(f"Writing to {output_file}...")

    with h5py.File(output_file, 'w') as f:
        # Track information (single track for single person)
        f.create_dataset('track_names', data=[b'track_0'])
        f.create_dataset('tracks', data=np.zeros((n_frames, 1), dtype='<f8'))  # All instances belong to track 0

        # Node (keypoint) names
        node_names = [name.encode('utf-8') for name in BODY_25_KEYPOINTS]
        f.create_dataset('node_names', data=node_names)

        # Frame indices
        f.create_dataset('frames', data=np.array(valid_frames, dtype='<f8'))

        # Point locations [n_frames, n_instances, n_nodes, 2]
        # For single person tracking: n_instances = 1
        points_reshaped = points[:, np.newaxis, :, :]  # Add instance dimension
        f.create_dataset('instance_points', data=points_reshaped.astype('<f8'))

        # Point scores [n_frames, n_instances, n_nodes]
        scores_reshaped = point_scores[:, np.newaxis, :]  # Add instance dimension
        f.create_dataset('point_scores', data=scores_reshaped.astype('<f8'))

        # Instance scores [n_frames, n_instances]
        instance_scores = np.mean(point_scores, axis=1, keepdims=True)  # Average confidence per frame
        f.create_dataset('instance_scores', data=instance_scores.astype('<f8'))

        # Metadata
        f.attrs['video_path'] = str(openpose_path.parent)
        f.attrs['fps'] = 30.0  # Default, adjust if known

    print(f"Successfully created {output_file}")
    print(f"  Frames: {n_frames}")
    print(f"  Keypoints: {n_nodes}")
    print(f"  Format: SLEAP-compatible HDF5")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert OpenPose BODY_25 output to SLEAP analysis.h5 format"
    )
    parser.add_argument(
        "session",
        help="Session ID (e.g., 20251231)"
    )

    args = parser.parse_args()

    session_dir = Path(f"session/{args.session}")

    if not session_dir.exists():
        print(f"Error: Session directory {session_dir} does not exist")
        return 1

    # Find all camera directories
    camera_dirs = sorted(session_dir.glob("camera*"))

    if not camera_dirs:
        print(f"Error: No camera directories found in {session_dir}")
        return 1

    print(f"Found {len(camera_dirs)} camera(s)")
    print()

    success_count = 0

    # Process each camera
    for camera_dir in camera_dirs:
        camera_name = camera_dir.name
        openpose_dir = camera_dir / "openpose_output"
        output_file = camera_dir / f"{camera_name}_analysis.h5"

        print(f"Processing {camera_name}...")

        if not openpose_dir.exists():
            print(f"  Warning: {openpose_dir} does not exist, skipping...")
            continue

        if convert_openpose_to_sleap(openpose_dir, output_file):
            success_count += 1

        print()

    print(f"\nConversion complete! {success_count}/{len(camera_dirs)} camera(s) processed.")

    if success_count == len(camera_dirs):
        print(f"\nYou can now run triangulation with: ./scripts/triangulate.sh {args.session}")
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
