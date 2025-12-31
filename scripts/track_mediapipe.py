#!/usr/bin/env python3
"""
Track human poses using MediaPipe Pose Landmarker and convert to SLEAP format.
Uses the heavy model for accurate 33-landmark detection.
"""

import argparse
import h5py
import numpy as np
from pathlib import Path
import cv2
import sys
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# MediaPipe Pose has 33 landmarks
MEDIAPIPE_LANDMARKS = [
    "nose",                    # 0
    "left_eye_inner",          # 1
    "left_eye",                # 2
    "left_eye_outer",          # 3
    "right_eye_inner",         # 4
    "right_eye",               # 5
    "right_eye_outer",         # 6
    "left_ear",                # 7
    "right_ear",               # 8
    "mouth_left",              # 9
    "mouth_right",             # 10
    "left_shoulder",           # 11
    "right_shoulder",          # 12
    "left_elbow",              # 13
    "right_elbow",             # 14
    "left_wrist",              # 15
    "right_wrist",             # 16
    "left_pinky",              # 17
    "right_pinky",             # 18
    "left_index",              # 19
    "right_index",             # 20
    "left_thumb",              # 21
    "right_thumb",             # 22
    "left_hip",                # 23
    "right_hip",               # 24
    "left_knee",               # 25
    "right_knee",              # 26
    "left_ankle",              # 27
    "right_ankle",             # 28
    "left_heel",               # 29
    "right_heel",              # 30
    "left_foot_index",         # 31
    "right_foot_index",        # 32
]


def process_video_with_mediapipe(video_path, model_path):
    """
    Process video with MediaPipe Pose Landmarker.

    Args:
        video_path: Path to input video file
        model_path: Path to pose_landmarker_heavy.task model

    Returns:
        List of detections per frame: [(landmarks, visibility_scores), ...]
    """
    print(f"  Opening video: {video_path}")

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Error: Could not open video {video_path}")
        return None

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"  Video: {total_frames} frames, {fps:.2f} fps, {width}x{height}")

    # Create MediaPipe Pose Landmarker
    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,  # Track single person
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    detector = vision.PoseLandmarker.create_from_options(options)

    # Process frames
    frame_data = []
    valid_frames = []
    frame_idx = 0

    print(f"  Processing frames...", end='', flush=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Calculate timestamp in milliseconds
        timestamp_ms = int(frame_idx * 1000 / fps)

        # Detect pose
        detection_result = detector.detect_for_video(mp_image, timestamp_ms)

        # Extract landmarks if person detected
        if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
            landmarks = detection_result.pose_landmarks[0]  # First person

            # Extract x, y coordinates (normalized 0-1, convert to pixels)
            points = np.array([
                [lm.x * width, lm.y * height] for lm in landmarks
            ])

            # Extract visibility scores
            scores = np.array([lm.visibility for lm in landmarks])

            frame_data.append((points, scores))
            valid_frames.append(frame_idx)

        frame_idx += 1

        # Progress indicator
        if frame_idx % 100 == 0:
            print(f"\r  Processing frames... {frame_idx}/{total_frames}", end='', flush=True)

    print(f"\r  Processing frames... {frame_idx}/{total_frames} - Done!")

    cap.release()
    detector.close()

    print(f"  Detected person in {len(frame_data)}/{frame_idx} frames")

    return frame_data, valid_frames, fps


def convert_to_sleap_format(frame_data, valid_frames, output_path, fps=30.0):
    """
    Convert MediaPipe detections to SLEAP analysis.h5 format.

    Args:
        frame_data: List of (landmarks, scores) tuples per frame
        valid_frames: List of frame indices with valid detections
        output_path: Output HDF5 file path
        fps: Video frame rate
    """
    if not frame_data:
        print(f"  Error: No valid detections to save")
        return False

    print(f"  Converting to SLEAP format...")

    # Stack all data
    all_points = np.array([data[0] for data in frame_data])  # [n_frames, n_landmarks, 2]
    all_scores = np.array([data[1] for data in frame_data])  # [n_frames, n_landmarks]

    n_frames, n_nodes, _ = all_points.shape

    # Create SLEAP-compatible HDF5 file
    with h5py.File(output_path, 'w') as f:
        # Track information (single track for single person)
        f.create_dataset('track_names', data=[b'track_0'])

        # Tracks dataset: shape (n_tracks, 2, n_nodes, n_frames)
        # all_points is (n_frames, n_nodes, 2), need to reshape to (1, 2, n_nodes, n_frames)
        tracks_data = all_points.transpose(2, 1, 0)  # (2, n_nodes, n_frames)
        tracks_data = tracks_data[np.newaxis, :, :, :]  # (1, 2, n_nodes, n_frames)
        f.create_dataset('tracks', data=tracks_data.astype('<f8'))

        # Node (landmark) names
        node_names = [name.encode('utf-8') for name in MEDIAPIPE_LANDMARKS]
        f.create_dataset('node_names', data=node_names)

        # Frame indices
        f.create_dataset('frames', data=np.array(valid_frames, dtype='<f8'))

        # Point locations [n_frames, n_instances, n_nodes, 2]
        points_reshaped = all_points[:, np.newaxis, :, :]
        f.create_dataset('instance_points', data=points_reshaped.astype('<f8'))

        # Point scores [n_frames, n_instances, n_nodes]
        scores_reshaped = all_scores[:, np.newaxis, :]
        f.create_dataset('point_scores', data=scores_reshaped.astype('<f8'))

        # Instance scores [n_frames, n_instances]
        instance_scores = np.mean(all_scores, axis=1, keepdims=True)
        f.create_dataset('instance_scores', data=instance_scores.astype('<f8'))

        # Metadata
        f.attrs['fps'] = fps
        f.attrs['model'] = 'mediapipe_pose_landmarker_heavy'

    print(f"  Saved: {output_path}")
    print(f"    Frames: {n_frames}")
    print(f"    Landmarks: {n_nodes}")

    return True


def process_camera(camera_dir, model_path):
    """Process a single camera directory."""
    camera_name = camera_dir.name

    # Find video file
    video_files = list(camera_dir.glob("*-recording.mp4"))
    if not video_files:
        video_files = list(camera_dir.glob("*.mp4"))

    if not video_files:
        print(f"  Warning: No video file found in {camera_dir}")
        return False

    video_path = video_files[0]
    output_path = camera_dir / f"{camera_name}_analysis.h5"

    print(f"\nProcessing {camera_name}:")

    # Process video with MediaPipe
    result = process_video_with_mediapipe(video_path, model_path)
    if result is None:
        return False

    frame_data, valid_frames, fps = result

    if not frame_data:
        print(f"  Warning: No valid detections found")
        return False

    # Convert to SLEAP format
    return convert_to_sleap_format(frame_data, valid_frames, output_path, fps)


def main():
    parser = argparse.ArgumentParser(
        description="Track poses using MediaPipe and convert to SLEAP format"
    )
    parser.add_argument(
        "session",
        help="Session ID (e.g., 20251231)"
    )
    parser.add_argument(
        "--model",
        default="pose_landmarker_heavy.task",
        help="Path to MediaPipe pose landmarker model (default: pose_landmarker_heavy.task)"
    )

    args = parser.parse_args()

    # Check model file
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        print("\nPlease download the model:")
        print("  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task")
        return 1

    # Check session directory
    session_dir = Path(f"session/{args.session}")
    if not session_dir.exists():
        print(f"Error: Session directory {session_dir} does not exist")
        return 1

    # Find camera directories
    camera_dirs = sorted(session_dir.glob("camera*"))
    if not camera_dirs:
        print(f"Error: No camera directories found in {session_dir}")
        return 1

    print(f"MediaPipe Pose Tracking")
    print(f"=" * 60)
    print(f"Session: {args.session}")
    print(f"Model: {model_path}")
    print(f"Cameras: {len(camera_dirs)}")

    # Process each camera
    success_count = 0
    for camera_dir in camera_dirs:
        if process_camera(camera_dir, model_path):
            success_count += 1

    print(f"\n{'=' * 60}")
    print(f"Tracking complete! {success_count}/{len(camera_dirs)} camera(s) processed.")

    if success_count == len(camera_dirs):
        print(f"\nNext step: Run triangulation")
        print(f"  ./scripts/triangulate.sh {args.session}")
        return 0
    else:
        print(f"\nWarning: Some cameras failed to process")
        return 1


if __name__ == "__main__":
    sys.exit(main())
