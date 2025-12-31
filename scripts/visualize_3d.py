#!/usr/bin/env python3
"""
Visualize 3D pose data from triangulation as an animated video.
Creates a rotating 3D skeleton visualization.
"""

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import sys

# MediaPipe skeleton connections (pairs of landmark indices)
MEDIAPIPE_CONNECTIONS = [
    # Face
    (0, 1), (1, 2), (2, 3),  # Right eye
    (0, 4), (4, 5), (5, 6),  # Left eye
    (0, 9), (0, 10),  # Mouth
    (2, 7), (5, 8),  # Ears

    # Torso
    (11, 12),  # Shoulders
    (11, 23), (12, 24),  # Shoulder to hip
    (23, 24),  # Hips

    # Right arm
    (11, 13), (13, 15),  # Shoulder -> Elbow -> Wrist
    (15, 17), (15, 19), (15, 21),  # Wrist -> fingers

    # Left arm
    (12, 14), (14, 16),  # Shoulder -> Elbow -> Wrist
    (16, 18), (16, 20), (16, 22),  # Wrist -> fingers

    # Right leg
    (23, 25), (25, 27),  # Hip -> Knee -> Ankle
    (27, 29), (27, 31),  # Ankle -> Heel, Toe
    (29, 31),  # Heel to toe

    # Left leg
    (24, 26), (26, 28),  # Hip -> Knee -> Ankle
    (28, 30), (28, 32),  # Ankle -> Heel, Toe
    (30, 32),  # Heel to toe
]


def load_3d_points(h5_file):
    """Load 3D points from triangulation output."""
    with h5py.File(h5_file, 'r') as f:
        # Check what datasets are available
        if 'tracks' in f:
            points = f['tracks'][:]
            print(f"Loaded 'tracks' dataset with shape: {points.shape}")
        elif 'points3d' in f:
            points = f['points3d'][:]
            print(f"Loaded 'points3d' dataset with shape: {points.shape}")
        else:
            print(f"Available datasets: {list(f.keys())}")
            raise ValueError("Could not find 3D points dataset in file")

        # Try to get node names
        node_names = None
        if 'node_names' in f:
            node_names = [name.decode('utf-8') if isinstance(name, bytes) else name
                         for name in f['node_names'][:]]

    return points, node_names


def setup_3d_plot(ax, points_3d):
    """Setup 3D plot with appropriate limits."""
    # Calculate data bounds
    all_points = points_3d.reshape(-1, 3)
    valid_points = all_points[~np.isnan(all_points).any(axis=1)]

    if len(valid_points) == 0:
        # Default bounds if no valid points
        bounds = [-1, 1]
    else:
        x_range = [valid_points[:, 0].min(), valid_points[:, 0].max()]
        y_range = [valid_points[:, 1].min(), valid_points[:, 1].max()]
        z_range = [valid_points[:, 2].min(), valid_points[:, 2].max()]

        # Make bounds equal for all axes
        max_range = max(
            x_range[1] - x_range[0],
            y_range[1] - y_range[0],
            z_range[1] - z_range[0]
        ) / 2.0

        mid_x = (x_range[0] + x_range[1]) / 2.0
        mid_y = (y_range[0] + y_range[1]) / 2.0
        mid_z = (z_range[0] + z_range[1]) / 2.0

        bounds = [
            [mid_x - max_range, mid_x + max_range],
            [mid_y - max_range, mid_y + max_range],
            [mid_z - max_range, mid_z + max_range]
        ]

    ax.set_xlim(bounds[0] if isinstance(bounds[0], list) else bounds)
    ax.set_ylim(bounds[2] if isinstance(bounds[2], list) else bounds)  # Data Z (depth) → Plot Y
    # Data Y is height but inverted, so flip it for Plot Z
    y_bounds = bounds[1] if isinstance(bounds[1], list) else bounds
    if isinstance(y_bounds, list):
        ax.set_zlim([-y_bounds[1], -y_bounds[0]])  # Data -Y (height) → Plot Z
    else:
        ax.set_zlim([-y_bounds, y_bounds])

    ax.set_xlabel('X (Left-Right)')
    ax.set_ylabel('Z (Depth)')
    ax.set_zlabel('Y (Height)')
    ax.set_box_aspect([1, 1, 1])


def render_3d_video(points_3d, output_file, fps=30, rotate=True):
    """
    Render 3D pose as video.

    Args:
        points_3d: Array of 3D points with shape (n_frames, n_tracks, n_nodes, 3)
                   or (n_frames, n_nodes, 3)
        output_file: Output video file path
        fps: Frame rate for output video
        rotate: Whether to rotate the view during animation
    """
    # Handle different input shapes
    if points_3d.ndim == 4:
        # (n_frames, n_tracks, n_nodes, 3) - take first track
        points_3d = points_3d[:, 0, :, :]
    elif points_3d.ndim == 3:
        # (n_frames, n_nodes, 3) - already good
        pass
    else:
        raise ValueError(f"Unexpected points shape: {points_3d.shape}")

    n_frames, n_nodes, _ = points_3d.shape
    print(f"Rendering {n_frames} frames with {n_nodes} landmarks...")

    # Create figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Setup plot limits
    setup_3d_plot(ax, points_3d)

    # Initialize plot elements
    scatter = ax.scatter([], [], [], c='red', marker='o', s=50)
    lines = [ax.plot([], [], [], 'b-', linewidth=2)[0] for _ in MEDIAPIPE_CONNECTIONS]
    title = ax.set_title('Frame 0')

    def update(frame):
        """Update function for animation."""
        # Get current frame points
        pts = points_3d[frame]

        # Update scatter plot (keypoints)
        # Swap Y and Z so Z is vertical (height)
        valid_mask = ~np.isnan(pts).any(axis=1)
        valid_pts = pts[valid_mask]

        if len(valid_pts) > 0:
            # Plot as (X, Z, -Y): X=left-right, Y=depth, Z=height (negated)
            scatter._offsets3d = (valid_pts[:, 0], valid_pts[:, 2], -valid_pts[:, 1])
        else:
            scatter._offsets3d = ([], [], [])

        # Update skeleton lines
        for line, (start_idx, end_idx) in zip(lines, MEDIAPIPE_CONNECTIONS):
            if (start_idx < len(pts) and end_idx < len(pts) and
                not np.isnan(pts[start_idx]).any() and not np.isnan(pts[end_idx]).any()):
                # Plot as (X, Z, -Y): matplotlib expects Z as vertical
                xs = [pts[start_idx, 0], pts[end_idx, 0]]  # Data X → Plot X (left-right)
                ys = [pts[start_idx, 2], pts[end_idx, 2]]  # Data Z → Plot Y (depth)
                zs = [-pts[start_idx, 1], -pts[end_idx, 1]]  # Data -Y → Plot Z (height, negated)
                line.set_data(xs, ys)
                line.set_3d_properties(zs)
            else:
                line.set_data([], [])
                line.set_3d_properties([])

        # Rotate view around vertical axis
        if rotate:
            ax.view_init(elev=15, azim=frame * 360 / n_frames)

        title.set_text(f'Frame {frame}/{n_frames}')

        return [scatter] + lines + [title]

    # Create animation
    print("Creating animation...")
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000/fps, blit=False)

    # Save video
    print(f"Saving video to {output_file}...")
    writer = FFMpegWriter(fps=fps, bitrate=5000)
    anim.save(output_file, writer=writer)

    plt.close()
    print(f"Video saved successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize 3D pose data as animated video"
    )
    parser.add_argument(
        "session",
        help="Session ID (e.g., 20251231)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Output video frame rate (default: 30)"
    )
    parser.add_argument(
        "--no-rotate",
        action="store_true",
        help="Disable automatic rotation of view"
    )
    parser.add_argument(
        "--output",
        help="Output video file path (default: session/SESSION/visualization_3d.mp4)"
    )

    args = parser.parse_args()

    # Find points3d file
    session_dir = Path(f"session/{args.session}")
    points3d_file = session_dir / "points3d.h5"

    if not points3d_file.exists():
        print(f"Error: {points3d_file} not found")
        return 1

    # Default output path
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = session_dir / "visualization_3d.mp4"

    print(f"Loading 3D points from {points3d_file}...")
    points_3d, node_names = load_3d_points(points3d_file)

    if node_names:
        print(f"Landmarks: {len(node_names)}")

    # Render video
    render_3d_video(
        points_3d,
        output_file,
        fps=args.fps,
        rotate=not args.no_rotate
    )

    print(f"\nVisualization complete!")
    print(f"Output: {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
