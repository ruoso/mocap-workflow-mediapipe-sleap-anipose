#!/usr/bin/env python3
"""
Prepare session from synced videos by extracting calibration and capture segments
and organizing them into the proper directory structure.

Usage:
  prepare_session.py <session_name> <synced_folder> \
    --calibration <start> <end> \
    --capture <start> <end> \
    --cameras <camera1> <camera2> ...

Example:
  prepare_session.py 20251230v3 synced/ \
    --calibration 5.0 15.0 \
    --capture 20.0 60.0 \
    --cameras camera1 camera2 camera3 camera4

Time format: seconds (float) from the start of synced videos
"""
import subprocess
import sys
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare session from synced videos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('session_name',
                        help='Name of the session (e.g., 20251230v3)')
    parser.add_argument('synced_folder',
                        help='Path to synced videos folder')
    parser.add_argument('--calibration',
                        nargs=2,
                        type=float,
                        required=True,
                        metavar=('START', 'END'),
                        help='Calibration start and end time in seconds')
    parser.add_argument('--capture',
                        nargs=2,
                        type=float,
                        required=True,
                        metavar=('START', 'END'),
                        help='Capture start and end time in seconds')
    parser.add_argument('--cameras',
                        nargs='+',
                        required=True,
                        help='Camera names in order (e.g., camera1 camera2 camera3)')
    parser.add_argument('--output-dir',
                        default='session',
                        help='Output directory for sessions (default: session)')

    return parser.parse_args()


def extract_segment(input_file, output_file, start_time, end_time):
    """Extract a time segment from a video using ffmpeg"""
    duration = end_time - start_time

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(input_file),
        "-ss", f"{start_time:.6f}",
        "-t", f"{duration:.6f}",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "23",
        "-c:a", "aac",
        "-ar", "48000",
        "-ac", "2",
        "-b:a", "192k",
        "-pix_fmt", "yuv420p",
        str(output_file)
    ]

    subprocess.run(cmd, check=True)


def main():
    args = parse_args()

    synced_folder = Path(args.synced_folder)
    output_dir = Path(args.output_dir)
    session_name = args.session_name

    if not synced_folder.is_dir():
        print(f"Error: {synced_folder} is not a directory")
        sys.exit(1)

    # Find all video files in synced folder
    video_extensions = {'.mp4', '.MP4', '.mov', '.MOV', '.avi', '.AVI', '.mkv', '.MKV'}
    video_files = sorted([f for f in synced_folder.iterdir() if f.suffix in video_extensions])

    if not video_files:
        print(f"Error: No video files found in {synced_folder}")
        sys.exit(1)

    # Validate camera count matches video count
    if len(video_files) != len(args.cameras):
        print(f"Error: Found {len(video_files)} videos but {len(args.cameras)} camera names")
        print(f"Videos: {[v.name for v in video_files]}")
        print(f"Cameras: {args.cameras}")
        sys.exit(1)

    # Validate time intervals
    cal_start, cal_end = args.calibration
    cap_start, cap_end = args.capture

    if cal_start >= cal_end:
        print(f"Error: Calibration start ({cal_start}) must be before end ({cal_end})")
        sys.exit(1)

    if cap_start >= cap_end:
        print(f"Error: Capture start ({cap_start}) must be before end ({cap_end})")
        sys.exit(1)

    print(f"{'='*70}")
    print(f"Preparing session: {session_name}")
    print(f"{'='*70}")
    print(f"Source: {synced_folder}")
    print(f"Calibration: {cal_start:.2f}s - {cal_end:.2f}s (duration: {cal_end-cal_start:.2f}s)")
    print(f"Capture: {cap_start:.2f}s - {cap_end:.2f}s (duration: {cap_end-cap_start:.2f}s)")
    print(f"\nProcessing {len(video_files)} video(s):\n")

    # Process each video
    for video_file, camera_name in zip(video_files, args.cameras):
        print(f"{'='*70}")
        print(f"{video_file.name} -> {camera_name}")
        print(f"{'='*70}")

        # Create directory structure
        camera_dir = output_dir / session_name / camera_name
        calibration_dir = camera_dir / "calibration_images"

        camera_dir.mkdir(parents=True, exist_ok=True)
        calibration_dir.mkdir(parents=True, exist_ok=True)

        # Define output paths
        calibration_output = calibration_dir / f"{session_name}-{camera_name}-calibration.mp4"
        capture_output = camera_dir / f"{camera_name}-view-recording.mp4"

        # Extract calibration segment
        print(f"  Extracting calibration ({cal_start:.2f}s - {cal_end:.2f}s)...")
        try:
            extract_segment(video_file, calibration_output, cal_start, cal_end)
            print(f"    ✓ {calibration_output}")
        except subprocess.CalledProcessError as e:
            print(f"    ✗ Failed to extract calibration: {e}")
            continue

        # Extract capture segment
        print(f"  Extracting capture ({cap_start:.2f}s - {cap_end:.2f}s)...")
        try:
            extract_segment(video_file, capture_output, cap_start, cap_end)
            print(f"    ✓ {capture_output}")
        except subprocess.CalledProcessError as e:
            print(f"    ✗ Failed to extract capture: {e}")
            continue

        print()

    print(f"{'='*70}")
    print(f"✓ Session prepared successfully!")
    print(f"{'='*70}")
    print(f"Output: {output_dir / session_name}")
    print()


if __name__ == '__main__':
    main()
