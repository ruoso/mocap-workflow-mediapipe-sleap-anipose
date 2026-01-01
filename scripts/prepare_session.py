#!/usr/bin/env python3
"""
Prepare session with multiple takes from synced videos.

Creates a session directory with:
- Calibration videos in session/calibration/ subdirectory
- Multiple take subdirectories with symlinked calibration
- Split videos for each take based on CSV timecodes

Directory structure:
  session/SESSION_NAME/
    calibration/
      camera1/calibration_images/video.mp4
      camera2/calibration_images/video.mp4
      calibration.toml (created by calibrate.sh)
      calibration_metadata.h5 (created by calibrate.sh)
    take1/
      calibration.toml -> ../calibration/calibration.toml
      calibration_metadata.h5 -> ../calibration/calibration_metadata.h5
      camera1/camera1-view-recording.mp4
      camera2/camera2-view-recording.mp4

CSV Format (takes.csv):
  take_name,start_time,end_time
  take1,20.5,45.3
  take2,50.0,75.8

Usage:
  prepare_session.py <session_name> <synced_folder> <takes_csv> \
    --calibration <start> <end> \
    --cameras <camera1> <camera2> ...

Example:
  prepare_session.py 20251231 synced/ takes.csv \
    --calibration 5.0 15.0 \
    --cameras camera1 camera2

Time format: seconds (float) from the start of synced videos
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare session with multiple takes from synced videos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('session_name',
                        help='Name of the session (e.g., 20251231)')
    parser.add_argument('synced_folder',
                        help='Path to synced videos folder')
    parser.add_argument('takes_csv',
                        help='Path to CSV file with take definitions')
    parser.add_argument('--calibration',
                        nargs=2,
                        type=float,
                        required=True,
                        metavar=('START', 'END'),
                        help='Calibration start and end time in seconds')
    parser.add_argument('--cameras',
                        nargs='+',
                        required=True,
                        help='Camera names in order (e.g., camera1 camera2)')
    parser.add_argument('--output-dir',
                        default='session',
                        help='Output directory for sessions (default: session)')

    return parser.parse_args()


def read_takes_csv(csv_path):
    """
    Read takes from CSV file.

    Returns: List of dicts with keys: take_name, start_time, end_time
    """
    takes = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)

        # Validate headers
        required_headers = {'take_name', 'start_time', 'end_time'}
        if not required_headers.issubset(reader.fieldnames):
            raise ValueError(f"CSV must have columns: {required_headers}")

        for row in reader:
            take_name = row['take_name'].strip()
            start_time = float(row['start_time'])
            end_time = float(row['end_time'])

            if start_time >= end_time:
                raise ValueError(f"Take '{take_name}': start ({start_time}) must be before end ({end_time})")

            takes.append({
                'take_name': take_name,
                'start_time': start_time,
                'end_time': end_time
            })

    return takes


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
    session_dir = output_dir / session_name
    takes_csv = Path(args.takes_csv)

    # Validate inputs
    if not synced_folder.is_dir():
        print(f"Error: {synced_folder} is not a directory")
        sys.exit(1)

    if not takes_csv.is_file():
        print(f"Error: CSV file not found: {takes_csv}")
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

    # Validate calibration times
    cal_start, cal_end = args.calibration
    if cal_start >= cal_end:
        print(f"Error: Calibration start ({cal_start}) must be before end ({cal_end})")
        sys.exit(1)

    # Read takes from CSV
    try:
        takes = read_takes_csv(takes_csv)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    if not takes:
        print(f"Error: No takes found in CSV")
        sys.exit(1)

    # Display summary
    print(f"{'='*70}")
    print(f"Preparing session: {session_name}")
    print(f"{'='*70}")
    print(f"Source: {synced_folder}")
    print(f"Calibration: {cal_start:.2f}s - {cal_end:.2f}s (duration: {cal_end-cal_start:.2f}s)")
    print(f"Takes: {len(takes)}")
    for take in takes:
        duration = take['end_time'] - take['start_time']
        print(f"  - {take['take_name']}: {take['start_time']:.2f}s - {take['end_time']:.2f}s (duration: {duration:.2f}s)")
    print(f"\nProcessing {len(video_files)} video(s):\n")

    # Step 1: Extract calibration videos to calibration subdirectory
    print(f"{'='*70}")
    print(f"STEP 1: Extracting calibration videos")
    print(f"{'='*70}\n")

    calibration_root = session_dir / "calibration"

    for video_file, camera_name in zip(video_files, args.cameras):
        print(f"{camera_name}: {video_file.name}")

        # Create camera directory in calibration subdirectory
        camera_dir = calibration_root / camera_name
        calibration_dir = camera_dir / "calibration_images"
        camera_dir.mkdir(parents=True, exist_ok=True)
        calibration_dir.mkdir(parents=True, exist_ok=True)

        # Extract calibration video
        calibration_output = calibration_dir / f"{session_name}-{camera_name}-calibration.mp4"

        print(f"  Extracting calibration ({cal_start:.2f}s - {cal_end:.2f}s)...", end='', flush=True)
        try:
            extract_segment(video_file, calibration_output, cal_start, cal_end)
            print(f" ✓")
        except subprocess.CalledProcessError as e:
            print(f" ✗ Failed: {e}")
            sys.exit(1)

    # Step 2: Create take directories with symlinked calibration and split videos
    print(f"\n{'='*70}")
    print(f"STEP 2: Creating takes")
    print(f"{'='*70}\n")

    for take in takes:
        take_name = take['take_name']
        start_time = take['start_time']
        end_time = take['end_time']
        duration = end_time - start_time

        take_dir = session_dir / take_name
        take_dir.mkdir(parents=True, exist_ok=True)

        print(f"Take: {take_name} ({start_time:.2f}s - {end_time:.2f}s, duration: {duration:.2f}s)")

        # Create symlinks to calibration files (will work after calibration is run)
        calib_files = ['calibration.toml', 'calibration_metadata.h5']
        for calib_file in calib_files:
            src_path = Path('..') / 'calibration' / calib_file
            dst_path = take_dir / calib_file

            # Remove existing symlink if present
            if dst_path.is_symlink() or dst_path.exists():
                dst_path.unlink()

            # Create relative symlink
            dst_path.symlink_to(src_path)

        print(f"  ✓ Symlinked calibration files")

        # Extract video for each camera
        for video_file, camera_name in zip(video_files, args.cameras):
            take_camera_dir = take_dir / camera_name
            take_camera_dir.mkdir(parents=True, exist_ok=True)

            output_video = take_camera_dir / f"{camera_name}-view-recording.mp4"

            print(f"  {camera_name}...", end='', flush=True)
            try:
                extract_segment(video_file, output_video, start_time, end_time)
                print(f" ✓")
            except subprocess.CalledProcessError as e:
                print(f" ✗ Failed: {e}")
                sys.exit(1)

        print()

    print(f"{'='*70}")
    print(f"✓ Session prepared successfully!")
    print(f"{'='*70}")
    print(f"Output: {session_dir}")
    print(f"\nNext steps:")
    print(f"  1. Run calibration: ./scripts/calibrate.sh {session_name}/calibration")
    print(f"  2. For each take:")
    print(f"     - Track: ./scripts/track_mediapipe.py {session_name}/<take_name>")
    print(f"     - Triangulate: ./scripts/triangulate.sh {session_name}/<take_name>")
    print(f"     - Visualize: ./scripts/visualize_3d.py {session_name}/<take_name>")
    print()


if __name__ == '__main__':
    main()
