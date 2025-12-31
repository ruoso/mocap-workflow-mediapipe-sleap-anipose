#!/usr/bin/env python3
import subprocess
import re
import math
import sys
import struct
import os
import numpy as np
from pathlib import Path

if len(sys.argv) < 3 or len(sys.argv) > 4:
    print("Usage: detect_clap.py <input_folder> <output_folder> [max_duration_seconds]")
    print("  max_duration_seconds: Optional, limits output video length (e.g., 10 for quick validation)")
    sys.exit(1)

input_folder = Path(sys.argv[1])
output_folder = Path(sys.argv[2])
max_duration = float(sys.argv[3]) if len(sys.argv) == 4 else None

if not input_folder.is_dir():
    print(f"Error: {input_folder} is not a directory")
    sys.exit(1)

# Create output folder if it doesn't exist
output_folder.mkdir(parents=True, exist_ok=True)

# Find all video files
video_extensions = {'.mp4', '.MP4', '.mov', '.MOV', '.avi', '.AVI', '.mkv', '.MKV'}
video_files = [f for f in input_folder.iterdir() if f.suffix in video_extensions]

if not video_files:
    print(f"No video files found in {input_folder}")
    sys.exit(1)

print(f"Found {len(video_files)} video file(s) to process\n")

scan_seconds = 15.0       # how far from start to scan
margin_db = 12.0          # onset = peak - margin
clap_window = 0.4         # Max duration to group peaks as same clap (seconds)

# Storage for all video clap data
video_data = []

# Pass 1: Extract clap region from all videos
for infile in video_files:
    print(f"{'='*60}")
    print(f"Processing: {infile.name}")
    print(f"{'='*60}")

    # ------------------------------------------------------------
    # Pass 1: ebur128 with high-pass filter → find all significant peaks
    # High-pass filter removes low-frequency noise (rumble, etc.)
    # Claps are primarily high-frequency transients (1kHz+)
    # ------------------------------------------------------------
    cmd_ebur = [
        "ffmpeg",
        "-hide_banner",
        "-t", str(scan_seconds),
        "-i", str(infile),
        "-vn",
        "-af", "highpass=f=1000,ebur128=peak=true",  # 1kHz high-pass filter
        "-f", "null", "-"
    ]

    p = subprocess.run(cmd_ebur, stderr=subprocess.PIPE, text=True)

    # Find all peaks above a threshold
    peak_threshold = -10.0  # dBFS
    peaks = []

    for line in p.stderr.splitlines():
        m_time = re.search(r"t:\s*([0-9.]+)", line)
        m_tpk  = re.search(r"TPK:\s*([-0-9.]+)\s+([-0-9.]+)", line)
        if not (m_time and m_tpk):
            continue

        t = float(m_time.group(1))
        peak = max(float(m_tpk.group(1)), float(m_tpk.group(2)))

        if peak > peak_threshold:
            peaks.append((t, peak))

    if not peaks:
        print(f"  ERROR: Could not detect any peaks in {infile.name}")
        continue

    # Group peaks that are close together (within 0.4s = same clap)
    # If there's a gap with silence, treat as separate claps
    grouped_peaks = []
    current_group = [peaks[0]]

    for i in range(1, len(peaks)):
        time_gap = peaks[i][0] - current_group[-1][0]

        # If within 0.4s, it's the same clap
        if time_gap < 0.4:
            current_group.append(peaks[i])
        else:
            # Find the loudest in this group
            best_in_group = max(current_group, key=lambda x: x[1])
            grouped_peaks.append(best_in_group)
            current_group = [peaks[i]]

    # Don't forget the last group
    if current_group:
        best_in_group = max(current_group, key=lambda x: x[1])
        grouped_peaks.append(best_in_group)

    print(f"  Found {len(grouped_peaks)} potential clap(s):")
    for i, (t, peak) in enumerate(grouped_peaks):
        print(f"    Candidate {i+1}: t={t:.3f}s, TPK={peak:.2f} dBFS")

    # Use the loudest peak as the clap
    best_time, best_peak = max(grouped_peaks, key=lambda x: x[1])
    print(f"  Selected clap: t={best_time:.6f}s  TPK={best_peak:.2f} dBFS")

    # ------------------------------------------------------------
    # Pass 2: derive linear threshold
    # ------------------------------------------------------------
    thr_db = best_peak - margin_db
    thr = math.pow(10.0, thr_db / 20.0)

    print(f"  Using onset threshold: {thr_db:.2f} dBFS  →  {thr:.6f}")


    # ------------------------------------------------------------
    # Pass 3: Sample-accurate detection using raw audio
    # ------------------------------------------------------------

    # Extract a wider window of raw audio around the detected peak
    # Need enough context for cross-correlation to work properly
    window_start = max(0, best_time - 1.0)  # Start 1s before detected peak
    window_duration = 2.0                    # 2 second window

    tmpfile = f"/tmp/clap_window_{infile.stem}.s16le"

    cmd_extract = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-ss", str(window_start),
        "-t", str(window_duration),
        "-i", str(infile),
        "-vn",
        "-af", "highpass=f=1000",  # Apply same high-pass filter
        "-ac", "2",  # Keep stereo
        "-ar", "48000",  # Resample to known rate
        "-f", "s16le",  # 16-bit signed PCM
        "-acodec", "pcm_s16le",
        tmpfile
    ]

    subprocess.run(cmd_extract, stderr=subprocess.PIPE, check=True)

    # Read raw samples and find first threshold crossing
    sample_rate = 48000
    channels = 2
    bytes_per_sample = 2

    with open(tmpfile, "rb") as f:
        data = f.read()

    # Convert to numpy array for easier processing (mix to mono)
    audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    # Mix stereo to mono
    audio_mono = (audio[0::2] + audio[1::2]) / 2.0

    # Store this video's data (don't pick clap yet - do it globally)
    video_data.append({
        'file': infile,
        'audio': audio_mono,
        'window_start': window_start,
        'best_time': best_time,
        'sample_rate': sample_rate
    })

    print(f"  Extracted {len(audio_mono)} samples\n")

# Pass 2: Find the loudest peak in each video (simple approach)
print(f"{'='*60}")
print(f"Finding loudest peak in each video (high-pass filtered)...")
print(f"{'='*60}\n")

# Simple approach: find the loudest peak in the extracted window
# The high-pass filter should have removed low-frequency noise already
for i, data in enumerate(video_data):
    audio = data['audio']

    # Find the loudest peak - this is where we want videos to start
    max_peak_sample = np.argmax(np.abs(audio))
    max_peak_value = np.abs(audio[max_peak_sample])

    clap_time = data['window_start'] + (max_peak_sample / sample_rate)

    data['clap_sample_in_window'] = max_peak_sample
    data['clap_time'] = clap_time

    print(f"  {data['file'].name}:")
    print(f"    Peak at: {clap_time:.6f}s (sample {max_peak_sample} in window)")
    print(f"    Peak amplitude: {max_peak_value:.4f}")

print(f"\n{'='*60}")
print(f"Trimming videos to synchronized clap start...")
print(f"{'='*60}\n")

# Pass 3: Trim all videos at their detected clap position
for i, data in enumerate(video_data):
    infile = data['file']

    trim_time = data['clap_time']

    print(f"Processing: {infile.name}")
    print(f"  Trim at: {trim_time:.6f}s (clap start)")

    outfile = output_folder / infile.name

    # Use accurate seeking by putting -ss after -i and re-encoding
    # Standardize format for easier post-processing
    cmd_trim = [
        "ffmpeg",
        "-y",
        "-i", str(infile),
        "-ss", f"{trim_time:.6f}",
    ]

    # Add duration limit if specified
    if max_duration is not None:
        cmd_trim.extend(["-t", str(max_duration)])

    cmd_trim.extend([
        "-c:v", "libx264",  # H.264 video
        "-preset", "medium",
        "-crf", "23",
        "-r", "30",  # Standardize to 30fps
        "-c:a", "aac",  # AAC audio
        "-ar", "48000",  # 48kHz sample rate
        "-ac", "2",  # Stereo
        "-b:a", "192k",
        "-pix_fmt", "yuv420p",  # Standard pixel format
        "-reset_timestamps", "1",
        "-avoid_negative_ts", "make_zero",
        str(outfile)
    ])

    subprocess.run(cmd_trim, stderr=subprocess.PIPE, check=True)
    print(f"  ✓ Wrote {outfile.name}\n")

print(f"{'='*60}")
print(f"Processing complete! Output files in: {output_folder}")
print(f"{'='*60}")

