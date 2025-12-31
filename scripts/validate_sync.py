#!/usr/bin/env python3
import subprocess
import re
import sys
from pathlib import Path

if len(sys.argv) != 2:
    print("Usage: validate_sync.py <synced_folder>")
    sys.exit(1)

synced_folder = Path(sys.argv[1])

if not synced_folder.is_dir():
    print(f"Error: {synced_folder} is not a directory")
    sys.exit(1)

# Find all video files
video_extensions = {'.mp4', '.MP4', '.mov', '.MOV', '.avi', '.AVI', '.mkv', '.MKV'}
video_files = [f for f in synced_folder.iterdir() if f.suffix in video_extensions]

if not video_files:
    print(f"No video files found in {synced_folder}")
    sys.exit(1)

print(f"Found {len(video_files)} video file(s) to validate\n")

scan_seconds = 5.0  # Check first 5 seconds

results = []

for videofile in video_files:
    print(f"{'='*60}")
    print(f"Validating: {videofile.name}")
    print(f"{'='*60}")

    # Use ebur128 to find the first significant peak in the synced video
    cmd_ebur = [
        "ffmpeg",
        "-hide_banner",
        "-t", str(scan_seconds),
        "-i", str(videofile),
        "-vn",
        "-af", "ebur128=peak=true",
        "-f", "null", "-"
    ]

    p = subprocess.run(cmd_ebur, stderr=subprocess.PIPE, text=True)

    first_peak_time = None
    first_peak_db = -999.0
    threshold_db = -20.0  # Look for peaks above -20 dB

    for line in p.stderr.splitlines():
        m_time = re.search(r"t:\s*([0-9.]+)", line)
        m_tpk  = re.search(r"TPK:\s*([-0-9.]+)\s+([-0-9.]+)", line)
        if not (m_time and m_tpk):
            continue

        t = float(m_time.group(1))
        peak = max(float(m_tpk.group(1)), float(m_tpk.group(2)))

        # Find first peak above threshold
        if peak > threshold_db and first_peak_time is None:
            first_peak_time = t
            first_peak_db = peak
            break

    if first_peak_time is None:
        print(f"  ⚠ WARNING: No significant audio peak detected in first {scan_seconds}s")
        results.append((videofile.name, None, "No peak"))
    else:
        print(f"  First peak at: {first_peak_time:.6f}s ({first_peak_db:.2f} dBFS)")

        # Check if it's close to the beginning (well synchronized)
        if first_peak_time < 0.1:
            print(f"  ✓ EXCELLENT: Peak within 100ms of start")
            results.append((videofile.name, first_peak_time, "Excellent"))
        elif first_peak_time < 0.5:
            print(f"  ✓ GOOD: Peak within 500ms of start")
            results.append((videofile.name, first_peak_time, "Good"))
        elif first_peak_time < 1.0:
            print(f"  ⚠ OK: Peak within 1s of start")
            results.append((videofile.name, first_peak_time, "OK"))
        else:
            print(f"  ✗ POOR: Peak is {first_peak_time:.3f}s from start (may not be synchronized)")
            results.append((videofile.name, first_peak_time, "Poor"))

    print()

# Summary
print(f"{'='*60}")
print(f"VALIDATION SUMMARY")
print(f"{'='*60}")
print(f"{'File':<40} {'Peak Time':<12} {'Status'}")
print(f"{'-'*60}")
for name, time, status in results:
    time_str = f"{time:.6f}s" if time is not None else "N/A"
    print(f"{name:<40} {time_str:<12} {status}")
print(f"{'='*60}")

# Overall assessment
excellent = sum(1 for _, _, s in results if s == "Excellent")
good = sum(1 for _, _, s in results if s == "Good")
ok = sum(1 for _, _, s in results if s == "OK")
poor = sum(1 for _, _, s in results if s == "Poor")
no_peak = sum(1 for _, _, s in results if s == "No peak")

print(f"\nExcellent: {excellent}, Good: {good}, OK: {ok}, Poor: {poor}, No peak: {no_peak}")

if excellent + good == len(results):
    print("✓ All videos appear well synchronized!")
elif excellent + good + ok == len(results):
    print("✓ All videos appear reasonably synchronized")
else:
    print("⚠ Some videos may need attention")
