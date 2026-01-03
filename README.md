# Motion Capture Pipeline

A complete multi-camera motion capture pipeline for markerless 3D pose estimation. This system synchronizes videos from multiple cameras, performs 2D pose tracking using MediaPipe, and triangulates to produce accurate 3D coordinates.

## Features

- **Multi-camera synchronization** via audio clap detection
- **Flexible session management** with multiple takes per session
- **2D pose tracking** using MediaPipe Pose (33 landmarks)
- **Interactive manual correction** for tracking errors
- **Camera calibration** using checkerboard patterns
- **3D triangulation** from multiple camera views
- **Visualization tools** for 2D tracking and 3D poses

## Installation

### Prerequisites

- Python 3.8+
- ffmpeg (for video processing)
- A printer (A3 recommended, or A4/Letter for multi-page printing)

### Setup

1. Clone or download this repository

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the MediaPipe Pose Landmarker model:
```bash
# Download the heavy model for best accuracy
wget https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task
```

5. **Print the calibration board:**

   This repository includes pre-generated ChArUco board files ready for printing:
   - `board_A3.pdf` - For A3 paper (single page, recommended)
   - `board_A4_multipage.pdf` - For A4 paper (2 pages)
   - `board_Letter_multipage.pdf` - For US Letter paper (2 pages)

   The board configuration is already set in `board.toml`:
   - 7×5 ChArUco board
   - 40mm ArUco markers
   - 52mm squares
   - 5×5 bit markers (DICT_5X5_50)

   **Printing instructions:**
   - Print at **100% scale** (do not scale to fit page)
   - For multi-page: trim pages at crop marks and align edges precisely
   - Mount on rigid backing (foam board recommended)
   - Verify marker size after printing (should be exactly 40mm)

   **To regenerate the board (optional):**
   ```bash
   python3 scripts/generate_board.py --marker-size 40
   ```

## Complete Workflow

### 1. Video Capture

Record videos from multiple cameras simultaneously:
- Use synchronized recording if possible
- Include a clear clap at the beginning for synchronization
- Record calibration footage with checkerboard visible in all cameras
- Record your motion capture takes

### 2. Video Synchronization

Synchronize multiple camera videos using audio clap detection:

```bash
python3 scripts/detect_clap.py <input_folder> <output_folder>
```

**Example:**
```bash
python3 scripts/detect_clap.py raw_videos/ synced/
```

This will:
- Detect the first loud clap in each video
- Align all videos to start at the same point
- Output synchronized videos to the specified folder

**Validation:**
```bash
python3 scripts/validate_sync.py synced/
```

### 3. Session Preparation

Organize synchronized videos into a session structure with calibration and takes:

```bash
python3 scripts/prepare_session.py <session_name> <synced_folder> <takes_csv> \
    --calibration <start> <end> \
    --cameras <camera1> <camera2> ...
```

**Example:**
```bash
# Create takes.csv defining your takes:
# take_name,start_time,end_time
# take1,20.5,45.3
# take2,50.0,75.8

python3 scripts/prepare_session.py 20251231 synced/ takes.csv \
    --calibration 5.0 15.0 \
    --cameras camera1 camera2 camera3
```

**Output structure:**
```
session/20251231/
├── calibration/
│   ├── camera1/
│   │   └── calibration_images/
│   │       └── video.mp4
│   ├── camera2/
│   │   └── calibration_images/
│   │       └── video.mp4
│   └── camera3/
│       └── calibration_images/
│           └── video.mp4
├── take1/
│   ├── calibration.toml -> ../calibration/calibration.toml
│   ├── calibration_metadata.h5 -> ../calibration/calibration_metadata.h5
│   ├── camera1/
│   │   └── camera1-view-recording.mp4
│   ├── camera2/
│   │   └── camera2-view-recording.mp4
│   └── camera3/
│       └── camera3-view-recording.mp4
└── take2/
    └── ...
```

### 4. Camera Calibration

Calibrate the cameras using the printed ChArUco board:

```bash
./scripts/calibrate.sh <session_name>
```

**Example:**
```bash
./scripts/calibrate.sh 20251231
```

This uses the `board.toml` configuration file (included in the repository) and creates:
- `calibration.toml` - Camera parameters and extrinsics
- `calibration_metadata.h5` - Detailed calibration data
- `reprojection_histogram.png` - Quality visualization

**Tips:**
- Move the ChArUco board to various positions and orientations
- Ensure the board is visible in all cameras simultaneously
- The ChArUco board allows partial visibility (doesn't need to be fully in frame)
- 50-100 frames with good board visibility is usually sufficient
- The 40mm markers provide good detection at typical calibration distances

### 5. 2D Pose Tracking

Extract 2D pose landmarks from each camera using MediaPipe:

```bash
python3 scripts/track_mediapipe.py <session_dir>
```

**Example:**
```bash
python3 scripts/track_mediapipe.py session/20251231/take1
```

This processes all camera videos and creates `*_analysis.h5` files in each camera directory with 33 MediaPipe landmarks per frame.

**MediaPipe Landmarks:**
- Face: nose, eyes, ears, mouth (11 points)
- Arms: shoulders, elbows, wrists, fingers (12 points)
- Legs: hips, knees, ankles, heels, feet (10 points)

### 6. Manual Tracking Correction (Optional)

If tracking errors are detected, use the interactive editor to manually correct them:

```bash
python3 scripts/edit_tracking.py <camera_directory>
```

**Example:**
```bash
python3 scripts/edit_tracking.py session/20251231/take1/camera1
```

**Features:**
- Frame-by-frame navigation
- Drag markers to correct positions
- Delta tracking: corrections propagate forward automatically
- Symmetric pair flipping for left/right swaps
- Zoom and pan for precision editing
- Automatic backup before saving

**Controls:**
- `Left/Right Arrow` - Navigate frames
- `Space` - Play/pause
- `Mouse drag` - Move markers
- `Shift + drag` - Move marker with downstream chain
- `Right click` - Delete marker
- `Middle click` - Restore marker
- `S/E/W/H/K/A` - Select symmetric pairs (shoulders, elbows, wrists, hips, knees, ankles)
- `Enter` - Flip selected pair
- `R` - Reset frame to original
- `C` - Add delta barrier
- `+/-` - Zoom in/out
- `Ctrl+S` - Save and quit

### 7. Tracking Visualization (Optional)

Create videos with 2D tracking overlay for quality assessment:

```bash
python3 scripts/visualize_tracking.py <session_path>
```

**Example:**
```bash
# Visualize all cameras in all takes
python3 scripts/visualize_tracking.py session/20251231

# Visualize specific take
python3 scripts/visualize_tracking.py session/20251231/take1
```

Output: `*_tracked.mp4` files in each camera directory showing skeleton overlay.

### 8. 3D Triangulation

Convert 2D tracking data from multiple cameras into 3D coordinates:

```bash
./scripts/triangulate.sh <session_name>
```

**Example:**
```bash
./scripts/triangulate.sh 20251231
```

This creates `points3d.h5` containing 3D coordinates for all landmarks across all frames.

**Note:** Edit `triangulate.sh` to adjust the frame range if needed.

### 9. 3D Visualization

Create an animated 3D skeleton video:

```bash
python3 scripts/visualize_3d.py <points3d_file> [output_video]
```

**Example:**
```bash
python3 scripts/visualize_3d.py session/20251231/points3d.h5 output_3d.mp4
```

Creates a rotating 3D visualization of the tracked skeleton.

## File Formats

### Analysis Files (`*_analysis.h5`)

HDF5 files containing 2D pose tracking data:
- **Dataset:** `tracks` with shape `(n_tracks, 2, n_nodes, n_frames)`
  - n_tracks: Usually 1 (single person tracking)
  - 2: x, y coordinates
  - n_nodes: 33 (MediaPipe landmarks)
  - n_frames: Number of video frames

### 3D Points File (`points3d.h5`)

HDF5 file containing triangulated 3D coordinates:
- **Dataset:** `points3d` with shape `(n_frames, n_nodes, 3)`
  - 3: x, y, z coordinates in meters

### Calibration File (`calibration.toml`)

TOML file containing:
- Camera intrinsic parameters (focal length, principal point, distortion)
- Camera extrinsic parameters (rotation, translation)
- Camera names and metadata

## Scripts Reference

### Core Pipeline Scripts

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `detect_clap.py` | Synchronize videos | Raw videos | Synced videos |
| `validate_sync.py` | Verify synchronization | Synced videos | Validation report |
| `prepare_session.py` | Create session structure | Synced videos + CSV | Session directory |
| `calibrate.sh` | Camera calibration | Calibration videos | calibration.toml |
| `track_mediapipe.py` | 2D pose tracking | Videos | *_analysis.h5 |
| `triangulate.sh` | 3D triangulation | 2D tracking + calib | points3d.h5 |

### Utility Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `generate_board.py` | Generate ChArUco calibration board | Creates board.png and PDFs |
| `edit_tracking.py` | Manual tracking correction | Interactive GUI |
| `visualize_tracking.py` | 2D tracking overlay | Creates *_tracked.mp4 |
| `visualize_3d.py` | 3D skeleton animation | Creates animated video |
| `convert_openpose_to_sleap.py` | Convert OpenPose → SLEAP | Format conversion |

## Tips and Best Practices

### Video Capture
- Use good lighting and avoid motion blur
- Ensure cameras have overlapping fields of view
- Keep the subject in frame for all cameras
- Use tripods for stable camera positions
- Record at 30-60 fps for smooth motion

### Synchronization
- Make a loud, sharp clap at the beginning
- Keep cameras recording through the clap
- The clap should be clearly audible in all cameras
- Avoid background noise during synchronization clap

### Calibration
- Use the provided ChArUco board PDFs (pre-generated in repository)
- Print at 100% scale and mount on rigid backing (foam board)
- Verify marker size after printing (should be exactly 40mm)
- Move board slowly and smoothly during calibration
- Cover the entire camera view with different board positions
- Tilt the board at various angles
- ChArUco allows partial board visibility for easier calibration
- Aim for reprojection error < 1 pixel

### Tracking
- Wear fitted clothing for better landmark detection
- Avoid baggy clothes that obscure body contours
- Ensure subject is clearly visible in all cameras
- Use the manual editor for any tracking failures
- Check tracking quality with visualization before triangulation

### Troubleshooting
- **Poor synchronization:** Check audio levels, ensure clap is loud enough
- **Calibration fails:** Add more ChArUco board positions, ensure board is flat and markers are not damaged
- **Board not detected:** Verify markers are sharp and high-contrast, check lighting
- **Wrong board dimensions:** Verify printing scale is 100%, measure markers (should be 40mm)
- **Tracking errors:** Use manual editor, improve lighting, reduce motion blur
- **Triangulation errors:** Verify calibration quality, check camera overlap

## Advanced Usage

### Custom MediaPipe Model Path

```bash
python3 scripts/track_mediapipe.py session/20251231/take1 \
    --model path/to/pose_landmarker_heavy.task
```

### Batch Processing Multiple Sessions

```bash
for session in session/*/; do
    echo "Processing $session"
    ./scripts/triangulate.sh $(basename "$session")
done
```

## Dependencies

See `requirements.txt` for complete list. Main dependencies:
- **numpy<2** - Numerical computing (version constraint for compatibility)
- **opencv-python** - Computer vision and video processing
- **mediapipe** - 2D pose estimation
- **h5py** - HDF5 file format support
- **matplotlib** - Visualization and animation
- **Pillow** - Image processing
- **reportlab** - PDF generation for calibration boards
- **sleap-anipose** - Multi-camera calibration and triangulation

## Project Structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── board.toml                       # ChArUco board configuration
├── board.png                        # High-res board image (300 DPI)
├── board_A3.pdf                     # A3 format (single page)
├── board_A4_multipage.pdf           # A4 format (2 pages)
├── board_Letter_multipage.pdf       # US Letter format (2 pages)
├── scripts/
│   ├── generate_board.py            # Board generation
│   ├── detect_clap.py               # Video synchronization
│   ├── validate_sync.py             # Sync validation
│   ├── prepare_session.py           # Session setup
│   ├── calibrate.sh                 # Camera calibration
│   ├── track_mediapipe.py           # 2D pose tracking
│   ├── edit_tracking.py             # Manual correction
│   ├── visualize_tracking.py        # 2D visualization
│   ├── triangulate.sh               # 3D triangulation
│   └── visualize_3d.py              # 3D visualization
└── session/                         # Session data (created by workflow)
    └── <session_name>/
        ├── calibration/
        ├── take1/
        └── take2/
```

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0).

**Why GPL-3.0?**

This project uses [sleap-anipose](https://github.com/talmolab/sleap-anipose), which is licensed under GPL-3.0. As required by GPL-3.0's copyleft provisions, this project must also be distributed under GPL-3.0 or a compatible license.

**What this means:**

- ✓ You can freely use, modify, and distribute this software
- ✓ You can use it for commercial purposes
- ✓ Source code must be made available when distributing
- ✓ Modifications must also be licensed under GPL-3.0
- ✓ Changes must be documented

**Other dependencies** (GPL-compatible):
- [MediaPipe](https://github.com/google-ai-edge/mediapipe) - Apache License 2.0
- numpy, h5py - BSD 3-Clause License
- opencv-python - MIT License
- matplotlib - PSF-based License
- Pillow - HPND License

See the [LICENSE](LICENSE) file for the full license text.

## Acknowledgments

- Built with [MediaPipe](https://google.github.io/mediapipe/) for 2D pose estimation
- Uses [sleap-anipose](https://github.com/talmolab/sleap-anipose) for calibration and triangulation
- Inspired by multi-camera motion capture research

## Support

For issues, questions, or contributions, please refer to the project repository.
