#!/bin/sh
# Run SLEAP pose tracking on camera videos to generate analysis.h5 files
# Required before running triangulation

session="$1"

if [ -z "$session" ]; then
    echo "Usage: $0 <session_id>"
    echo "Example: $0 20251231"
    exit 1
fi

SESSION_DIR="session/${session}"

if [ ! -d "$SESSION_DIR" ]; then
    echo "Error: Session directory $SESSION_DIR does not exist"
    exit 1
fi

# Check if SLEAP is installed
if ! command -v sleap-track >/dev/null 2>&1; then
    echo "Error: SLEAP is not installed or not in PATH"
    echo ""
    echo "Please install SLEAP first:"
    echo "  - Via conda: conda install -c conda-forge -c nvidia -c sleap sleap"
    echo "  - Or see: https://sleap.ai/installation.html"
    echo ""
    echo "Note: sleap-anipose (for triangulation) is separate from SLEAP (for tracking)"
    exit 1
fi

# Check if model file exists (you need to provide your trained SLEAP model)
MODEL_FILE="models/your_model.slp"
if [ ! -f "$MODEL_FILE" ]; then
    echo "Error: Model file $MODEL_FILE not found"
    echo ""
    echo "You need to provide a trained SLEAP model (.slp file)"
    echo "Update the MODEL_FILE variable in this script to point to your model"
    exit 1
fi

echo "Running SLEAP tracking on camera views for session: $session"
echo ""

# Find all camera directories
CAMERA_DIRS=$(find "$SESSION_DIR" -maxdepth 1 -type d -name "camera*" | sort)

if [ -z "$CAMERA_DIRS" ]; then
    echo "Error: No camera directories found in $SESSION_DIR"
    exit 1
fi

# Count cameras
CAMERA_COUNT=$(echo "$CAMERA_DIRS" | wc -l)
echo "Found $CAMERA_COUNT camera(s) to process"
echo ""

# Track each camera
for CAMERA_DIR in $CAMERA_DIRS; do
    CAMERA_NAME=$(basename "$CAMERA_DIR")

    # Find video file in camera directory
    VIDEO_FILE=$(find "$CAMERA_DIR" -maxdepth 1 -name "*-recording.mp4" -o -name "*.mp4" | head -1)

    if [ -z "$VIDEO_FILE" ]; then
        echo "Warning: No video file found in $CAMERA_DIR, skipping..."
        continue
    fi

    OUTPUT_FILE="$CAMERA_DIR/${CAMERA_NAME}_analysis.h5"

    echo "Processing $CAMERA_NAME..."
    echo "  Video: $VIDEO_FILE"
    echo "  Output: $OUTPUT_FILE"

    sleap-track \
        "$VIDEO_FILE" \
        --model "$MODEL_FILE" \
        --output "$OUTPUT_FILE" \
        --verbosity json \
        --no-empty-frames

    echo ""
done

echo "Tracking complete! Analysis files created in each camera directory."
echo "You can now run triangulation with: ./scripts/triangulate.sh $session"
