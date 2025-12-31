#!/bin/sh
# Run OpenPose BODY_25 on camera videos to generate pose keypoints
# Outputs JSON files that will be converted to SLEAP format

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

# Check if OpenPose is installed
if ! command -v openpose.bin >/dev/null 2>&1 && ! command -v openpose >/dev/null 2>&1; then
    echo "Error: OpenPose is not installed or not in PATH"
    echo ""
    echo "Please install OpenPose:"
    echo "  - See: https://github.com/CMU-Perceptual-Computing-Lab/openpose"
    echo "  - Or use Docker: docker pull cwaffles/openpose"
    echo ""
    exit 1
fi

# Determine OpenPose command
OPENPOSE_CMD="openpose.bin"
if command -v openpose >/dev/null 2>&1; then
    OPENPOSE_CMD="openpose"
fi

echo "Running OpenPose BODY_25 on camera views for session: $session"
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

# Process each camera
for CAMERA_DIR in $CAMERA_DIRS; do
    CAMERA_NAME=$(basename "$CAMERA_DIR")

    # Find video file in camera directory
    VIDEO_FILE=$(find "$CAMERA_DIR" -maxdepth 1 -name "*-recording.mp4" -o -name "*.mp4" | head -1)

    if [ -z "$VIDEO_FILE" ]; then
        echo "Warning: No video file found in $CAMERA_DIR, skipping..."
        continue
    fi

    # Create output directory for OpenPose JSON files
    OUTPUT_DIR="$CAMERA_DIR/openpose_output"
    mkdir -p "$OUTPUT_DIR"

    echo "Processing $CAMERA_NAME..."
    echo "  Video: $VIDEO_FILE"
    echo "  Output: $OUTPUT_DIR"

    # Run OpenPose with BODY_25 model
    $OPENPOSE_CMD \
        --video "$VIDEO_FILE" \
        --model_pose BODY_25 \
        --write_json "$OUTPUT_DIR" \
        --display 0 \
        --render_pose 0

    echo ""
done

echo "OpenPose processing complete!"
echo "JSON keypoint files created in each camera's openpose_output directory."
echo ""
echo "Next step: Convert OpenPose output to SLEAP format"
echo "  ./scripts/convert_openpose_to_sleap.py $session"
