#!/bin/bash
#
# Card Counter Pipeline
#
# Flow:
#   1. Record screen (or use provided video)
#   2. Process video with card detection model (popup window)
#   3. Process video with depth model for heat map (popup window)
#   4. Use depth to classify dealer vs player cards (farther = dealer, closer = player)
#   5. Send detected cards to decision algorithm
#
# Usage:
#   ./run_counter.sh                     # Record screen, then detect cards
#   ./run_counter.sh path/to/video.mp4   # Use specific video file (skip recording)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$SCRIPT_DIR/model"
ALGO_DIR="$SCRIPT_DIR/algorithm"

# Compile algorithm if needed
if [ ! -f "$ALGO_DIR/alg" ]; then
    echo "Compiling algorithm..."
    g++ -std=c++17 -O2 -o "$ALGO_DIR/alg" "$ALGO_DIR/main.cpp"
fi

echo "=== Card Counter ==="
echo ""

# Check if a video path was provided
if [ -n "$1" ]; then
    VIDEO_PATH="$1"
    echo "Using provided video: $VIDEO_PATH"
else
    # Step 1: Screen recording
    echo "Step 1: Recording screen..."
    VIDEO_PATH=$(python3 "$MODEL_DIR/screen_record.py")
    echo "Recording saved to: $VIDEO_PATH"
fi

echo ""
echo "Step 2: Detecting cards with depth analysis..."
echo "         (Two popup windows will appear: Card Detection & Depth Heat Map)"
echo "         Press Q to skip processing early"
CARDS=$(python3 "$MODEL_DIR/detect_and_decide.py" --video "$VIDEO_PATH")

if [ -z "$CARDS" ]; then
    echo "Failed to detect cards"
    exit 1
fi

echo ""
echo "Step 3: Running decision algorithm..."
DECISION=$("$ALGO_DIR/alg" "$CARDS")

echo ""
echo "================================"
echo "  >>> $DECISION <<<"
echo "================================"
