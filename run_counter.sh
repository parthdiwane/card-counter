#!/bin/bash
#
# Card Counter - Detects cards from latest recording and outputs HIT/STAND decision
#
# Usage:
#   ./run_counter.sh                  # Use latest recording
#   ./run_counter.sh path/to/video.mp4  # Use specific video file
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

# Run detection and get card list
echo "=== Card Counter ==="
echo ""

# Check if a video path was provided
if [ -n "$1" ]; then
    CARDS=$(python3 "$MODEL_DIR/detect_and_decide.py" --video "$1")
else
    CARDS=$(python3 "$MODEL_DIR/detect_and_decide.py")
fi

if [ -z "$CARDS" ]; then
    echo "Failed to detect cards"
    exit 1
fi

# Get decision from algorithm
DECISION=$("$ALGO_DIR/alg" "$CARDS")

echo ""
echo "================================"
echo "  >>> $DECISION <<<"
echo "================================"
