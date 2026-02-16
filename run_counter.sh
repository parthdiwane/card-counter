#!/bin/bash
#
# Card Counter - Detects cards from latest recording and outputs HIT/STAND decision
#
# Usage: ./run_counter.sh
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

CARDS=$(python3 "$MODEL_DIR/detect_and_decide.py")

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
