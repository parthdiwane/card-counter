#!/bin/bash

# Load environment variables
set -a
source "$(dirname "$0")/../.env"
set +a

# Activate the virtual environment
source "$(dirname "$0")/.venv/bin/activate"

# Start Ollama if not already running
if ! curl -s http://localhost:11434/v1/models > /dev/null 2>&1; then
    echo "Starting Ollama server..."
    ollama serve > /dev/null 2>&1 &
    sleep 3  # Wait for server to start
fi

# Get screen dimensions on macOS
SCREEN_WIDTH=$(system_profiler SPDisplaysDataType | grep Resolution | head -1 | awk '{print $2}')
SCREEN_HEIGHT=$(system_profiler SPDisplaysDataType | grep Resolution | head -1 | awk '{print $4}')

echo "Detected screen: ${SCREEN_WIDTH}x${SCREEN_HEIGHT}"

agent_s \
    --provider openai \
    --model qwen2.5:14b \
    --model_url http://localhost:11434/v1 \
    --ground_provider openai \
    --ground_url http://localhost:11434/v1 \
    --ground_model llava \
    --grounding_width $SCREEN_WIDTH \
    --grounding_height $SCREEN_HEIGHT