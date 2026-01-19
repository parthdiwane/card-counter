#!/bin/bash

# Get screen dimensions on macOS
SCREEN_WIDTH=$(system_profiler SPDisplaysDataType | grep Resolution | head -1 | awk '{print $2}')
SCREEN_HEIGHT=$(system_profiler SPDisplaysDataType | grep Resolution | head -1 | awk '{print $4}')

echo "Detected screen: ${SCREEN_WIDTH}x${SCREEN_HEIGHT}"

agent_s \
    --provider ollama \
    --model qwen2.5:14b \
    --ground_provider ollama \
    --ground_url http://localhost:11434 \
    --ground_model llava \
    --grounding_width $SCREEN_WIDTH \
    --grounding_height $SCREEN_HEIGHT