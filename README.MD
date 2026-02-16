<div align="center">

# Card Counter

**Real-time playing card detection and blackjack decision engine**

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Computer%20Vision-00FFFF?style=for-the-badge)](https://github.com/ultralytics/ultralytics)
[![C++](https://img.shields.io/badge/C++-Algorithm-00599C?style=for-the-badge&logo=cplusplus&logoColor=white)](https://isocpp.org)

Uses YOLOv8 for card recognition and expected value calculations to determine optimal play.

</div>

---

## Overview

This project combines computer vision with game theory to create a blackjack assistant that:

- Detects playing cards in real-time using a custom-trained YOLOv8 model
- Tracks cards seen throughout the game
- Calculates expected value for hit/stand decisions
- Recommends optimal plays based on remaining deck composition

## Demo

| Demo | Description |
|------|-------------|
| [Demo 1](https://github.com/user-attachments/assets/8e14fd42-f3c8-4fce-b0aa-a1febc32b74e

) | Card detection in action |
| [Demo 2](model/recordings/Screen%20Recording%202026-02-15%20at%205.39.55%20PM.mov) | Full gameplay example |

## Project Structure

```
card-counter/
├── model/                  # Card detection
│   ├── live_detect.py      # Run detection on camera or video
│   ├── screen_record.py    # Record screen for later analysis
│   ├── train.py            # Train the YOLO model
│   ├── recordings/         # Saved screen recordings
│   └── data/               # Training dataset (52 card classes)
├── algorithm/              # Blackjack strategy
│   ├── main.cpp            # EV-based hit/stand decision engine
│   └── card_helpers.h      # Deck tracking and dealer simulation
└── runs/                   # Trained model weights
```

## Getting Started

### Prerequisites

- Python 3.8+
- C++ compiler (for algorithm)
- Webcam or screen capture capability

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/card-counter.git
cd card-counter

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Card Detection

```bash
cd model

# Live webcam detection
python live_detect.py

# Process most recent screen recording
python live_detect.py --latest

# Process specific video file
python live_detect.py --video recordings/your_video.mp4

# Adjust confidence threshold
python live_detect.py --latest --conf 0.7
```

**Output:** List of detected card values (e.g., `['4', '10', 'A', 'K']`)

### Screen Recording

```bash
cd model
python screen_record.py  # Records 7 seconds, saves to recordings/
```

### Training Your Own Model

```bash
cd model
python train.py --epochs 50 --batch 16 --device mps  # or 'cuda', 'cpu'
```

## Model Performance

| Metric | Value |
|--------|-------|
| **mAP@0.5** | 0.83 |
| **Classes** | 52 (full deck) |
| **Cards** | 2-10, J, Q, K, A × 4 suits |

## How the Algorithm Works

The decision engine calculates expected value for hitting vs standing:

```
┌─────────────────────────────────────────────────────────┐
│  1. Track cards seen → compute remaining deck           │
│  2. Simulate dealer outcomes via dealerRecurse()        │
│     (dealer hits <17, stands 17+)                       │
│  3. Calculate standingEV() and hittingEV()              │
│  4. Hit if hittingEV > standingEV                       │
└─────────────────────────────────────────────────────────┘
```

```cpp
bool shouldHit(const vector<int>& played, int num_decks = 1);
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Card Detection | YOLOv8 (Ultralytics) |
| Video Processing | OpenCV |
| Deep Learning | PyTorch |
| Screen Capture | MSS |
| Decision Engine | C++ |

## Dependencies

```
ultralytics
opencv-python
torch
mss
```

---

<div align="center">

**Built for educational purposes. We are not responsible for any losses or damages associated with using this technology.**

**Developed By: Parth Diwane and Aarnav Koushik**

</div>

---

> **Disclaimer:** This README was generated using Claude and may contain inaccuracies. If you encounter any issues, please email: parthdiwane@ucsb.edu
