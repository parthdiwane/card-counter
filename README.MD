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
- Automatically determines dealer vs player cards based on screen position
- Calculates expected value for hit/stand decisions
- Recommends optimal plays based on remaining deck composition
- Displays live detection visualization with bounding boxes, center points, and distance lines

## Demo

### EV Algorithm

https://github.com/user-attachments/assets/8e14fd42-f3c8-4fce-b0aa-a1febc32b74e

### Vision Model

https://github.com/user-attachments/assets/055eb7a2-118c-4e09-86d8-228efabf4f95

## Project Structure

```
card-counter/
├── run_counter.sh          # Main pipeline script
├── requirements.txt        # Python dependencies
├── model/                  # Card detection
│   ├── detector.py         # Shared detection module (YOLO wrapper, Detection class)
│   ├── detect_and_decide.py # Automated detection with dealer/player sorting
│   ├── live_detect.py      # Interactive detection with visualization
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
- C++ compiler (g++ with C++17 support)
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

# Compile the algorithm (optional - run_counter.sh does this automatically)
cd algorithm && g++ -std=c++17 -O2 -o alg main.cpp
```

## Usage

### Full Pipeline (Recommended)

The easiest way to use the card counter is with the main pipeline script:

```bash
# Record screen, detect cards, get decision
./run_counter.sh

# Use a specific video file
./run_counter.sh /path/to/video.mp4
```

This will:
1. Record your screen (or use provided video)
2. Open a live detection popup showing:
   - Dark blue bounding boxes around detected cards
   - Center points on each card
   - Colored lines from origin to each card center (for distance visualization)
   - Card labels with confidence percentages
3. Automatically sort cards into dealer/player hands based on distance from origin
4. Output HIT or STAND decision

Press **Q** during the visualization to skip ahead.

### Card Detection Only

```bash
cd model

# Detect cards with live visualization
python detect_and_decide.py --video /path/to/video.mp4

# Detect without visualization (headless mode)
python detect_and_decide.py --video /path/to/video.mp4 --no-display

# Interactive detection (for debugging)
python live_detect.py --latest
python live_detect.py --video /path/to/video.mp4
```

### Screen Recording

```bash
cd model
python screen_record.py  # Records 15 seconds, saves to recordings/
```

### Training Your Own Model

```bash
cd model
python train.py --epochs 50 --batch 16 --device mps  # or 'cuda', 'cpu'
```

## How It Works

### Card Detection & Position Tracking

The detection pipeline uses distance from the screen origin (top-left corner) to determine which cards belong to the dealer vs player:

1. **Detection**: YOLOv8 detects cards and returns bounding box coordinates
2. **Center Calculation**: For each card, the center point is calculated
3. **Distance Sorting**: Cards closer to the origin are assigned to the dealer; cards farther away are assigned to the player
4. **Pairing**: Cards are paired (dealer card, player card) based on their relative distances

### Decision Algorithm

The C++ decision engine calculates expected value for hitting vs standing:

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

## Model Performance

| Metric | Value |
|--------|-------|
| **mAP@0.5** | 0.83 |
| **Classes** | 52 (full deck) |
| **Cards** | 2-10, J, Q, K, A × 4 suits |

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
ultralytics>=8.0.0
opencv-python>=4.8.0
torch>=2.0.0
numpy>=1.24.0
mss
```

See `requirements.txt` for the full list.

---

<div align="center">

**Built for educational purposes. We are not responsible for any losses or damages associated with using this technology.**

**Developed By: Parth Diwane and Aarnav Koushik**


</div>

---
<div align="center">
**This README file was created using claude, so it may contain inaccuracies. If you run into any issues please email use at: parthdiwane@ucsb.edu or akoushik@ucsb.edu**
</div>