# Card Counter

Real-time playing card detection and blackjack decision engine. Uses YOLOv8 for card recognition and expected value calculations to determine optimal play.

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

## Setup

```bash
python -m venv venv
source venv/bin/activate
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
python live_detect.py --video recordings/20260212_165642.mp4

# Adjust confidence threshold
python live_detect.py --latest --conf 0.7
```

Output: List of detected card values (e.g., `['4', '10', 'A', 'K']`)

### Screen Recording

```bash
cd model
python screen_record.py  # Records 7 seconds, saves to recordings/
```

### Training

```bash
cd model
python train.py --epochs 50 --batch 16 --device mps  # or 'cuda', 'cpu'
```

## Model Performance

- **mAP@0.5**: 0.83 (at 50% confidence)
- **Classes**: 52 (full deck: 2-10, J, Q, K, A × 4 suits)

## Algorithm

The decision engine calculates expected value for hitting vs standing:

1. Track cards seen → compute remaining deck
2. Simulate dealer outcomes via `dealerRecurse()` (dealer hits <17, stands 17+)
3. Calculate `standingEV()` and `hittingEV()`
4. Hit if `hittingEV > standingEV`

```cpp
bool shouldHit(const vector<int>& played, int num_decks = 1);
```

## Dependencies

- ultralytics (YOLOv8)
- opencv-python
- torch
- mss (screen capture)
