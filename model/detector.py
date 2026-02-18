"""
Shared card detection module.
Centralizes YOLO model loading, detection logic, and card constants.
"""

from ultralytics import YOLO
from pathlib import Path
from dataclasses import dataclass


# Paths
MODEL_PATH = Path(__file__).parent.parent / "runs/detect/train/weights/best.pt"
RECORDINGS_DIR = Path(__file__).parent / "recordings"

# Card value conversion for blackjack
CARD_TO_INT = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
    'J': 10, 'Q': 10, 'K': 10, 'A': 11
}

# Full card names for display
CARD_NAMES = {
    '10c': '10 of Clubs', '10d': '10 of Diamonds', '10h': '10 of Hearts', '10s': '10 of Spades',
    '2c': '2 of Clubs', '2d': '2 of Diamonds', '2h': '2 of Hearts', '2s': '2 of Spades',
    '3c': '3 of Clubs', '3d': '3 of Diamonds', '3h': '3 of Hearts', '3s': '3 of Spades',
    '4c': '4 of Clubs', '4d': '4 of Diamonds', '4h': '4 of Hearts', '4s': '4 of Spades',
    '5c': '5 of Clubs', '5d': '5 of Diamonds', '5h': '5 of Hearts', '5s': '5 of Spades',
    '6c': '6 of Clubs', '6d': '6 of Diamonds', '6h': '6 of Hearts', '6s': '6 of Spades',
    '7c': '7 of Clubs', '7d': '7 of Diamonds', '7h': '7 of Hearts', '7s': '7 of Spades',
    '8c': '8 of Clubs', '8d': '8 of Diamonds', '8h': '8 of Hearts', '8s': '8 of Spades',
    '9c': '9 of Clubs', '9d': '9 of Diamonds', '9h': '9 of Hearts', '9s': '9 of Spades',
    'Ac': 'Ace of Clubs', 'Ad': 'Ace of Diamonds', 'Ah': 'Ace of Hearts', 'As': 'Ace of Spades',
    'Jc': 'Jack of Clubs', 'Jd': 'Jack of Diamonds', 'Jh': 'Jack of Hearts', 'Js': 'Jack of Spades',
    'Kc': 'King of Clubs', 'Kd': 'King of Diamonds', 'Kh': 'King of Hearts', 'Ks': 'King of Spades',
    'Qc': 'Queen of Clubs', 'Qd': 'Queen of Diamonds', 'Qh': 'Queen of Hearts', 'Qs': 'Queen of Spades',
}

# Suit colors for visualization (BGR format)
SUIT_COLORS = {
    'c': (0, 100, 0),      # Clubs - Dark Green
    'd': (0, 0, 255),      # Diamonds - Red
    'h': (0, 0, 200),      # Hearts - Red
    's': (100, 100, 100),  # Spades - Gray
}


@dataclass
class Detection:
    """Represents a single card detection."""
    label: str          # e.g., '4s', 'Ah'
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def card_value(self) -> str:
        # value of the card ex: 4, A, K, Q
        return self.label[:-1].upper()

    @property
    def int_value(self) -> int:
        # black jack integer value
        return CARD_TO_INT.get(self.card_value, 0)

    @property
    def suit(self) -> str:
        return self.label[-1].lower()

    @property
    def color(self) -> tuple:
        # color card
        return SUIT_COLORS.get(self.suit, (255, 255, 255))

    @property
    def display_name(self) -> str:
        return CARD_NAMES.get(self.label, self.label)

    @property
    def center_x(self) -> int:
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self) -> int:
        """Get y-coordinate of bounding box center."""
        return (self.y1 + self.y2) / 2


def load_model(model_path: str = None) -> YOLO:
    """Load YOLO model from path."""
    path = model_path or str(MODEL_PATH)
    print(f"Loading model: {path}")
    return YOLO(path)


def detect_cards_in_frame(model: YOLO, frame, conf: float = 0.5, iou: float = 0.45) -> list[Detection]:
    results = model(frame, conf=conf, iou=iou, verbose=False)
    detections = []

    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                conf_val = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                label = model.names[cls]

                detections.append(Detection(
                    label=label,
                    confidence=conf_val,
                    x1=x1, y1=y1, x2=x2, y2=y2
                ))

    return detections


def get_latest_recording() -> str | None:
    """Get the most recent recording from the recordings folder."""
    if not RECORDINGS_DIR.exists():
        return None
    recordings = sorted(RECORDINGS_DIR.glob("*.mp4"))
    return str(recordings[-1]) if recordings else None
