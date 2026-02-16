#!/usr/bin/env python3


import sys
from pathlib import Path
from ultralytics import YOLO
import cv2

# card value conversion
CARD_TO_INT = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
    'J': 10, 'Q': 10, 'K': 10, 'A': 11
}

MODEL_PATH = Path(__file__).parent.parent / "runs/detect/train/weights/best.pt"
RECORDINGS_DIR = Path(__file__).parent / "recordings"


def get_latest_recording():
    if not RECORDINGS_DIR.exists():
        return None
    recordings = sorted(RECORDINGS_DIR.glob("*.mp4"))
    return str(recordings[-1]) if recordings else None


def detect_cards(video_path, model, conf=0.7):
    """
    Detect cards and return them sorted by x-position (left to right).
    Returns list of (x_position, card_value_int) tuples.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

  
    best_detections = {}  

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf, verbose=False)

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0].cpu().numpy())
                    conf_val = float(box.conf[0].cpu().numpy())
                    label = model.names[cls]  # e.g., '4s', 'Ah'

                    # get x position (center of bounding box)
                    xyxy = box.xyxy[0].cpu().numpy()
                    x_center = (xyxy[0] + xyxy[2]) / 2


                    card_value = label[:-1].upper()
                    int_value = CARD_TO_INT.get(card_value, 0)

                    if label not in best_detections or conf_val > best_detections[label][1]:
                        best_detections[label] = (x_center, conf_val, int_value, card_value)

    cap.release()

    sorted_cards = sorted(best_detections.values(), key=lambda x: x[0])

    return [(c[3], c[2]) for c in sorted_cards]  # (card_str, int_value)


def build_played_list(cards):

    if len(cards) < 2:
        return [], [], []

    
    player_cards = [cards[i][1] for i in range(0, len(cards), 2)]  # indices 0, 2, 4, ...
    dealer_cards = [cards[i][1] for i in range(1, len(cards), 2)]  # indices 1, 3, 5, ...

   # pad w/ zeros if the number of cards in the dist is diff
    while len(dealer_cards) < len(player_cards):
        dealer_cards.append(0)


    played = []
    for i in range(len(player_cards)):
        played.append(dealer_cards[i])
        played.append(player_cards[i])

    return played, player_cards, dealer_cards


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Detect cards and output for algorithm')
    parser.add_argument('--video', type=str, help='Path to specific video file')
    args = parser.parse_args()

    if args.video:
        video_path = args.video
    else:
        video_path = get_latest_recording()
        if not video_path:
            print("ERROR: No recordings found", file=sys.stderr)
            sys.exit(1)

    print(f"Processing: {video_path}", file=sys.stderr)

    model = YOLO(str(MODEL_PATH))
    cards = detect_cards(video_path, model)

    if not cards:
        print("ERROR: No cards detected", file=sys.stderr)
        sys.exit(1)

    print(f"Detected cards (left to right): {[c[0] for c in cards]}", file=sys.stderr)

    played, player_cards, dealer_cards = build_played_list(cards)

    if not played:
        print("ERROR: Need at least 2 cards", file=sys.stderr)
        sys.exit(1)

    player_total = sum(player_cards)
    dealer_total = sum(dealer_cards)

    print(f"Player hand: {player_cards} (total: {player_total})", file=sys.stderr)
    print(f"Dealer hand: {dealer_cards} (total: {dealer_total})", file=sys.stderr)

    
    print(",".join(map(str, played)))


if __name__ == "__main__":
    main()
