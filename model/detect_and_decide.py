#!/usr/bin/env python3

import sys
import cv2

from detector import (
    load_model,
    detect_cards_in_frame,
    get_latest_recording,
    CARD_TO_INT,
    MODEL_PATH,
)


def detect_cards(video_path, model, conf=0.7):
    """
    Detect cards and return them sorted by x-position (left to right).
    Returns list of (card_str, int_value) tuples.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    # Track best detection per card label
    best_detections = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection using shared detector
        detections = detect_cards_in_frame(model, frame, conf=conf)

        for det in detections:
            # Keep the highest confidence detection for each card label
            if det.label not in best_detections or det.confidence > best_detections[det.label][1]:
                best_detections[det.label] = (det.center_x, det.confidence, det.int_value, det.card_value)

    cap.release()

    # Sort by x position (left to right)
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

    model = load_model(str(MODEL_PATH))
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
