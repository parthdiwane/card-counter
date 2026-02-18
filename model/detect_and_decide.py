#!/usr/bin/env python3

import sys
import math
import cv2

from detector import (
    load_model,
    detect_cards_in_frame,
    get_latest_recording,
    MODEL_PATH,
)


def detect_cards(video_path, model, conf=0.7):
    """
    Detect cards and return them sorted by x-position (left to right).
    Returns list of dicts with card info and bounding box coordinates.
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
            if det.label not in best_detections or det.confidence > best_detections[det.label].confidence:
                best_detections[det.label] = det

    cap.release()

    # Sort by x position (left to right)
    sorted_cards = sorted(best_detections.values(), key=lambda d: d.center_x)

    # Return list of dicts with all info
    return [
        {
            'card_str': det.card_value,
            'int_value': det.int_value,
            'x1': det.x1,
            'y1': det.y1,
            'x2': det.x2,
            'y2': det.y2,
            'center_x': det.center_x,
            'center_y': det.center_y,
        }
        for det in sorted_cards
    ]


def build_played_list(cards):
    """
    Build played list from detected cards.
    cards: list of dicts with 'card_str', 'int_value', 'center_x', 'center_y', etc.
    """
    if len(cards) < 2:
        return [], [], []

    player_cards, dealer_cards = [], []
    num_cards = len(cards)


    for index in range(0, num_cards, 2):

        if(index + 1 >= num_cards):
            break # odd number of cards thus no pair for this card

        card1_coords = [cards[index]['center_x'], cards[index]['center_y']]
        card2_coords = [cards[index + 1]['center_x'], cards[index + 1]['center_y']]



        card1_x_sqaured, card1_y_sqaured = pow(card1_coords[0], 2), pow(card1_coords[1], 2)
        card2_x_sqaured, card2_y_sqaured = pow(card2_coords[0], 2), pow(card2_coords[1], 2)


        # getting the distance from the origin to the center of the card --> card with closer distance is the dealer card 
        dist_card1 = math.sqrt(card1_x_sqaured + card1_y_sqaured)
        dist_card2 = math.sqrt(card2_x_sqaured + card2_y_sqaured)

        
        if(dist_card1 < dist_card2): # first card is the dealer card --> close to the dealer
            dealer_cards.append(cards[index]['int_value'])
            player_cards.append(cards[index + 1]['int_value'])
        else:
            dealer_cards.append(cards[index + 1]['int_value'])
            player_cards.append(cards[index]['int_value'])


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

    print(f"Detected cards (left to right): {[c['card_str'] for c in cards]}", file=sys.stderr)

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
