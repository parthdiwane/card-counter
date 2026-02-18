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


def get_distinct_colors(n):
    """Generate n visually distinct colors in BGR format."""
    colors = [
        (0, 255, 0),     # Green
        (255, 0, 0),     # Blue
        (0, 0, 255),     # Red
        (255, 255, 0),   # Cyan
        (255, 0, 255),   # Magenta
        (0, 255, 255),   # Yellow
        (128, 0, 255),   # Orange
        (255, 128, 0),   # Light blue
        (0, 128, 255),   # Orange-red
        (128, 255, 0),   # Spring green
    ]
    # Cycle through colors if we need more than 10
    return [colors[i % len(colors)] for i in range(n)]


def detect_cards(video_path, model, conf=0.7, display=True):
    """
    Detect cards and return them sorted by x-position (left to right).
    Shows live detection popup if display=True.
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

        # Draw detections on frame for display
        if display:
            display_frame = frame.copy()

            # Generate distinct colors for each detection
            colors = get_distinct_colors(len(detections))

            # First pass: draw all lines from origin (so they appear behind everything)
            for i, det in enumerate(detections):
                card_color = colors[i]
                center_x = int(det.center_x)
                center_y = int(det.center_y)
                cv2.line(display_frame, (0, 0), (center_x, center_y), card_color, 2)

            # Second pass: draw bounding boxes, center points, and labels on top
            DARK_BLUE = (139, 0, 0)  # Dark blue in BGR format

            for i, det in enumerate(detections):
                card_color = colors[i]
                center_x = int(det.center_x)
                center_y = int(det.center_y)

                # Draw bounding box in dark blue
                cv2.rectangle(display_frame, (det.x1, det.y1), (det.x2, det.y2), DARK_BLUE, 3)

                # Draw center point (colored per card for the line)
                cv2.circle(display_frame, (center_x, center_y), 8, card_color, -1)  # Filled circle
                cv2.circle(display_frame, (center_x, center_y), 8, (0, 0, 0), 2)  # Black outline

                # Draw label with confidence (with background for visibility)
                label_text = f"{det.card_value} ({det.confidence:.0%})"
                (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(display_frame, (det.x1, det.y1 - text_h - 10), (det.x1 + text_w + 5, det.y1), DARK_BLUE, -1)
                cv2.putText(display_frame, label_text, (det.x1 + 2, det.y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Draw center coordinates
                coord_text = f"({center_x}, {center_y})"
                cv2.putText(display_frame, coord_text, (center_x - 40, center_y + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # Draw origin marker
            cv2.circle(display_frame, (0, 0), 10, (255, 255, 255), -1)
            cv2.putText(display_frame, "Origin", (5, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Show count of unique cards detected so far
            info_text = f"Unique cards detected: {len(best_detections)}"
            cv2.putText(display_frame, info_text, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow('Card Detection - Press Q to skip', display_frame)

            # Allow early exit with 'q' key, but don't block
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if display:
        cv2.destroyAllWindows()

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
    parser.add_argument('--no-display', action='store_true', help='Disable live detection popup')
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
    cards = detect_cards(video_path, model, display=not args.no_display)

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
