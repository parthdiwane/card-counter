#!/usr/bin/env python3

import sys
import cv2
import numpy as np

from detector import (
    load_model,
    detect_cards_in_frame,
    get_latest_recording,
    MODEL_PATH,
)

from card_depth import (
    load_depth_model,
    process_frame_depth,
    depth_to_heatmap,
    get_card_depths_from_frame,
    draw_depth_heatmap_with_cards,
    get_detected_cards,
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


def draw_card_detection_frame(frame, detections, best_detections_count):
    """Draw card detection visualization on a frame."""
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
    info_text = f"Unique cards detected: {best_detections_count}"
    cv2.putText(display_frame, info_text, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return display_frame


def detect_cards_with_depth(video_path, card_model, depth_model, depth_transform, device=None, conf=0.7, display=True, depth_skip=3):
    """
    Detect cards with depth information.
    Shows combined view: card detection and depth heat map side by side.
    Returns list of dicts with card info, bounding box, and depth.

    depth_skip: Only compute depth every N frames for performance (default: 3)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    # Track best detection per card label (with depth info)
    best_detections = {}

    frame_count = 0
    cached_depth_map = None
    cached_heatmap = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Run card detection (every frame)
        detections = detect_cards_in_frame(card_model, frame, conf=conf)

        # Get depth map (only every N frames for performance)
        if cached_depth_map is None or frame_count % depth_skip == 0:
            depth_map = process_frame_depth(depth_model, depth_transform, frame, device)
            cached_depth_map = depth_map
            cached_heatmap = depth_to_heatmap(depth_map)
        else:
            depth_map = cached_depth_map

        # Get detected cards as dicts for depth processing
        detected_cards_dicts = get_detected_cards(frame, card_model)

        # Get depth for each card
        card_depths = get_card_depths_from_frame(depth_map, detected_cards_dicts)

        # Update best detections - keep highest confidence for each label
        for card_info in card_depths:
            label = card_info['label']
            if label not in best_detections:
                best_detections[label] = card_info

        if display:
            # Card Detection view
            detection_display = draw_card_detection_frame(frame, detections, len(best_detections))

            # Depth Heat Map view (use cached heatmap for performance)
            heatmap_display = draw_depth_heatmap_with_cards(cached_heatmap.copy(), card_depths)

            # Resize both to same height for side-by-side display
            h1, w1 = detection_display.shape[:2]
            h2, w2 = heatmap_display.shape[:2]
            target_height = min(h1, h2, 720)  # Cap at 720p for performance

            scale1 = target_height / h1
            scale2 = target_height / h2

            detection_resized = cv2.resize(detection_display, (int(w1 * scale1), target_height))
            heatmap_resized = cv2.resize(heatmap_display, (int(w2 * scale2), target_height))

            # Combine side by side
            combined = np.hstack([detection_resized, heatmap_resized])

            cv2.imshow('Card Detection + Depth - Press Q to skip', combined)

            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if display:
        cv2.destroyAllWindows()

    # Sort by x position (left to right)
    sorted_cards = sorted(best_detections.values(), key=lambda d: d['center_x'])

    return sorted_cards


def build_played_list_by_depth(cards):
    """
    Build played list from detected cards using DEPTH for dealer/player classification.
    Further cards (higher depth) = dealer cards
    Closer cards (lower depth) = player cards

    cards: list of dicts with 'card_str', 'int_value', 'depth', 'center_x', etc.
    """
    if len(cards) < 2:
        return [], [], []

    player_cards, dealer_cards = [], []
    num_cards = len(cards)

    # Process cards in pairs (sorted left to right)
    for index in range(0, num_cards, 2):
        if index + 1 >= num_cards:
            break  # Odd number of cards, no pair for this card

        card1 = cards[index]
        card2 = cards[index + 1]

        # Use depth to classify: higher depth = farther = dealer
        depth1 = card1['depth']
        depth2 = card2['depth']

        if depth1 > depth2:
            # Card 1 is farther (dealer), Card 2 is closer (player)
            dealer_cards.append(card1['card'])
            player_cards.append(card2['card'])
        else:
            # Card 2 is farther (dealer), Card 1 is closer (player)
            dealer_cards.append(card2['card'])
            player_cards.append(card1['card'])

    # Pad with zeros if needed
    while len(dealer_cards) < len(player_cards):
        dealer_cards.append(0)

    # Interleave: dealer_1, player_1, dealer_2, player_2, ...
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

    # Load both models
    print("Loading card detection model...", file=sys.stderr)
    card_model = load_model(str(MODEL_PATH))

    print("Loading depth estimation model...", file=sys.stderr)
    depth_model, depth_transform, device = load_depth_model()

    # Detect cards with depth
    cards = detect_cards_with_depth(
        video_path, card_model, depth_model, depth_transform, device,
        display=not args.no_display
    )

    if not cards:
        print("ERROR: No cards detected", file=sys.stderr)
        sys.exit(1)

    print(f"Detected cards (left to right): {[c['card_str'] for c in cards]}", file=sys.stderr)
    depth_info = [(c['card_str'], f"{c['depth']:.3f}") for c in cards]
    print(f"Card depths: {depth_info}", file=sys.stderr)

    # Build played list using depth-based classification
    played, player_cards, dealer_cards = build_played_list_by_depth(cards)

    if not played:
        print("ERROR: Need at least 2 cards", file=sys.stderr)
        sys.exit(1)

    player_total = sum(player_cards)
    dealer_total = sum(dealer_cards)

    print(f"Player hand (closer/lower depth): {player_cards} (total: {player_total})", file=sys.stderr)
    print(f"Dealer hand (farther/higher depth): {dealer_cards} (total: {dealer_total})", file=sys.stderr)

    # Output for algorithm
    print(",".join(map(str, played)))


if __name__ == "__main__":
    main()
