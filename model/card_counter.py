#!/usr/bin/env python3
"""
Card Counter Integration Script

Records screen, detects cards using YOLO, and sends to the C++ algorithm
for hit/stand decisions.

Usage:
    python card_counter.py                    # Live screen capture mode
    python card_counter.py --record 10        # Record for 10 seconds first
    python card_counter.py --latest           # Use most recent recording
    python card_counter.py --video path.mp4   # Use specific video file
"""

import cv2
import subprocess
import sys
from pathlib import Path
import argparse
import numpy as np
import mss
import time
from datetime import datetime

# Add model directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from ultralytics import YOLO

# Card value conversion: string card values to blackjack integer values
CARD_TO_INT = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
    'J': 10, 'Q': 10, 'K': 10, 'A': 11
}

# Paths
MODEL_PATH = Path(__file__).parent.parent / "runs/detect/train/weights/best.pt"
ALGORITHM_PATH = Path(__file__).parent.parent / "algorithm" / "alg"
RECORDINGS_DIR = Path(__file__).parent / "recordings"


def card_string_to_value(card_str: str) -> int:
    """Convert card string (e.g., '10', 'A', 'K') to blackjack integer value."""
    card_str = card_str.upper().strip()
    # Handle card codes like '10c', 'Ah' - extract just the value
    if len(card_str) > 1 and card_str[-1].lower() in 'cdhs':
        card_str = card_str[:-1]
    return CARD_TO_INT.get(card_str, 0)


def record_screen(duration: float = 10.0) -> str:
    """Record screen for specified duration, return filepath."""
    RECORDINGS_DIR.mkdir(exist_ok=True)

    FPS = 30
    MONITOR_NUM = 1  # Primary monitor

    with mss.mss() as sct:
        monitor = sct.monitors[MONITOR_NUM]
        w, h = monitor["width"], monitor["height"]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".mp4"
        filepath = str(RECORDINGS_DIR / filename)
        out = cv2.VideoWriter(filepath, fourcc, FPS, (w, h))

        frame_time = 1 / FPS
        print(f"Recording screen for {duration} seconds...")
        start_time = time.time()

        while time.time() - start_time < duration:
            start = time.time()
            frame = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            out.write(frame)

            elapsed = time.time() - start
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)

        out.release()
        print(f"Saved recording to: {filepath}")
        return filepath


def get_latest_recording() -> str:
    """Get the most recent recording from the recordings folder."""
    if not RECORDINGS_DIR.exists():
        return None
    recordings = sorted(RECORDINGS_DIR.glob("*.mp4"))
    return str(recordings[-1]) if recordings else None


def detect_cards_in_video(video_source, model, conf=0.5) -> list:
    """
    Detect cards in video and return list of unique card values.
    Returns list of integers (blackjack values).
    """
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Could not open video source: {video_source}")
        return []

    all_detected_cards = set()
    frame_count = 0

    print("Processing video for card detection...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Run detection
        results = model(frame, conf=conf, verbose=False)

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0].cpu().numpy())
                    label = model.names[cls]
                    # Extract just the value (e.g., '4s' -> '4', 'Ah' -> 'A')
                    card_value = label[:-1].upper()
                    all_detected_cards.add(card_value)

    cap.release()
    print(f"Processed {frame_count} frames")

    return list(all_detected_cards)


def detect_cards_live(model, conf=0.5, display=True) -> list:
    """
    Live screen capture with real-time card detection.
    Press 'q' to quit and get the final card list.
    Press 'c' to clear detected cards.
    Press 'enter' to send current cards to algorithm.
    """
    print("Starting live detection...")
    print("Controls:")
    print("  'q' - Quit")
    print("  'c' - Clear detected cards")
    print("  'enter' - Send cards to algorithm")
    print("  'd' - Add card as DEALER card")
    print("  'p' - Add card as PLAYER card")
    print()

    with mss.mss() as sct:
        monitor = sct.monitors[1]  # Primary monitor

        # Track cards by position (dealer vs player)
        dealer_cards = []
        player_cards = []
        current_frame_cards = set()

        while True:
            # Capture screen
            frame = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # Resize for display
            scale = 0.5
            display_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

            # Run detection
            results = model(frame, conf=conf, verbose=False)

            current_frame_cards.clear()
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        cls = int(box.cls[0].cpu().numpy())
                        label = model.names[cls]
                        card_value = label[:-1].upper()
                        current_frame_cards.add(card_value)

                        # Draw on display frame
                        xyxy = box.xyxy[0].cpu().numpy() * scale
                        x1, y1, x2, y2 = map(int, xyxy)
                        conf_val = box.conf[0].cpu().numpy()
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(display_frame, f"{card_value} ({conf_val:.0%})",
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Display info
            info_y = 30
            cv2.putText(display_frame, f"Current frame: {list(current_frame_cards)}",
                        (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            info_y += 30
            cv2.putText(display_frame, f"Dealer: {dealer_cards}",
                        (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            info_y += 30
            cv2.putText(display_frame, f"Player: {player_cards}",
                        (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            info_y += 30
            cv2.putText(display_frame, "d=add dealer, p=add player, c=clear, enter=decide, q=quit",
                        (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            if display:
                cv2.imshow('Card Counter - Live', display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('c'):
                dealer_cards.clear()
                player_cards.clear()
                print("Cleared all cards")
            elif key == ord('d'):
                # Add currently detected cards to dealer
                for card in current_frame_cards:
                    if card not in dealer_cards:
                        dealer_cards.append(card)
                        print(f"Added {card} to dealer. Dealer: {dealer_cards}")
            elif key == ord('p'):
                # Add currently detected cards to player
                for card in current_frame_cards:
                    if card not in player_cards:
                        player_cards.append(card)
                        print(f"Added {card} to player. Player: {player_cards}")
            elif key == 13:  # Enter key
                if dealer_cards and player_cards:
                    # Build alternating list
                    played = build_played_list(dealer_cards, player_cards)
                    print(f"\nSending to algorithm: {played}")
                    decision = get_algorithm_decision(played)
                    print(f">>> DECISION: {decision} <<<\n")
                else:
                    print("Need at least 1 dealer and 1 player card")

    cv2.destroyAllWindows()

    # Return final state
    return build_played_list(dealer_cards, player_cards)


def build_played_list(dealer_cards: list, player_cards: list) -> list:
    """
    Build the alternating [dealer, player, dealer, player, ...] list
    that the C++ algorithm expects.

    Returns list of integers (blackjack values).
    """
    played = []
    max_len = max(len(dealer_cards), len(player_cards))

    for i in range(max_len):
        if i < len(dealer_cards):
            played.append(card_string_to_value(dealer_cards[i]))
        if i < len(player_cards):
            played.append(card_string_to_value(player_cards[i]))

    return played


def get_algorithm_decision(cards: list) -> str:
    """
    Send card list to C++ algorithm and get hit/stand decision.

    Args:
        cards: List of integers (blackjack values) in alternating dealer/player order

    Returns:
        'HIT', 'STAND', or error message
    """
    if not ALGORITHM_PATH.exists():
        # Try to compile
        print("Compiling algorithm...")
        compile_result = subprocess.run(
            ["g++", "-std=c++17", "-O2", "-o", "alg", "main.cpp"],
            cwd=ALGORITHM_PATH.parent,
            capture_output=True
        )
        if compile_result.returncode != 0:
            return f"COMPILE_ERROR: {compile_result.stderr.decode()}"

    # Convert cards list to comma-separated string
    cards_str = ",".join(map(str, cards))

    try:
        result = subprocess.run(
            [str(ALGORITHM_PATH), cards_str],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    except Exception as e:
        return f"ERROR: {e}"


def main():
    parser = argparse.ArgumentParser(description='Card Counter - Screen capture + YOLO + Algorithm')
    parser.add_argument('--record', type=float, metavar='SECONDS',
                        help='Record screen for N seconds before processing')
    parser.add_argument('--latest', action='store_true',
                        help='Use the most recent recording')
    parser.add_argument('--video', type=str,
                        help='Path to video file')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Detection confidence threshold')
    parser.add_argument('--model', type=str, default=str(MODEL_PATH),
                        help='Path to YOLO model')
    parser.add_argument('--no-display', action='store_true',
                        help='Disable visual display (for headless mode)')
    args = parser.parse_args()

    # Load model
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    if args.record:
        # Record then process
        video_path = record_screen(args.record)
        cards = detect_cards_in_video(video_path, model, args.conf)
        print(f"\nDetected cards: {cards}")

        if cards:
            # Convert to integers
            int_cards = [card_string_to_value(c) for c in cards]
            print(f"Card values: {int_cards}")

            # For recorded video, we need manual dealer/player assignment
            print("\nNote: For automatic play, use live mode (no --record flag)")

    elif args.latest:
        video_path = get_latest_recording()
        if not video_path:
            print("No recordings found")
            return
        print(f"Using recording: {video_path}")
        cards = detect_cards_in_video(video_path, model, args.conf)
        print(f"\nDetected cards: {cards}")

    elif args.video:
        cards = detect_cards_in_video(args.video, model, args.conf)
        print(f"\nDetected cards: {cards}")

    else:
        # Live mode - interactive
        print("Starting live mode...")
        played = detect_cards_live(model, args.conf, display=not args.no_display)
        if played:
            print(f"\nFinal played list: {played}")


if __name__ == '__main__':
    main()
