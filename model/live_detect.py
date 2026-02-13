"""
Card detection using webcam or video file.
Displays bounding boxes with card labels in real-time.
"""

import cv2
from ultralytics import YOLO
import argparse
from pathlib import Path

# Card name mappings for better display
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

# Colors for suits (BGR format)
SUIT_COLORS = {
    'c': (0, 100, 0),      # Clubs - Dark Green
    'd': (0, 0, 255),      # Diamonds - Red
    'h': (0, 0, 200),      # Hearts - Red
    's': (100, 100, 100),  # Spades - Gray
}


def get_color_for_card(card_code):
    """Get display color based on suit."""
    suit = card_code[-1]
    return SUIT_COLORS.get(suit, (255, 255, 255))


def format_card(card_code):
    """Format card code to uppercase (e.g., '4s' -> '4S', 'Ah' -> 'AH')."""
    return card_code.upper()


def get_latest_recording():
    """Get the most recent recording from the recordings folder."""
    recordings_dir = Path(__file__).parent / "recordings"
    if not recordings_dir.exists():
        return None

    recordings = sorted(recordings_dir.glob("*.mp4"))
    if not recordings:
        return None

    # Filenames are YYYYMMDD_HHMMSS.mp4, so sorting alphabetically gives chronological order
    return str(recordings[-1])


def draw_detection(frame, box, label, confidence, color):
    """Draw bounding box and label on frame."""
    x1, y1, x2, y2 = map(int, box)

    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Prepare label text
    display_name = CARD_NAMES.get(label, label)
    text = f"{display_name} ({confidence:.0%})"

    # Calculate text size for background
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

    # Draw background rectangle for text
    cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width + 5, y1), color, -1)

    # Draw text
    cv2.putText(frame, text, (x1 + 2, y1 - 5), font, font_scale, (255, 255, 255), thickness)

    return frame


def main():
    parser = argparse.ArgumentParser(description='Card detection with webcam or video file')
    parser.add_argument('--model', type=str, default='runs/detect/train/weights/best.pt',
                        help='Path to trained model')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--latest', action='store_true', help='Use the most recent recording')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--width', type=int, default=1280, help='Frame width (camera only)')
    parser.add_argument('--height', type=int, default=720, help='Frame height (camera only)')
    args = parser.parse_args()

    # Load model
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    # Determine video source
    video_source = None
    is_video_file = False

    if args.latest:
        video_source = get_latest_recording()
        if video_source is None:
            print(" no recordings found in recordings folder")
            return
        is_video_file = True
        print(f"using the most recent recording file : {video_source}")
    elif args.video:
        video_source = args.video
        is_video_file = True
    else:
        video_source = args.camera
        print(f"opening camera {args.camera}...")

    # Open video source
    cap = cv2.VideoCapture(video_source)

    if not is_video_file:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print("camera could not be opend")
        return

    # Track all unique cards detected throughout the video
    all_detected_cards = set()

    # FPS calculation
    fps_counter = 0
    fps_start_time = cv2.getTickCount()
    fps_display = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            if is_video_file:
                print("vidoe complete")
            else:
                print("could not read cam frames")
            break

        # Run inference
        results = model(frame, conf=args.conf, iou=args.iou, verbose=False)

        # Track detected cards for display
        detected_cards = []

        # Process detections
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())

                    # Get label
                    label = model.names[cls]
                    detected_cards.append((label, conf))
                    all_detected_cards.add(format_card(label))

                    # Get color based on suit
                    color = get_color_for_card(label)

                    # Draw detection
                    frame = draw_detection(frame, xyxy, label, conf, color)

        # Calculate FPS
        fps_counter += 1
        if fps_counter >= 30:
            fps_end_time = cv2.getTickCount()
            fps_display = 30 / ((fps_end_time - fps_start_time) / cv2.getTickFrequency())
            fps_start_time = fps_end_time
            fps_counter = 0

        # Draw FPS and card count
        info_text = f"FPS: {fps_display:.1f} | Cards: {len(detected_cards)}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Draw detected cards list
        if detected_cards:
            y_offset = 60
            cv2.putText(frame, "Detected:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            for card, _ in detected_cards[:10]:  # Show max 10
                y_offset += 25
                card_text = f"{CARD_NAMES.get(card, card)}"
                color = get_color_for_card(card)
                cv2.putText(frame, card_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display frame
        cv2.imshow('Card Detection', frame)

        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Return the list of all unique detected cards
    cards_list = sorted(list(all_detected_cards))
    print(f"Detected cards: {cards_list}")
    return cards_list


if __name__ == '__main__':
    main()
