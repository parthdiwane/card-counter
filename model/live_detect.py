"""
Card detection using webcam or video file.
Displays bounding boxes with card labels in real-time.
"""

import cv2
import argparse

from detector import (
    load_model,
    detect_cards_in_frame,
    get_latest_recording,
    CARD_NAMES,
)


def draw_detection(frame, detection):
    """Draw bounding box and label on frame."""
    x1, y1, x2, y2 = detection.x1, detection.y1, detection.x2, detection.y2
    color = detection.color

    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Prepare label text
    text = f"{detection.display_name} ({detection.confidence:.0%})"

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
    parser.add_argument('--model', type=str, default=None,
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
    model = load_model(args.model)

    video_source = None
    is_video_file = False

    if args.latest:
        video_source = get_latest_recording()
        if video_source is None:
            print("No recordings found in recordings folder")
            return
        is_video_file = True
        print(f"Using the most recent recording file: {video_source}")
    elif args.video:
        video_source = args.video
        is_video_file = True
    else:
        video_source = args.camera
        print(f"Opening camera {args.camera}...")

    cap = cv2.VideoCapture(video_source)

    if not is_video_file:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print("Camera could not be opened")
        return

    all_detected_cards = set()
    bounding_box_coords = []

    fps_counter = 0
    fps_start_time = cv2.getTickCount()
    fps_display = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            if is_video_file:
                print("Video complete")
            else:
                print("Could not read camera frames")
            break

        # Run detection using shared detector
        detections = detect_cards_in_frame(model, frame, conf=args.conf, iou=args.iou)

        # Clear per-frame bounding box coords (keep last frame's)
        bounding_box_coords = []

        for det in detections:
            all_detected_cards.add(det.card_value)

            # Store bounding box coordinates
            bounding_box_coords.append({
                'label': det.label,
                'x1': det.x1,
                'y1': det.y1,
                'x2': det.x2,
                'y2': det.y2,
                'center_x': det.center_x,
                'center_y': det.center_y
            })

            frame = draw_detection(frame, det)

        # FPS calculation
        fps_counter += 1
        if fps_counter >= 30:
            fps_end_time = cv2.getTickCount()
            fps_display = 30 / ((fps_end_time - fps_start_time) / cv2.getTickFrequency())
            fps_start_time = fps_end_time
            fps_counter = 0

        # Display info
        info_text = f"FPS: {fps_display:.1f} | Cards: {len(detections)}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Show detected cards list
        if detections:
            y_offset = 60
            cv2.putText(frame, "Detected:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            for det in detections[:10]:  # Show max 10
                y_offset += 25
                cv2.putText(frame, det.display_name, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, det.color, 2)

        cv2.imshow('Card Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    cards_list = sorted(list(all_detected_cards))
    print(f"Detected cards: {cards_list}")
    return cards_list, bounding_box_coords


if __name__ == '__main__':
    main()
