import torch
import cv2
import numpy as np
from detector import get_latest_recording, detect_cards_in_frame, load_model


frame_output_path = "model/video_frames"
video_path = "model/recordings"


def get_detected_cards(frame, model):
    detections = detect_cards_in_frame(model, frame)

    return [
        {
            'card_str': det.card_value,
            'int_value': det.int_value,
            'label': det.label,
            'x1': det.x1,
            'y1': det.y1,
            'x2': det.x2,
            'y2': det.y2,
            'center_x': int(det.center_x),
            'center_y': int(det.center_y),
        }
        for det in detections
    ]


def get_frames(video_path=None):
    if video_path is None:
        video_path = get_latest_recording()
    if video_path is None:
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames


def load_depth_model():
    # load MiDas
    # Use MiDaS small for faster inference
    # Redirect stdout to stderr during loading to prevent breaking shell parsing
    import sys

    # Suppress stdout during model loading (nested deps print to stdout)
    old_stdout = sys.stdout
    sys.stdout = sys.stderr

    try:
        model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", verbose=False)
        model.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", verbose=False)
        transform = midas_transforms.small_transform
    finally:
        sys.stdout = old_stdout

    return model, transform, device


def depth_to_heatmap(depth_map):
    # fun visuals
    # Normalize depth to 0-255
    depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_normalized.astype(np.uint8)


    heatmap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)

    return heatmap


def process_frame_depth(depth_model, transform, frame, device=None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Apply MiDaS transform
    input_batch = transform(frame_rgb).to(device)

    # Run inference
    with torch.no_grad():
        prediction = depth_model(input_batch)

        # Resize to original frame size
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    return depth_map


def get_card_depths_from_frame(depth_map, detected_cards):
    # depth for each frame
    card_depths = []

    for card in detected_cards:
        x1, x2, y1, y2 = card['x1'], card['x2'], card['y1'], card['y2']

        # Ensure bounds are within depth map
        h, w = depth_map.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)

        if x2 > x1 and y2 > y1:
            card_region = depth_map[y1:y2, x1:x2]
            mean_depth = float(np.mean(card_region))
        else:
            mean_depth = 0.0

        card_depths.append({
            'card': card['int_value'],
            'card_str': card['card_str'],
            'label': card['label'],
            'depth': mean_depth,
            'center_x': card['center_x'],
            'center_y': card['center_y'],
            'x1': card['x1'],
            'y1': card['y1'],
            'x2': card['x2'],
            'y2': card['y2'],
        })

    return card_depths


def draw_depth_heatmap_with_cards(heatmap, card_depths):
    """Draw card annotations on the depth heatmap."""
    display = heatmap.copy()

    for card in card_depths:
        x1, y1, x2, y2 = card['x1'], card['y1'], card['x2'], card['y2']
        center_x, center_y = card['center_x'], card['center_y']

        # Draw bounding box
        cv2.rectangle(display, (x1, y1), (x2, y2), (255, 255, 255), 2)

        # Draw depth value label
        depth_text = f"{card['card_str']} d:{card['depth']:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(depth_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(display, (x1, y1 - text_h - 10), (x1 + text_w + 5, y1), (0, 0, 0), -1)
        cv2.putText(display, depth_text, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw center point
        cv2.circle(display, (center_x, center_y), 5, (255, 255, 255), -1)

    return display


def process_video_with_depth(video_path, card_model, depth_model, depth_transform, device=None, display=True):
    """
    Process video showing both card detection and depth heat map.
    Returns the best detections with depth information for dealer/player classification.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    # Track best detection per card label (highest confidence)
    best_detections = {}

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Run card detection
        detected_cards = get_detected_cards(frame, card_model)

        # Get depth map for this frame
        depth_map = process_frame_depth(depth_model, depth_transform, frame, device)

        # Get depth for each detected card
        card_depths = get_card_depths_from_frame(depth_map, detected_cards)

        # Update best detections (track by label, keep highest confidence)
        for card_info in card_depths:
            label = card_info['label']
            if label not in best_detections:
                best_detections[label] = card_info

        if display:
            # Create depth heat map visualization
            heatmap = depth_to_heatmap(depth_map)
            heatmap_display = draw_depth_heatmap_with_cards(heatmap, card_depths)

            # Show heat map window
            cv2.imshow('Depth Heat Map - Press Q to skip', heatmap_display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if display:
        cv2.destroyWindow('Depth Heat Map - Press Q to skip')

    # Sort by x position (left to right) and return
    sorted_cards = sorted(best_detections.values(), key=lambda d: d['center_x'])

    return sorted_cards


if __name__ == "__main__":
    # Test the depth visualization
    video_path = get_latest_recording()
    if video_path:
        print(f"Processing: {video_path}")
        card_model = load_model()
        depth_model, depth_transform, device = load_depth_model()
        cards = process_video_with_depth(video_path, card_model, depth_model, depth_transform, device, display=True)
        print(f"Detected {len(cards)} unique cards with depth info")
        for card in cards:
            print(f"  {card['card_str']}: depth={card['depth']:.3f}")
