import torch
import cv2
from detector import get_latest_recording, detect_cards_in_frame, load_model


frame_output_path = "model/video_frames"
video_path = "model/recordings"


def get_detected_cards(frame, model):
    detections = detect_cards_in_frame(model, frame)

    return [
        {
            'card_str': det.card_value,
            'int_value': det.int_value,
            'x1': det.x1,
            'y1': det.y1,
            'x2': det.x2,
            'y2': det.y2,
            'center_x': int(det.center_x),
            'center_y': int(det.center_y),
        }
        for det in detections
    ]


def get_frames():
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


# returns a list of depths for the cards
def get_depth():
    model = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True)
    model.eval()

    video_frames = get_frames()
    net_card_depth = []

    for frame in video_frames:
        depth_map = model.infer_pil(frame)

        detected_cards = get_detected_cards()

        for card in detected_cards:
            x1, x2, y1, y2 = card['x1'], card['x2'], card['y1'], card['y2']

            card_depth = depth_map[y1:y2, x1:x2].mean()
            net_card_depth.append({
                'card': card['int_value'],
                'depth': card_depth
            })

    return net_card_depth
    
