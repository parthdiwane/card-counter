"""
Training script for YOLOv8 card detection model.
Trains on 52-class playing card dataset for real-time detection.
"""

from ultralytics import YOLO
import argparse
import torch


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 card detection model')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help='Base model: yolov8n.pt (fast), yolov8s.pt (balanced), yolov8m.pt (accurate)')
    parser.add_argument('--epochs', type=int, default=25, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--device', type=str, default='', help='Device: cpu, 0, 0,1, mps')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    args = parser.parse_args()

    # Auto-detect device
    if not args.device:
        if torch.cuda.is_available():
            args.device = '0'
        elif torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'

    print(f"Using device: {args.device}")

    # Load model
    if args.resume:
        model = YOLO('runs/detect/train/weights/last.pt')
    else:
        model = YOLO(args.model)

    # Train
    model.train(
        data='data.yaml',
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        patience=50,  # Early stopping patience
        save=True,
        plots=True,
        verbose=True,
        val=True,
        # Data augmentation for cards
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15,
        translate=0.1,
        scale=0.5,
        shear=5,
        flipud=0.0,  # Don't flip cards upside down
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        # Validation settings
        max_det=50,  # Max detections per image
        conf=0.1,  # Filter low-confidence predictions to speed up NMS
    )

    print("\nTraining complete!")
    print(f"Best model saved to: runs/detect/train/weights/best.pt")

    # Validate on test set
    print("\nValidating on test set...")
    metrics = model.val(data='data.yaml', split='test')
    print(f"Test mAP50: {metrics.box.map50:.4f}")
    print(f"Test mAP50-95: {metrics.box.map:.4f}")


if __name__ == '__main__':
    main()
