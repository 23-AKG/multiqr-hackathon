import argparse
import json
from pathlib import Path
from ultralytics import YOLO

def run_inference(weights, input_dir, output_json, conf=0.25, iou=0.5):
    model = YOLO(weights)

    results_json = []
    input_dir = Path(input_dir)
    for img_path in input_dir.glob("*.jpg"):
        results = model.predict(
            source=str(img_path),
            conf=conf,
            iou=iou,
            verbose=False
        )

        qrs = []
        for box in results[0].boxes:
            xyxy = box.xyxy[0].tolist()  # [x_min, y_min, x_max, y_max]
            qrs.append({"bbox": [round(v, 2) for v in xyxy]})

        results_json.append({
            "image_id": img_path.stem,
            "qrs": qrs
        })

    # Save JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2)

    print(f"Inference complete. Results saved to {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLOv8 inference for QR code detection")
    parser.add_argument("--weights", type=str, default="runs/detect/qr_augmented2/weights/best.pt", help="Path to trained weights") #use "src/models/best.pt"
    parser.add_argument("--input", type=str, default="dataset/images/test_images", help="Input images folder")
    parser.add_argument("--output", type=str, default="outputs/submission_detection_1.json", help="Output JSON file")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold")

    args = parser.parse_args()
    run_inference(args.weights, args.input, args.output, conf=args.conf, iou=args.iou)
