import cv2
import json
import argparse
from pathlib import Path
from ultralytics import YOLO

def run_inference(model_path, input_dir, output_file, conf=0.25):
    model = YOLO(model_path)
    detector = cv2.QRCodeDetector()
    results = []

    input_dir = Path(input_dir)
    for img_path in input_dir.glob("*.jpg"):
        image = cv2.imread(str(img_path))
        h, w = image.shape[:2]

        dets = model.predict(source=str(img_path), conf=conf, verbose=False)[0]

        qr_list = []
        for box in dets.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            crop = image[y1:y2, x1:x2]

            value = "unknown"
            try:
                decoded, points, _ = detector.detectAndDecode(crop)
                if decoded:
                    value = decoded
            except Exception:
                pass

            qr_list.append({"bbox": [x1, y1, x2, y2], "value": value})

        results.append({"image_id": img_path.stem, "qrs": qr_list})

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Inference complete. Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="runs/detect/qr_augmented2/weights/best.pt", help="Path to trained model") #use "src/models/best.pt"
    parser.add_argument("--input", type=str, default="dataset/images/test_images", help="Path to test images")
    parser.add_argument("--output", type=str, default="outputs/submission_decoding_2.json", help="Output JSON file")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    run_inference(args.model, args.input, args.output, conf=args.conf)
