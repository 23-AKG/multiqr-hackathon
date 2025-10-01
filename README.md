# Multi-QR Code Recognition

This repository contains our solution for the **Multi-QR Code Recognition Hackathon**. The task is to detect multiple QR codes in medicine pack images, and optionally decode their contents.

## Repository Structure

```
multiqr-hackathon/
│
├── README.md                  # Setup & usage instructions (this file)
├── requirements.txt           # Python dependencies
├── train.py                   # Training script (YOLOv8)
├── infer.py                   # Inference script for detection
├── infer_decode.py            # Inference script with decoding (bonus)
│
├── data/                      # Placeholder only (not committed)
│   └── demo_images/           # Small demo set of 2-3 images to showcase usage
│
├── dataset/                   # (NOT included in repo)
│   ├── images/
│   │   ├── train_images/      # 200 training images
│   │   └── test_images/       # 50 test images
│   ├── labels/
│   │   └── train_images/      # YOLO-format annotations for train images
│   └── splits/
│       ├── train.txt          # Train split list (paths to images)
│       └── val.txt            # Validation split list
│   └── dataset.yaml           # Dataset definition file
│
├── outputs/
│   ├── submission_detection_1.json   # Stage 1 detection output
│   └── submission_decoding_2.json    # Stage 2 decoding output (bonus)
│
└── src/
    ├── models/
    │   └── best.pt            # Trained YOLOv8 model weights (provided)
```

## Dataset

The dataset folder is **not included** in this repository. Please download it from the official link provided in the hackathon problem statement. Once downloaded, the structure should look like this:

```
dataset/
├── images/
│   ├── train_images/      # 200 images with QR codes
│   └── test_images/       # 50 test images (no annotations)
├── labels/
│   └── train_images/      # YOLO annotations (generated from COCO JSON)
├── splits/
│   ├── train.txt          # List of training images
│   └── val.txt            # List of validation images
└── dataset.yaml           # Dataset definition file
```

This description is included here so that when evaluators see local paths inside scripts, they can understand why paths were structured that way.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/23-AKG/multiqr-hackathon.git
cd multiqr-hackathon
```

2. Create a virtual environment (recommended):

```bash
python -m venv .venv
.venv\Scripts\activate    # Windows
# or
source .venv/bin/activate # Linux/Mac
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## 🏋️ Training

Trained YOLOv8s with augmentations designed for QR robustness (tilt, blur, occlusion, etc.).

To retrain:

```bash
python train.py
```

* Default settings (epochs=100, imgsz=640, batch=16) are already inside `train.py`.
* If you want to change these, **edit `train.py` directly**.
* Trained weights are provided at `src/models/best.pt` for reproducibility.

> ⚠️ Note: Training requires a GPU (tested on RTX 2060, ~2 hours for 100 epochs).

## Inference (Detection Only)

To run detection on the test images and generate `submission_detection_1.json`:

```bash
python infer.py --weights src/models/best.pt --input dataset/images/test_images --output outputs/submission_detection_1.json
```

This will output results in the required JSON format:

```json
[
  {"image_id": "img001", "qrs": [{"bbox": [x_min, y_min, x_max, y_max]}]},
  {"image_id": "img002", "qrs": [{"bbox": [x_min, y_min, x_max, y_max]}, {"bbox": [...]}]}
]
```

## Inference (Detection + Decoding)

To run detection + decoding and generate `submission_decoding_2.json`:

```bash
python infer_decode.py --model src/models/best.pt --input dataset/images/test_images --output outputs/submission_decoding_2.json
```

This will output results like:

```json
[
  {"image_id": "img001", "qrs": [
      {"bbox": [x_min, y_min, x_max, y_max], "value": "B12345"},
      {"bbox": [x_min, y_min, x_max, y_max], "value": "MFR56789"}
  ]}
]
```

> ⚠️ Note: Some QR codes may return `"unknown"` if decoding fails due to blur/occlusion.

## Key Notes

* **Trained model**: Provided at `src/models/best.pt`. No need to retrain for evaluation.
* **Dataset**: Must be downloaded separately by organizers. Paths in scripts assume the structure shown above.
* **Reproducibility**: All scripts are runnable with a single command. Evaluators only need to adjust the dataset path.
* **Rules compliance**: We do not use any external APIs; only open-source libraries (`ultralytics`, `opencv`).

## Authors

Hackathon submission by Akarsh Kumar Gowda.