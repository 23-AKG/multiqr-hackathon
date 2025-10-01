# scripts/coco_to_yolo.py
import json, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "dataset" / "images" / "train_images"
COCO_JSON = ROOT / "dataset" / "merged_labels.json"
LBL_DIR  = ROOT / "dataset" / "labels" / "train_images"
SPLIT_DIR = ROOT / "dataset" / "splits"
DATASET_YAML = ROOT / "dataset" / "dataset.yaml"

VAL_FRACTION = 0.20

def to_yolo(b, W, H):
    """Convert COCO [x,y,w,h] to YOLO [cx,cy,w,h] normalized."""
    x, y, w, h = b
    cx, cy = (x + w/2)/W, (y + h/2)/H
    return cx, cy, w/W, h/H

def main():
    LBL_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    coco = json.load(open(COCO_JSON, "r", encoding="utf-8"))
    images = {im["id"]: im for im in coco["images"]}

    anns_by_img = {}
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    ids = sorted(images.keys())
    random.seed(42)
    random.shuffle(ids)
    val_n = max(1, int(len(ids) * VAL_FRACTION))
    val_ids = set(ids[:val_n])
    train_ids = set(ids[val_n:])

    # Write YOLO label files for ALL images
    for img_id, im in images.items():
        fname = Path(im["file_name"]).with_suffix(".txt").name
        lp = LBL_DIR / fname
        with open(lp, "w", encoding="utf-8") as f:
            for ann in anns_by_img.get(img_id, []):
                cx, cy, ww, hh = to_yolo(ann["bbox"], im["width"], im["height"])
                f.write(f"0 {cx:.6f} {cy:.6f} {ww:.6f} {hh:.6f}\n")

    # Correct relative paths: always start with dataset/
    def rel_img_path(im):
        return f"dataset/images/train_images/{im['file_name']}"

    with open(SPLIT_DIR / "train.txt", "w", encoding="utf-8") as ft:
        for i in sorted(train_ids):
            ft.write(rel_img_path(images[i]) + "\n")

    with open(SPLIT_DIR / "val.txt", "w", encoding="utf-8") as fv:
        for i in sorted(val_ids):
            fv.write(rel_img_path(images[i]) + "\n")

    # Write dataset.yaml
    DATASET_YAML.write_text(
        "path: dataset\n"
        "train: splits/train.txt\n"
        "val: splits/val.txt\n"
        "names:\n"
        "  0: qrcode\n",
        encoding="utf-8"
    )
    print("✅ Wrote labels, splits, and dataset.yaml")

if __name__ == "__main__":
    main()
