# train.py
from ultralytics import YOLO

def main():
    # Use YOLOv8 small model (more powerful than nano)
    model = YOLO("yolov8s.pt")

    model.train(
        data="dataset/dataset.yaml",  # points to our dataset.yaml
        epochs=100,
        batch=16,
        imgsz=640,
        name="qr_augmented",
        project="runs/detect",
        device=0,     # use GPU
        deterministic=True,
        seed=42,

        # Augmentations for QR robustness
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,   # lighting / color changes
        degrees=15,                          # rotation (simulate tilted scans)
        translate=0.1,                       # slight shifts
        scale=0.5,                           # zoom in/out
        shear=5,                             # shear distortions
        perspective=0.0005,                  # camera angle
        flipud=0.0,                          # no vertical flip (QRs rarely upside-down in packs)
        fliplr=0.5,                          # horizontal flip (possible rotation)
        mosaic=1.0,                          # mosaic augmentation
        mixup=0.1,                           # slight blending of images
        copy_paste=0.1                       # simulate occlusions
    )

if __name__ == "__main__":
    main()
