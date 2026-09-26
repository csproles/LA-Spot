"""Export raster binary and boundary masks from the trained YOLO segmenter."""

from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
IMAGE_ROOT = ROOT / "Images"
CHECKPOINT = Path(
    r"C:\Users\sirjanaa\Downloads\runs\runs\segment\melanoma_yolo26n_seg\weights\best.pt"
)
OUTPUT_ROOT = CHECKPOINT.parent.parent / "final_masks"


def export_masks() -> int:
    binary_dir = OUTPUT_ROOT / "binary"
    boundary_dir = OUTPUT_ROOT / "boundary"
    binary_dir.mkdir(parents=True, exist_ok=True)
    boundary_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(CHECKPOINT))
    image_paths = sorted(
        path
        for path in IMAGE_ROOT.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    exported = 0
    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue

        result = model.predict(
            source=str(image_path),
            imgsz=640,
            conf=0.25,
            verbose=False,
        )[0]
        binary = np.zeros(image.shape[:2], dtype=np.uint8)

        if result.masks is not None:
            for polygon in result.masks.xy:
                points = np.round(polygon).astype(np.int32)
                cv2.fillPoly(binary, [points], 255)

        boundary = cv2.morphologyEx(
            binary,
            cv2.MORPH_GRADIENT,
            np.ones((3, 3), dtype=np.uint8),
        )
        output_name = f"{image_path.stem}.png"
        cv2.imwrite(str(binary_dir / output_name), binary)
        cv2.imwrite(str(boundary_dir / output_name), boundary)
        exported += 1

    print(f"exported={exported}")
    print(f"binary_dir={binary_dir}")
    print(f"boundary_dir={boundary_dir}")
    return exported


if __name__ == "__main__":
    raise SystemExit(0 if export_masks() else 1)