"""Build representative visual review panels: Original | Official Ground
Truth | LAB/Otsu | YOLO, for interpretation only (not used to tune anything).
"""

import csv
from pathlib import Path

import cv2
import numpy as np

BENCH_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BENCH_DIR / "images"
GT_DIR = BENCH_DIR / "ground_truth"
OTSU_MASK_DIR = BENCH_DIR / "otsu_masks"
YOLO_MASK_DIR = BENCH_DIR / "yolo_masks"
OUT_DIR = BENCH_DIR / "visual_panels"
OUT_DIR.mkdir(exist_ok=True)

PANEL_W = 380  # each of the 4 tiles is resized to this width for the composite

SELECTIONS = {
    "yolo_substantially_better": ["ISIC_0036212", "ISIC_0023752"],
    "otsu_substantially_better": ["ISIC_0023269", "ISIC_0021596"],
    "both_good": ["ISIC_0016438", "ISIC_0015381"],
    "both_poor": ["ISIC_0022219", "ISIC_0021379"],
}


def load_tile(path, is_mask=False):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE if is_mask else cv2.IMREAD_COLOR)
    if is_mask:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    scale = PANEL_W / w
    return cv2.resize(img, (PANEL_W, int(h * scale)))


def label_tile(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def build_panel(image_id, metrics_by_id):
    orig = load_tile(IMAGES_DIR / f"{image_id}.jpg")
    gt = load_tile(GT_DIR / f"{image_id}_gt.png", is_mask=True)
    otsu = load_tile(OTSU_MASK_DIR / f"{image_id}_otsu.png", is_mask=True)
    yolo = load_tile(YOLO_MASK_DIR / f"{image_id}_yolo.png", is_mask=True)

    m = metrics_by_id[image_id]
    tiles = [
        label_tile(orig, f"{image_id} - Original"),
        label_tile(gt, "Official ISIC-2018 GT"),
        label_tile(otsu, f"LAB/Otsu  IoU={m['otsu_iou']}  Dice={m['otsu_dice']}"),
        label_tile(yolo, f"YOLO  IoU={m['yolo_iou']}  Dice={m['yolo_dice']}  n_inst={m['yolo_num_instances']}"),
    ]
    max_h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, max_h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(40, 40, 40))
             for t in tiles]
    return np.hstack(tiles)


def main():
    rows = list(csv.DictReader(open(BENCH_DIR / "segmentation_metrics_per_image.csv", newline="", encoding="utf-8")))
    metrics_by_id = {r["image_id"]: r for r in rows}

    for category, ids in SELECTIONS.items():
        panels = [build_panel(iid, metrics_by_id) for iid in ids]
        max_w = max(p.shape[1] for p in panels)
        panels = [cv2.copyMakeBorder(p, 4, 4, 0, max_w - p.shape[1], cv2.BORDER_CONSTANT, value=(200, 200, 200))
                  for p in panels]
        composite = np.vstack(panels)
        out_path = OUT_DIR / f"{category}.png"
        cv2.imwrite(str(out_path), composite)
        print(f"wrote {out_path} ({', '.join(ids)})")


if __name__ == "__main__":
    main()
