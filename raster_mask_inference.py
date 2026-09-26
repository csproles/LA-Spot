"""Raster mask inference for the trained YOLO lesion-segmentation model.

Runs images through the trained Ultralytics YOLO segmentation checkpoint
(best.pt) and extracts the RASTER segmentation masks that Ultralytics
generates internally at inference time via ``result.masks.data``.

This script intentionally does NOT use polygon outputs. It never touches:
    result.masks.xy
    result.masks.xyn
    cv2.fillPoly()
    polygon annotation coordinates

``result.masks.data`` is a torch.Tensor of shape (num_instances, H, W)
produced fresh by the model for each prediction -- it is not a file that
exists anywhere on disk beforehand.

Usage:
    python raster_mask_inference.py
    python raster_mask_inference.py --source test_images --output raster_masks
    python raster_mask_inference.py --model "path\\to\\best.pt" --conf 0.25
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

DEFAULT_MODEL = (
    r"C:\Users\sirjanaa\Downloads\runs\runs\segment\melanoma_yolo26n_seg\weights\best.pt"
)
DEFAULT_SOURCE = "test_images"
DEFAULT_OUTPUT = "raster_masks"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def gather_images(source: Path):
    """Return a sorted list of image file paths under ``source``."""
    if source.is_file():
        return [source]
    return sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def raster_mask_from_result(result, debug=False):
    """Extract the combined raster mask (float, H x W) from one Results object.

    Uses ONLY ``result.masks.data`` (the raster tensor Ultralytics produces
    at runtime). Returns None if no mask was predicted for this image.
    """
    has_masks = result.masks is not None
    if debug:
        print(f"    result.masks exists: {has_masks}")

    if not has_masks:
        return None

    masks_data = result.masks.data  # torch.Tensor, shape (N, mh, mw)

    if debug:
        print(f"    result.masks.data.shape: {tuple(masks_data.shape)}")
        print(f"    number of predicted instances: {masks_data.shape[0]}")

    if masks_data.shape[0] == 0:
        return None

    # Convert the raster tensor to NumPy.
    masks_np = masks_data.detach().cpu().numpy()  # (N, mh, mw)

    if masks_np.shape[0] == 1:
        combined = masks_np[0]
    else:
        # Multiple lesion instances predicted for this image: combine the
        # raster masks with a per-pixel max (equivalent to logical OR once
        # thresholded).
        combined = np.max(masks_np, axis=0)

    return combined


def to_binary_mask(raster_mask, target_shape):
    """Threshold a float raster mask to {0, 255} and match target (H, W).

    Resizing (if needed) uses nearest-neighbor interpolation so the output
    stays strictly binary.
    """
    target_h, target_w = target_shape[:2]

    binary = (raster_mask > 0.5).astype(np.uint8) * 255

    if binary.shape[0] != target_h or binary.shape[1] != target_w:
        binary = cv2.resize(
            binary, (target_w, target_h), interpolation=cv2.INTER_NEAREST
        )
        binary = (binary > 127).astype(np.uint8) * 255

    return binary


def run_inference(model_path, source_dir, output_dir, conf, debug_count):
    model_path = Path(model_path)
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if not source_dir.exists():
        raise FileNotFoundError(f"Source folder not found: {source_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = gather_images(source_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {source_dir}")

    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))

    print(f"Running inference on {len(image_paths)} image(s) from: {source_dir}\n")

    results = model.predict(
        source=str(source_dir),
        retina_masks=True,
        conf=conf,
        verbose=False,
    )

    n_no_mask = 0
    n_saved = 0

    for idx, result in enumerate(results):
        image_path = Path(result.path)
        debug = idx < debug_count

        source_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if source_image is None:
            print(f"[WARN] Could not read source image, skipping: {image_path}")
            continue
        src_shape = source_image.shape

        if debug:
            print(f"--- [{idx + 1}/{len(image_paths)}] {image_path.name} ---")
            print(f"    source image shape: {src_shape}")

        try:
            raster_mask = raster_mask_from_result(result, debug=debug)
        except Exception as exc:  # defensive: never crash the whole batch
            print(f"[WARN] Failed to extract mask for {image_path.name}: {exc}")
            raster_mask = None

        if raster_mask is None:
            if debug:
                print("    -> no lesion mask predicted; saving all-black mask")
            binary_mask = np.zeros(src_shape[:2], dtype=np.uint8)
            n_no_mask += 1
        else:
            binary_mask = to_binary_mask(raster_mask, src_shape)

        out_name = f"{image_path.stem}_mask.png"
        out_path = output_dir / out_name
        cv2.imwrite(str(out_path), binary_mask)
        n_saved += 1

        if debug:
            print(f"    saved mask shape: {binary_mask.shape}")
            print(f"    unique pixel values in saved mask: {np.unique(binary_mask)}")
            print(f"    saved to: {out_path}\n")

    print(f"Done. Saved {n_saved} mask(s) to '{output_dir}'.")
    print(f"Images with no predicted mask (all-black mask saved): {n_no_mask}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run YOLO segmentation inference and export raster binary lesion "
            "masks from result.masks.data (no polygon reconstruction)."
        )
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Path to the trained YOLO segmentation checkpoint (best.pt).",
    )
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help="Folder of input images (default: test_images/).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Folder to save binary raster masks (default: raster_masks/).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for predictions (default: 0.25).",
    )
    parser.add_argument(
        "--debug-count",
        type=int,
        default=5,
        help="Number of images to print detailed debug info for (default: 5).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args.model, args.source, args.output, args.conf, args.debug_count)
