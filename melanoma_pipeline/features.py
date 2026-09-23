"""Real, validated ABCD/structured-feature extraction -- reuses this
project's existing, frozen V5 pipeline logic (YOLO segmentation + the
unchanged legacy border/color formulas + the 11 additional V5 features),
consolidated into the standalone `abcd/` package (see abcd/__init__.py)
so this module runs independently of the rest of the original repository.
Not a reimplementation, and not the simpler Otsu-threshold placeholder an
earlier version of this file used.

Feature order and count are DERIVED from `abcd.feature_config.
V5_ALL_FEATURES`, not hardcoded -- if that list ever changes, this
module's output shape changes with it automatically rather than silently
going out of sync.

Design note: this module loads a YOLO/torch model once and processes
images SEQUENTIALLY, not via joblib multiprocessing -- running several
copies of a loaded neural net across parallel worker processes (especially
on a single GPU) would be wasteful at best and resource-contending at
worst. If this becomes a throughput bottleneck on the full dataset, batch
YOLO inference is the right next optimization, not multiprocessing.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import config
from abcd.feature_config import V5_ALL_FEATURES
from abcd.pipeline import process_image as v5_process_image, preprocess_image as v5_preprocess_image
from abcd.new_features import extract_v5_new_features

YOLO_CONF = 0.25  # matches every prior evaluation of this feature set in the original project

ABCD_FEATURE_NAMES: list[str] = list(V5_ALL_FEATURES)  # derived from the real, versioned config -- not hardcoded
N_ABCD_FEATURES: int = len(ABCD_FEATURE_NAMES)


def load_yolo_model():
    """Loads the YOLO checkpoint via config.get_yolo_weights_path() (which
    raises a clear, actionable error if the checkpoint hasn't been placed
    yet -- see README.md's "External assets" section). Load once, reuse
    for every image -- do not call this per-image."""
    from ultralytics import YOLO
    return YOLO(str(config.get_yolo_weights_path()))


def extract_abcd_features(image_path: str | Path, yolo_model) -> np.ndarray:
    """Extracts the real, validated `ABCD_FEATURE_NAMES`-ordered feature
    vector for one image: YOLO segmentation -> primary (highest-confidence)
    instance's mask -> the unchanged base-6 ABCD features (A_value,
    B_circularity, C_value, D_px, confidence, lesion_fraction) -> the 11
    additional V5 features on the same mask/preprocessed image.

    Returns an all-NaN vector (never raises) if no lesion is detected, the
    image can't be read, or any required feature comes back missing/NaN.
    Callers (`extract_features_for_dataframe`) should check for NaN rows.
    """
    try:
        rows, _orig_shape = v5_process_image(yolo_model, str(image_path), conf=YOLO_CONF)
    except Exception:  # noqa: BLE001 -- one bad image must not abort a batch extraction run
        return np.full(N_ABCD_FEATURES, np.nan, dtype=np.float64)

    if not rows:
        return np.full(N_ABCD_FEATURES, np.nan, dtype=np.float64)

    primary = rows[0]  # highest-confidence instance -- the "primary lesion" convention this feature set uses
    mask = primary["mask"]

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        return np.full(N_ABCD_FEATURES, np.nan, dtype=np.float64)

    no_hair, circle_info = v5_preprocess_image(image_bgr)
    new_features = extract_v5_new_features(mask, no_hair, circle_info, primary["D_px"])

    feature_row = {
        "A_value": primary["A_value"], "B_circularity": primary["B_circularity"],
        "C_value": primary["C_value"], "D_px": primary["D_px"],
        "confidence": primary["confidence"], "lesion_fraction": primary["lesion_fraction"],
        **new_features,
    }

    values = [feature_row.get(name) for name in ABCD_FEATURE_NAMES]
    if any(v is None or v != v for v in values):  # v != v is the NaN check
        return np.full(N_ABCD_FEATURES, np.nan, dtype=np.float64)
    return np.array(values, dtype=np.float64)


def extract_features_for_dataframe(
    df: pd.DataFrame,
    image_dir: str | Path,
    image_id_col: str = "isic_id",
    yolo_model=None,
) -> np.ndarray:
    """Runs `extract_abcd_features` over every row of `df`, sequentially,
    reusing one loaded YOLO model (loaded once here if `yolo_model` is not
    supplied). Returns an (N, N_ABCD_FEATURES) array in the same row order
    as `df` -- callers needing to align features back to image ids should
    zip against `df[image_id_col]` (or use the paired `abcd_index.npy` this
    module's __main__ block writes)."""
    image_dir = Path(image_dir)
    image_ids = df[image_id_col].astype(str).tolist()

    def _resolve(image_id: str) -> Path:
        candidate = image_dir / image_id
        if candidate.suffix:
            return candidate
        for ext in (".jpg", ".jpeg", ".png"):
            p = image_dir / f"{image_id}{ext}"
            if p.exists():
                return p
        return image_dir / f"{image_id}.jpg"

    model = yolo_model or load_yolo_model()
    results = [
        extract_abcd_features(_resolve(image_id), model)
        for image_id in tqdm(image_ids, desc="Extracting validated ABCD features (YOLO + V5 formulas)")
    ]
    return np.vstack(results)


if __name__ == "__main__":
    config.ensure_dirs()

    metadata = pd.read_csv(config.METADATA_CSV)
    features = extract_features_for_dataframe(metadata, image_dir=config.IMAGE_DIR)

    out_features = config.OUTPUT_DIR / "abcd_features.npy"
    out_index = config.OUTPUT_DIR / "abcd_index.npy"
    out_names = config.OUTPUT_DIR / "abcd_feature_names.npy"
    np.save(out_features, features)
    np.save(out_index, metadata["isic_id"].astype(str).to_numpy())
    np.save(out_names, np.array(ABCD_FEATURE_NAMES))

    n_failed = int(np.isnan(features).any(axis=1).sum())
    print(f"Extracted {features.shape[1]} validated ABCD features (derived from "
          f"abcd.feature_config.V5_ALL_FEATURES) for {len(metadata)} images -> {out_features} "
          f"(shape {features.shape})")
    print(f"Feature order: {ABCD_FEATURE_NAMES}")
    print(f"Saved matching index -> {out_index}, feature names -> {out_names}")
    print(f"Rows with no detection / failed extraction (all-NaN): {n_failed}")
