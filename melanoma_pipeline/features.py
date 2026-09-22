"""Real, validated ABCD/structured-feature extraction -- reuses this
project's existing, frozen V5 pipeline (YOLO segmentation + the unchanged
Code/MelanomaDeterminingStuff formulas + pipeline_v5's 11 additional
features), exactly the same call sequence the live application's own
`MelanomaDetection/MelanomaDetection.Python/v5_detector.py::V5Detector.
process_image` uses -- not a reimplementation, and not the simpler
Otsu-threshold placeholder an earlier version of this file used.

Feature order and count are DERIVED from the real, live config object
(`pipeline_v5.decision_model.V5_CONFIG["features"]`), not hardcoded --
if V5's feature set ever changes, this module's output shape changes with
it automatically rather than silently going out of sync.

Design note vs. the earlier Otsu-based version of this file: that version
used `joblib.Parallel` (process-based) to fan pure-NumPy/OpenCV work across
CPU cores, which is safe because there was no shared heavy model state.
Real extraction here loads a YOLO/torch model, which should be loaded ONCE
and reused -- running several copies of it across parallel worker
processes (especially on a single GPU) would be wasteful at best and
resource-contending at worst, so `extract_features_for_dataframe` below
loads the model once and processes images sequentially instead of via
joblib. If this becomes a throughput bottleneck on the real 402k-image
dataset, batch YOLO inference (not naive multiprocessing) is the right next
optimization, not restoring joblib.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import config

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON_APP_DIR = REPO_ROOT / "MelanomaDetection" / "MelanomaDetection.Python"
for _p in (REPO_ROOT, PYTHON_APP_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Exact same imports v5_detector.py uses for its own live scoring path --
# see that module's docstring for why preprocess_image is called a second
# time here (it needs no_hair/circle_info, which process_image() computes
# internally but does not return).
from revised_abcd.pipeline_v2 import process_image as v5_process_image, preprocess_image as v5_preprocess_image  # noqa: E402
from pipeline_v5.decision_model import V5_CONFIG  # noqa: E402
from pipeline_v5.feature_extraction import extract_v5_new_features  # noqa: E402

YOLO_CONF = 0.25  # matches v5_detector.py's CONF and every prior V4/V5 evaluation in this project

ABCD_FEATURE_NAMES: list[str] = list(V5_CONFIG["features"])  # derived from the real, live config -- not hardcoded
N_ABCD_FEATURES: int = len(ABCD_FEATURE_NAMES)


def load_yolo_model():
    """Loads the same frozen YOLO checkpoint the live app uses, via the
    app's own `yolo_config.get_yolo_weights_path()` so the checkpoint path
    resolution (including its `YOLO_WEIGHTS_PATH` env var override) stays
    identical to production rather than being re-specified here. Load once,
    reuse for every image -- do not call this per-image."""
    from ultralytics import YOLO
    from yolo_config import get_yolo_weights_path
    return YOLO(get_yolo_weights_path())


def extract_abcd_features(image_path: str | Path, yolo_model) -> np.ndarray:
    """Extracts the real, validated `ABCD_FEATURE_NAMES`-ordered feature
    vector for one image, via the exact same call sequence as
    `v5_detector.py::V5Detector.process_image`'s live scoring path:
    YOLO segmentation -> primary (highest-confidence) instance's mask ->
    the unchanged base-6 ABCD features (A_value, B_circularity, C_value,
    D_px, confidence, lesion_fraction) -> pipeline_v5's 11 additional
    features on the same mask/preprocessed image.

    Returns an all-NaN vector (never raises) if no lesion is detected, the
    image can't be read, or any required feature comes back missing/NaN --
    matching v5_detector.py's own NO_DETECTION-equivalent handling rather
    than guessing or silently scoring a degenerate case. Callers
    (`extract_features_for_dataframe`) should check for NaN rows.
    """
    try:
        rows, _orig_shape = v5_process_image(yolo_model, str(image_path), conf=YOLO_CONF)
    except Exception:  # noqa: BLE001 -- one bad image must not abort a batch extraction run
        return np.full(N_ABCD_FEATURES, np.nan, dtype=np.float64)

    if not rows:
        return np.full(N_ABCD_FEATURES, np.nan, dtype=np.float64)

    primary = rows[0]  # highest-confidence instance -- same "primary lesion" convention as v5_detector.py
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
    if any(v is None or v != v for v in values):  # v != v is the NaN check (works without a numpy import per-value)
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
    print(f"Extracted {features.shape[1]} validated ABCD features (derived from the live "
          f"pipeline_v5.decision_model.V5_CONFIG) for {len(metadata)} images -> {out_features} "
          f"(shape {features.shape})")
    print(f"Feature order: {ABCD_FEATURE_NAMES}")
    print(f"Saved matching index -> {out_index}, feature names -> {out_names}")
    print(f"Rows with no detection / failed extraction (all-NaN): {n_failed}")
