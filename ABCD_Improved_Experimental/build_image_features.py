"""Phase 1 (color half) + Phase 3 -- the IMAGE pass.

Loads each of the 2090 dev-cohort original images ONCE, runs the existing,
unmodified preprocessing chain (remove_vignette -> remove_salt_pepper_noise
-> apply_bilateral_filter -> remove_hair, same order/params as
revised_abcd/pipeline_v2.py and Evaluation_FeatureEngineering/extract_features.py),
and computes every feature that needs the actual image from that single
loaded/preprocessed image -- never reloading an image twice. This is the
expensive part of the task (~0.75 img/s per the task brief's prior
measurement); the mask-only pass (build_mask_features.py) is separate and
much cheaper.

READ-ONLY with respect to pipeline_v5/, revised_abcd/, Code/, and every
existing Evaluation_* directory. Never reads the locked test set -- only
Evaluation_FinalTargeted/Cohort/development_manifest.csv (dev cohort) and
Evaluation_FinalTargeted/Cohort/cohort_v2_results.csv (mask paths) are used.
"""

import contextlib
import io
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "Code"))
sys.path.insert(0, str(ROOT / "Evaluation_FinalTargeted"))

import common_features as cf  # noqa: E402
import metrics_lib as ml  # noqa: E402  (existing, unmodified helper module)
from HandlingStuff import load_image  # noqa: E402
from ComputerVisionStuff import (  # noqa: E402
    remove_vignette, remove_salt_pepper_noise, apply_bilateral_filter, remove_hair,
)

EXT_TABLE = ROOT / "ABCD_Audit_V5" / "_source_from_research_branch" / "Evaluation_FeatureEngineering" / "extended_feature_table.csv"
DEV_MANIFEST = ROOT / "Evaluation_FinalTargeted" / "Cohort" / "development_manifest.csv"
COHORT_RESULTS = ROOT / "Evaluation_FinalTargeted" / "Cohort" / "cohort_v2_results.csv"
OUT_DIR = Path(__file__).resolve().parent
OUT_CSV = OUT_DIR / "_image_features.csv"

NEW_KEYS = [
    "C_value_pc", "red_fraction_pc", "bluegray_fraction_pc", "dark_fraction_pc",
    "color_entropy_pc", "lab_a_std_pc", "lab_b_std_pc", "skin_contrast_pc",
    "entropy_L", "entropy_a", "entropy_b",
]


def build_image_id_to_mask_path():
    cohort = pd.read_csv(COHORT_RESULTS, encoding="utf-8")
    cohort = cohort[cohort["evaluation_status"] == "SINGLE_LESION_EVALUABLE"].copy()
    cohort["image_id"] = cohort["image_name"].apply(lambda n: Path(n).stem)
    cohort = cohort.drop_duplicates(subset=["image_id"], keep="first")
    return dict(zip(cohort["image_id"], cohort["mask_path"]))


def process_one(image_id, image_path, mask_path):
    result = {"image_id": image_id}
    errors = []

    mask_raw = cv2.imread(str(ROOT / mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_raw is None:
        for k in NEW_KEYS:
            result[k] = np.nan
        result["_errors"] = f"mask_unreadable:{mask_path}"
        return result

    mask = (mask_raw > 127).astype(np.uint8) * 255

    stdout_buf = io.StringIO()
    no_hair = None
    mask_use = mask
    circle_info = None
    try:
        with contextlib.redirect_stdout(stdout_buf), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            original = load_image(str(image_path))
            no_vignette, circle_info = remove_vignette(original)
            denoised = remove_salt_pepper_noise(no_vignette, kernel_size=3)
            bilateral = apply_bilateral_filter(denoised, diameter=9, sigma_color=75, sigma_space=75)
            no_hair = remove_hair(bilateral, kernel_size=17, threshold=10)
            if no_hair.shape[:2] != mask.shape:
                mask_use = cv2.resize(mask, (no_hair.shape[1], no_hair.shape[0]), interpolation=cv2.INTER_NEAREST)
    except Exception as e:  # noqa: BLE001
        errors.append(f"preproc:{e!r}")
        no_hair = None

    if no_hair is not None:
        primary_mask, _ = cf.select_primary_component(mask_use)

        try:
            with contextlib.redirect_stdout(stdout_buf), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result.update(cf.score_color_pc(primary_mask, no_hair, circle_info))
        except Exception as e:  # noqa: BLE001
            for k in ["C_value_pc", "red_fraction_pc", "bluegray_fraction_pc", "dark_fraction_pc"]:
                result[k] = np.nan
            errors.append(f"score_color_pc:{e!r}")

        try:
            result.update(cf.color_stats_pc(no_hair, primary_mask))
        except Exception as e:  # noqa: BLE001
            for k in ["color_entropy_pc", "lab_a_std_pc", "lab_b_std_pc"]:
                result[k] = np.nan
            errors.append(f"color_stats_pc:{e!r}")

        try:
            result["skin_contrast_pc"] = cf.skin_contrast_pc(no_hair, primary_mask)
        except Exception as e:  # noqa: BLE001
            result["skin_contrast_pc"] = np.nan
            errors.append(f"skin_contrast_pc:{e!r}")

        try:
            result.update(cf.channel_entropy_lab(no_hair, primary_mask))
        except Exception as e:  # noqa: BLE001
            for k in ["entropy_L", "entropy_a", "entropy_b"]:
                result[k] = np.nan
            errors.append(f"channel_entropy_lab:{e!r}")
    else:
        for k in NEW_KEYS:
            result[k] = np.nan

    result["_errors"] = "; ".join(errors)
    return result


def main(limit=None, start=0):
    ext = pd.read_csv(EXT_TABLE)
    dev_ids = list(ext["image_id"])

    manifest = pd.read_csv(DEV_MANIFEST)
    path_by_id = dict(zip(manifest["image_id"], manifest["image_path"]))

    id_to_mask = build_image_id_to_mask_path()

    if start:
        dev_ids = dev_ids[start:]
    if limit:
        dev_ids = dev_ids[:limit]

    rows = []
    n_errors = 0
    t0 = time.time()
    for i, iid in enumerate(dev_ids):
        img_path = path_by_id.get(iid)
        mask_path = id_to_mask.get(iid)
        if img_path is None or mask_path is None:
            rec = {"image_id": iid, "_errors": "missing_image_path_or_mask_path"}
            for k in NEW_KEYS:
                rec[k] = np.nan
            rows.append(rec)
            n_errors += 1
            continue

        rec = process_one(iid, img_path, mask_path)
        if rec.get("_errors"):
            n_errors += 1
        rows.append(rec)

        if (i + 1) % 100 == 0 or i == len(dev_ids) - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(dev_ids) - (i + 1)) / rate if rate > 0 else float("nan")
            print(f"[image pass {i+1}/{len(dev_ids)}] {iid}  elapsed={elapsed:.0f}s  "
                  f"rate={rate:.2f} img/s  eta={eta:.0f}s  errors_so_far={n_errors}", flush=True)
            pd.DataFrame(rows).to_csv(OUT_CSV, index=False)  # checkpoint

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(df)} rows to {OUT_CSV}")
    print(f"Rows with at least one error: {n_errors}")
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()
    main(limit=args.limit, start=args.start)
