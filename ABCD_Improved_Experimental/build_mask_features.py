"""Phase 1 (shape half) + Phase 5 + Phase 7 -- the MASK-ONLY pass.

Everything computed here only needs the cached mask PNG for each of the
2090 dev-cohort images, never the original image -- cheap, per the task's
efficiency note. Produces ABCD_Improved_Experimental/_mask_features.csv,
later merged with the (slower) image-pass output and the original
extended_feature_table.csv columns in build_dev_feature_table_v2.py.

READ-ONLY with respect to pipeline_v5/, revised_abcd/, Code/, and every
existing Evaluation_* directory. Never reads the locked test set.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "Evaluation_FinalTargeted"))

import common_features as cf  # noqa: E402
import metrics_lib as ml  # noqa: E402  (existing, unmodified helper module)

EXT_TABLE = ROOT / "ABCD_Audit_V5" / "_source_from_research_branch" / "Evaluation_FeatureEngineering" / "extended_feature_table.csv"
COHORT_RESULTS = ROOT / "Evaluation_FinalTargeted" / "Cohort" / "cohort_v2_results.csv"
OUT_DIR = Path(__file__).resolve().parent
OUT_CSV = OUT_DIR / "_mask_features.csv"


def build_image_id_to_mask_path():
    cohort = pd.read_csv(COHORT_RESULTS, encoding="utf-8")
    cohort = cohort[cohort["evaluation_status"] == "SINGLE_LESION_EVALUABLE"].copy()
    cohort["image_id"] = cohort["image_name"].apply(lambda n: Path(n).stem)
    # one row per image_id expected for SINGLE_LESION_EVALUABLE; guard anyway
    cohort = cohort.drop_duplicates(subset=["image_id"], keep="first")
    return dict(zip(cohort["image_id"], cohort["mask_path"]))


def process_one(mask_path):
    mask_raw = cv2.imread(str(ROOT / mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_raw is None:
        return None, f"mask_unreadable:{mask_path}"
    mask = (mask_raw > 127).astype(np.uint8) * 255
    return mask, None


def main(limit=None):
    ext = pd.read_csv(EXT_TABLE)
    dev_ids = list(ext["image_id"])
    very_small_floor_px = float(np.percentile(ext["lesion_area_px"].dropna(), 1))
    print(f"very_small_contour floor (1st percentile of lesion_area_px, n={ext['lesion_area_px'].notna().sum()}): "
          f"{very_small_floor_px:.1f} px")

    id_to_mask = build_image_id_to_mask_path()
    missing_mask = [iid for iid in dev_ids if iid not in id_to_mask]
    if missing_mask:
        print(f"WARNING: {len(missing_mask)} image_ids from extended_feature_table.csv have no "
              f"SINGLE_LESION_EVALUABLE mask_path in cohort_v2_results.csv: {missing_mask[:10]}...")

    if limit:
        dev_ids = dev_ids[:limit]

    rows = []
    errors = []
    t0 = time.time()
    for i, iid in enumerate(dev_ids):
        rec = {"image_id": iid}
        mask_path = id_to_mask.get(iid)
        if mask_path is None:
            rec["_mask_error"] = "no_mask_path_matched"
            rows.append(rec)
            errors.append((iid, "no_mask_path_matched"))
            continue

        mask, err = process_one(mask_path)
        if err:
            rec["_mask_error"] = err
            rows.append(rec)
            errors.append((iid, err))
            continue

        try:
            primary_mask, primary_stats = cf.select_primary_component(mask)
            shape = cf.shape_features_pc(primary_mask)
            asym = cf.fold_asymmetry_hv(primary_mask)
            fractal_dim, fractal_n = cf.border_fractal_dimension(primary_mask)
            flags = cf.compute_quality_flags(mask, primary_mask, primary_stats, very_small_floor_px)

            rec.update(shape)
            rec.update(asym)
            rec["border_fractal_dimension"] = fractal_dim
            rec["border_fractal_dimension_n_box_sizes"] = fractal_n
            rec.update(flags)
            rec["primary_component_area_px"] = float(np.sum(primary_mask > 0))
            rec["mask_h"] = mask.shape[0]
            rec["mask_w"] = mask.shape[1]
            rec["_mask_error"] = ""
        except Exception as e:  # noqa: BLE001
            rec["_mask_error"] = f"compute:{e!r}"
            errors.append((iid, f"compute:{e!r}"))
        rows.append(rec)

        if (i + 1) % 100 == 0 or i == len(dev_ids) - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"[mask pass {i+1}/{len(dev_ids)}] {iid}  elapsed={elapsed:.0f}s  rate={rate:.2f} img/s")
            pd.DataFrame(rows).to_csv(OUT_CSV, index=False)  # checkpoint

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(df)} rows to {OUT_CSV}")
    if errors:
        print(f"{len(errors)} rows had a mask-pass error:")
        for iid, err in errors[:30]:
            print(f"  {iid}: {err}")
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    main(limit=args.limit)
