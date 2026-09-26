"""Final merge step: combine
  - extended_feature_table.csv (2090 rows, all existing V4/V5/ablation columns, carried through unchanged)
  - _mask_features.csv (this task's mask-only pass: Phase 1 shape *_pc, Phase 4 asymmetry_h/v,
    Phase 5 border_fractal_dimension, Phase 7 quality flags)
  - _image_features.csv (this task's image pass: Phase 1 color *_pc, Phase 3 entropy_L/a/b)

into ABCD_Improved_Experimental/dev_feature_table_v2.csv, one row per of the
2090 development-cohort images. Every image_id from extended_feature_table.csv
gets a row; if a mask/image pass genuinely failed for some image, the
original columns are still populated and the new columns are NaN (never
drop the row).

READ-ONLY with respect to pipeline_v5/, revised_abcd/, Code/, and every
existing Evaluation_* directory. Never reads the locked test set.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EXT_TABLE = ROOT / "ABCD_Audit_V5" / "_source_from_research_branch" / "Evaluation_FeatureEngineering" / "extended_feature_table.csv"
OUT_DIR = Path(__file__).resolve().parent
MASK_CSV = OUT_DIR / "_mask_features.csv"
IMAGE_CSV = OUT_DIR / "_image_features.csv"
OUT_CSV = OUT_DIR / "dev_feature_table_v2.csv"


def main():
    ext = pd.read_csv(EXT_TABLE)
    mask_df = pd.read_csv(MASK_CSV)
    image_df = pd.read_csv(IMAGE_CSV)

    assert ext["image_id"].is_unique, "extended_feature_table.csv image_id not unique"

    merged = ext.merge(mask_df, on="image_id", how="left", validate="one_to_one")
    merged = merged.merge(image_df, on="image_id", how="left", validate="one_to_one", suffixes=("", "_imgpass"))

    # Phase 4: verified major/minor-axis aliases (see verify_axis_convention.py /
    # improved_feature_definitions.md for the empirical verification).
    #   asymmetry_h (np.fliplr, array axis=1) == asymmetry_major_axis
    #   asymmetry_v (np.flipud, array axis=0) == asymmetry_minor_axis
    merged["asymmetry_major_axis"] = merged["asymmetry_h"]
    merged["asymmetry_minor_axis"] = merged["asymmetry_v"]

    missing_mask = merged["_mask_error"].isna() & merged["A_value_pc"].isna()
    n_missing_mask_feats = int((merged["A_value_pc"].isna()).sum())
    n_missing_image_feats = int((merged["C_value_pc"].isna()).sum())
    n_mask_errors = int((merged["_mask_error"].fillna("") != "").sum())
    n_image_errors = int((merged["_errors"].fillna("") != "").sum())

    merged.to_csv(OUT_CSV, index=False)

    print(f"Wrote {len(merged)} rows, {len(merged.columns)} columns to {OUT_CSV}")
    print(f"Rows missing mask-pass shape features (A_value_pc NaN): {n_missing_mask_feats}")
    print(f"Rows missing image-pass color features (C_value_pc NaN): {n_missing_image_feats}")
    print(f"Rows with a non-empty _mask_error: {n_mask_errors}")
    print(f"Rows with a non-empty image-pass _errors: {n_image_errors}")
    print("\nColumn list:")
    for c in merged.columns:
        print(f"  {c}")

    print("\nQuality flag counts:")
    for col in ["flag_fragmented", "flag_multiple_substantial_components", "flag_edge_adjacent",
                "flag_very_small_contour", "flag_oversized_mask"]:
        if col in merged.columns:
            print(f"  {col}: {int(merged[col].fillna(False).astype(bool).sum())} / {len(merged)}")
    if "quality_flags" in merged.columns:
        print("\nquality_flags value counts (top 15):")
        print(merged["quality_flags"].fillna("MISSING").value_counts().head(15).to_string())

    return merged


if __name__ == "__main__":
    main()
