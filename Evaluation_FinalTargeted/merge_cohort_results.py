"""Merge existing Evaluation_3500 V2 results + newly-computed cohort results
into one unified per-instance dataset covering the full 3482-image targeted
cohort. Read-only with respect to both source files; writes a new merged
CSV only.
"""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXISTING_RESULTS = ROOT / "Evaluation_3500" / "full_run" / "results.csv"
NEW_RESULTS = ROOT / "Evaluation_FinalTargeted" / "Cohort" / "new_v2_results" / "results.csv"
COHORT_CSV = ROOT / "Evaluation_FinalTargeted" / "Cohort" / "targeted_cohort_manifest.csv"
OUT_CSV = ROOT / "Evaluation_FinalTargeted" / "Cohort" / "cohort_v2_results.csv"


def image_id_from_name(name):
    return Path(name).stem


def main():
    cohort = list(csv.DictReader(open(COHORT_CSV, newline="", encoding="utf-8")))
    cohort_ids = {r["image_id"] for r in cohort}
    cohort_gt = {r["image_id"]: r["ground_truth"] for r in cohort}
    print(f"Cohort size: {len(cohort_ids)}")

    existing = list(csv.DictReader(open(EXISTING_RESULTS, newline="", encoding="utf-8")))
    new = list(csv.DictReader(open(NEW_RESULTS, newline="", encoding="utf-8")))
    print(f"Existing V2 results rows: {len(existing)}  New V2 results rows: {len(new)}")

    fieldnames = list(existing[0].keys())
    assert fieldnames == list(new[0].keys()), "schema mismatch between existing and new results!"

    merged = []
    seen_ids = set()
    mismatches = 0
    for r in existing:
        image_id = image_id_from_name(r["image_name"])
        if image_id not in cohort_ids:
            continue
        if r["ground_truth"] != cohort_gt[image_id]:
            mismatches += 1
        merged.append(r)
        seen_ids.add(image_id)

    for r in new:
        image_id = image_id_from_name(r["image_name"])
        if image_id not in cohort_ids:
            continue
        if r["ground_truth"] != cohort_gt[image_id]:
            mismatches += 1
        merged.append(r)
        seen_ids.add(image_id)

    missing = cohort_ids - seen_ids
    print(f"Ground-truth mismatches between results and cohort: {mismatches} (should be 0)")
    print(f"Cohort images missing from merged results: {len(missing)} (should be 0)")
    if missing:
        print("  examples:", list(missing)[:10])

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(merged)

    n_images = len(seen_ids)
    n_rows = len(merged)
    print(f"\nWrote {n_rows} rows covering {n_images} unique images to {OUT_CSV}")
    assert mismatches == 0, "STOP: ground-truth mismatch between cohort and V2 results — invalid merge"
    assert len(missing) == 0, "STOP: cohort images missing from merged results — incomplete merge"


if __name__ == "__main__":
    main()
