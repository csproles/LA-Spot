"""Builds the fixed, reproducible 3,500-image evaluation manifest from the
downloaded full ISIC dataset (Downloads/ISIC-images/).

Read-only with respect to the source dataset: never deletes, renames,
moves, or modifies anything under the ISIC-images folder. The manifest
references the ORIGINAL image paths directly — images are never copied.

Sampling method:
  1. Load metadata.csv.
  2. Keep only rows whose diagnosis_1 is exactly "Benign" or "Malignant"
     (any other/blank value would be treated as indeterminate/unlabeled
     and excluded — none exist in this download, but the filter is applied
     unconditionally rather than assumed).
  3. Drop duplicate isic_id rows (keep first occurrence) — defensive; none
     were found in this dataset (verified separately: 8277 unique ids for
     8277 rows).
  4. Split into benign/malignant pools (mutually exclusive by construction,
     since diagnosis_1 is a single categorical field per row).
  5. Deterministically sample 1750 from each pool with random.Random(SEED),
     SEED fixed and reported below.
  6. Write the manifest CSV. This file, once written, IS the evaluation
     set — it is never regenerated or resampled by this script once it
     exists (rerunning requires deleting it first, which is a manual,
     deliberate action, not a rerun default).
"""

import csv
import random
from pathlib import Path

SEED = 20260918  # fixed seed, reported in every output of this run

SOURCE_DIR = Path(r"C:\Users\sirjanaa\Downloads\ISIC-images")
SOURCE_METADATA = SOURCE_DIR / "metadata.csv"

OUTPUT_DIR = Path(__file__).resolve().parent / "Evaluation_3500"
MANIFEST_PATH = OUTPUT_DIR / "evaluation_manifest_3500.csv"

N_PER_CLASS = 1750

MANIFEST_FIELDNAMES = [
    "image_id", "image_path", "ground_truth", "ground_truth_binary",
    "diagnosis_1", "diagnosis_2", "diagnosis_3", "diagnosis_confirm_type",
    "lesion_id", "clin_size_long_diam_mm", "attribution", "copyright_license",
    "anatom_site_1", "age_approx", "sex",
]


def load_metadata_rows():
    with open(SOURCE_METADATA, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    if MANIFEST_PATH.exists():
        raise SystemExit(
            f"Refusing to overwrite existing manifest: {MANIFEST_PATH}\n"
            f"This evaluation set is fixed once created. Delete the file "
            f"manually first if a genuinely new manifest is intended."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_metadata_rows()
    print(f"Loaded {len(rows)} metadata rows from {SOURCE_METADATA}")

    diag_counts_raw = {}
    for r in rows:
        diag_counts_raw[r["diagnosis_1"]] = diag_counts_raw.get(r["diagnosis_1"], 0) + 1
    print(f"Raw diagnosis_1 distribution: {diag_counts_raw}")

    labeled_rows = [r for r in rows if r["diagnosis_1"] in ("Benign", "Malignant")]
    n_indeterminate = len(rows) - len(labeled_rows)
    print(f"Labeled (Benign/Malignant): {len(labeled_rows)}  |  "
          f"indeterminate/unlabeled excluded: {n_indeterminate}")

    seen_ids = set()
    deduped_rows = []
    n_dupes_dropped = 0
    for r in labeled_rows:
        if r["isic_id"] in seen_ids:
            n_dupes_dropped += 1
            continue
        seen_ids.add(r["isic_id"])
        deduped_rows.append(r)
    print(f"Duplicate isic_id rows dropped: {n_dupes_dropped}  |  remaining: {len(deduped_rows)}")

    benign_pool = [r for r in deduped_rows if r["diagnosis_1"] == "Benign"]
    malignant_pool = [r for r in deduped_rows if r["diagnosis_1"] == "Malignant"]
    print(f"Benign pool: {len(benign_pool)}  |  Malignant pool: {len(malignant_pool)}")

    if len(benign_pool) < N_PER_CLASS or len(malignant_pool) < N_PER_CLASS:
        raise SystemExit(
            f"Not enough images to sample {N_PER_CLASS} per class: "
            f"benign_pool={len(benign_pool)} malignant_pool={len(malignant_pool)}"
        )

    rng = random.Random(SEED)  # single generator, both draws sequential from it — reproducible end to end
    benign_sample = rng.sample(benign_pool, N_PER_CLASS)
    malignant_sample = rng.sample(malignant_pool, N_PER_CLASS)

    manifest_rows = []
    missing_files = []
    for r in benign_sample + malignant_sample:
        image_path = SOURCE_DIR / f"{r['isic_id']}.jpg"
        if not image_path.exists():
            missing_files.append(r["isic_id"])
            continue
        gt = "benign" if r["diagnosis_1"] == "Benign" else "malignant"
        manifest_rows.append({
            "image_id": r["isic_id"],
            "image_path": str(image_path),
            "ground_truth": gt,
            "ground_truth_binary": 0 if gt == "benign" else 1,
            "diagnosis_1": r["diagnosis_1"],
            "diagnosis_2": r["diagnosis_2"],
            "diagnosis_3": r["diagnosis_3"],
            "diagnosis_confirm_type": r["diagnosis_confirm_type"],
            "lesion_id": r["lesion_id"],
            "clin_size_long_diam_mm": r["clin_size_long_diam_mm"],
            "attribution": r["attribution"],
            "copyright_license": r["copyright_license"],
            "anatom_site_1": r["anatom_site_1"],
            "age_approx": r["age_approx"],
            "sex": r["sex"],
        })

    if missing_files:
        raise SystemExit(f"CRITICAL: {len(missing_files)} sampled image_ids have no file on disk: "
                          f"{missing_files[:10]}")

    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDNAMES)
        writer.writeheader()
        writer.writerows(manifest_rows)

    n_benign_final = sum(1 for r in manifest_rows if r["ground_truth"] == "benign")
    n_malignant_final = sum(1 for r in manifest_rows if r["ground_truth"] == "malignant")
    print(f"\nWrote manifest: {MANIFEST_PATH}")
    print(f"Total: {len(manifest_rows)}  |  benign: {n_benign_final}  |  malignant: {n_malignant_final}")
    print(f"Seed used: {SEED}")

    ids_check = set(r["image_id"] for r in manifest_rows)
    print(f"Unique image_ids in manifest: {len(ids_check)} (should equal {len(manifest_rows)})")


if __name__ == "__main__":
    main()
