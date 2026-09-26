"""PHASE 1 — Construct the targeted melanoma-vs-benign-melanocytic cohort.

READ-ONLY with respect to source data. Does not modify Evaluation_3500,
V2 code, YOLO weights, or the ISIC source images/metadata. Builds a new,
independent cohort manifest from the FULL 8277-image ISIC source, not
restricted to the earlier 3500-image random sample (that sample is reused
in Phase 3 wherever it overlaps, to avoid unnecessary reprocessing, but the
cohort DEFINITION here is driven purely by the clinical/task criteria
below, never by which images already happen to have V2 results).

INCLUSION CRITERIA (documented, not assumed):
  POSITIVE (melanoma):
    - diagnosis_confirm_type == "histopathology"  (true for all 8277 rows
      in this filtered download, verified, so this is not a filter that
      removes anything here — recorded for transparency)
    - diagnosis_2 == "Malignant melanocytic proliferations (Melanoma)"
    - diagnosis_3 in {"Melanoma Invasive", "Melanoma in situ",
      "Melanoma, NOS", "Melanoma metastasis"}
    - melanocytic == "True" (verified: 100% of the above already are)
    - image_type == "dermoscopic" (true for all 8277 rows, recorded for
      transparency, not an active filter)

  NEGATIVE (benign melanocytic / nevus):
    - diagnosis_confirm_type == "histopathology" (same as above)
    - diagnosis_2 == "Benign melanocytic proliferations"
    - diagnosis_3 in {"Nevus", "Lentigo simplex"} — both are the only two
      diagnosis_3 values that occur under this diagnosis_2 in the full
      dataset; both are melanocytic=True. Lentigo simplex is included
      because it is a benign MELANOCYTIC lesion (matches "benign
      melanocytic lesions/nevi" as written), not because it improves any
      metric — flagged explicitly here so it can be reviewed and excluded
      later if a stricter "nevus-only" definition is preferred.
    - melanocytic == "True" (verified: 100% of the above already are)
    - image_type == "dermoscopic"

EXPLICITLY EXCLUDED from melanoma (positive class):
  BCC, SCC, keratoacanthoma, Paget disease, any other malignant diagnosis_2
  category, and the 2 "Collision" cases with no specific diagnosis_3.

EXPLICITLY EXCLUDED from benign melanocytic (negative class):
  All other benign categories (seborrheic keratosis, dermatofibroma,
  solar lentigo, lentigo NOS, hemangioma, etc.) — these are benign but NOT
  melanocytic-nevus-family lesions, so they don't match the task definition
  and are excluded rather than folded in.

AMBIGUOUS / EXCLUDED:
  Any row where diagnosis_2 or diagnosis_3 falls outside the exact sets
  above is excluded from this cohort entirely (neither class) — no case is
  guessed into a bucket.

No selection here is based on whether the existing V2 pipeline classified
an image correctly — that information isn't even loaded in this script.
"""

import csv
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
SOURCE_METADATA = Path(r"C:\Users\sirjanaa\Downloads\ISIC-images\metadata.csv")
SOURCE_IMAGES_DIR = Path(r"C:\Users\sirjanaa\Downloads\ISIC-images")
OUT_DIR = ROOT / "Evaluation_FinalTargeted" / "Cohort"
EXISTING_RESULTS_CSV = ROOT / "Evaluation_3500" / "full_run" / "results.csv"

MELANOMA_DIAG2 = "Malignant melanocytic proliferations (Melanoma)"
MELANOMA_DIAG3 = {"Melanoma Invasive", "Melanoma in situ", "Melanoma, NOS", "Melanoma metastasis"}
BENIGN_MEL_DIAG2 = "Benign melanocytic proliferations"
BENIGN_MEL_DIAG3 = {"Nevus", "Lentigo simplex"}

COHORT_FIELDNAMES = [
    "image_id", "image_path", "ground_truth", "ground_truth_binary",
    "diagnosis_1", "diagnosis_2", "diagnosis_3", "diagnosis_4", "diagnosis_5",
    "diagnosis_confirm_type", "melanocytic", "image_type", "dermoscopic_type",
    "patient_id", "lesion_id", "attribution", "copyright_license",
    "anatom_site_1", "age_approx", "sex", "has_existing_v2_result",
]


def load_metadata():
    with open(SOURCE_METADATA, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_existing_result_ids():
    if not EXISTING_RESULTS_CSV.exists():
        return set()
    with open(EXISTING_RESULTS_CSV, newline="", encoding="utf-8") as f:
        return {Path(r["image_name"]).stem for r in csv.DictReader(f)}


def classify(row):
    d2, d3 = row["diagnosis_2"], row["diagnosis_3"]
    if d2 == MELANOMA_DIAG2 and d3 in MELANOMA_DIAG3:
        return "malignant"
    if d2 == BENIGN_MEL_DIAG2 and d3 in BENIGN_MEL_DIAG3:
        return "benign"
    return None


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = load_metadata()
    existing_ids = load_existing_result_ids()
    print(f"Loaded {len(meta)} metadata rows; {len(existing_ids)} images already have V2 results")

    cohort_rows = []
    excluded_count = 0
    for r in meta:
        gt = classify(r)
        if gt is None:
            excluded_count += 1
            continue
        image_id = r["isic_id"]
        image_path = SOURCE_IMAGES_DIR / f"{image_id}.jpg"
        if not image_path.exists():
            excluded_count += 1
            continue
        cohort_rows.append({
            "image_id": image_id, "image_path": str(image_path),
            "ground_truth": gt, "ground_truth_binary": 1 if gt == "malignant" else 0,
            "diagnosis_1": r["diagnosis_1"], "diagnosis_2": r["diagnosis_2"], "diagnosis_3": r["diagnosis_3"],
            "diagnosis_4": r["diagnosis_4"], "diagnosis_5": r["diagnosis_5"],
            "diagnosis_confirm_type": r["diagnosis_confirm_type"], "melanocytic": r["melanocytic"],
            "image_type": r["image_type"], "dermoscopic_type": r["dermoscopic_type"],
            "patient_id": r["patient_id"], "lesion_id": r["lesion_id"],
            "attribution": r["attribution"], "copyright_license": r["copyright_license"],
            "anatom_site_1": r["anatom_site_1"], "age_approx": r["age_approx"], "sex": r["sex"],
            "has_existing_v2_result": image_id in existing_ids,
        })

    with open(OUT_DIR / "targeted_cohort_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COHORT_FIELDNAMES)
        w.writeheader(); w.writerows(cohort_rows)

    n_mel = sum(1 for r in cohort_rows if r["ground_truth"] == "malignant")
    n_ben = sum(1 for r in cohort_rows if r["ground_truth"] == "benign")
    n_with_results = sum(1 for r in cohort_rows if r["has_existing_v2_result"])
    n_needing_processing = len(cohort_rows) - n_with_results

    n_patient_mel = sum(1 for r in cohort_rows if r["ground_truth"] == "malignant" and r["patient_id"].strip())
    n_patient_ben = sum(1 for r in cohort_rows if r["ground_truth"] == "benign" and r["patient_id"].strip())
    n_lesion_mel = sum(1 for r in cohort_rows if r["ground_truth"] == "malignant" and r["lesion_id"].strip())
    n_lesion_ben = sum(1 for r in cohort_rows if r["ground_truth"] == "benign" and r["lesion_id"].strip())

    diag3_mel = Counter(r["diagnosis_3"] for r in cohort_rows if r["ground_truth"] == "malignant")
    diag3_ben = Counter(r["diagnosis_3"] for r in cohort_rows if r["ground_truth"] == "benign")
    attr_mel = Counter(r["attribution"] for r in cohort_rows if r["ground_truth"] == "malignant")
    attr_ben = Counter(r["attribution"] for r in cohort_rows if r["ground_truth"] == "benign")

    report_lines = [
        "PHASE 1 — TARGETED COHORT REPORT",
        "=" * 50,
        "",
        "INCLUSION CRITERIA:",
        f"  Positive (melanoma): diagnosis_2='{MELANOMA_DIAG2}' AND diagnosis_3 in {sorted(MELANOMA_DIAG3)}",
        f"  Negative (benign melanocytic): diagnosis_2='{BENIGN_MEL_DIAG2}' AND diagnosis_3 in {sorted(BENIGN_MEL_DIAG3)}",
        "  Both classes additionally require melanocytic=True and image_type=dermoscopic",
        "  (verified: both are already 100% true for every qualifying row — not an active filter)",
        "  All 8277 source rows are diagnosis_confirm_type=histopathology (verified dataset-wide)",
        "",
        "EXCLUSION:",
        f"  {excluded_count} images excluded (all other diagnoses, or missing image file) — no guessing",
        "",
        f"Melanoma (positive):        {n_mel}",
        f"  by subtype: {dict(diag3_mel)}",
        f"Benign melanocytic (negative): {n_ben}",
        f"  by subtype: {dict(diag3_ben)}",
        "",
        f"patient_id populated: melanoma {n_patient_mel}/{n_mel} ({100*n_patient_mel/n_mel:.1f}%), "
        f"benign {n_patient_ben}/{n_ben} ({100*n_patient_ben/n_ben:.1f}%)",
        f"lesion_id populated:  melanoma {n_lesion_mel}/{n_mel} ({100*n_lesion_mel/n_mel:.1f}%), "
        f"benign {n_lesion_ben}/{n_ben} ({100*n_lesion_ben/n_ben:.1f}%)",
        "",
        f"Source/institution (melanoma): {dict(attr_mel)}",
        f"Source/institution (benign):   {dict(attr_ben)}",
        "",
        f"Total cohort: {len(cohort_rows)}",
        f"Already have existing V2 pipeline results (from Evaluation_3500): {n_with_results}",
        f"Require NEW V2 pipeline execution: {n_needing_processing}",
    ]
    report = "\n".join(report_lines)
    print("\n" + report)
    with open(OUT_DIR / "cohort_report.txt", "w", encoding="utf-8") as f:
        f.write(report)


if __name__ == "__main__":
    main()
