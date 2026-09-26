"""Diagnosis-specific analysis of the completed Evaluation_3500 run.

READ-ONLY: joins the existing manifest + results.csv + ISIC source metadata
by isic_id. Does not rerun YOLO/ABCD, does not touch masks, thresholds, or
the evaluation policy, and does not modify any prior output.

Categorization of malignant diagnoses (melanoma / BCC / SCC / other /
unknown) reuses the exact logic already validated and reported in the
earlier metadata composition check (Evaluation_3500/VisualVerification/
malignant_and_benign_diagnosis_composition.txt), cross-checked there against
the melanocytic field.

Population conventions used throughout (stated explicitly so nothing is
ambiguous):
  - "total selected"                    = every manifest row in that category
  - SINGLE_LESION_EVALUABLE / MULTI_LESION_AMBIGUOUS / NO_DETECTION / FAILED
                                         = per-image evaluation_status counts
                                           (every image counted exactly once)
  - TP / FN / sensitivity                = SINGLE_LESION_EVALUABLE only, per
                                            instruction
  - mean/median A, B, C, concern rates, mask area fraction, large_area_flag
    rate, mean confidence
                                         = ALSO restricted to
                                           SINGLE_LESION_EVALUABLE rows only,
                                           for internal consistency (one row
                                           per image, unambiguous ground
                                           truth, no double-counting from
                                           multi-instance images)
"""

import csv
import statistics as st
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent
RESULTS_CSV = ROOT / "Evaluation_3500" / "full_run" / "results.csv"
MANIFEST_CSV = ROOT / "Evaluation_3500" / "evaluation_manifest_3500.csv"
SOURCE_METADATA = Path(r"C:\Users\sirjanaa\Downloads\ISIC-images\metadata.csv")
OUT_DIR = ROOT / "Evaluation_3500" / "DiagnosisSpecificAnalysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def classify_malignant_category(meta_row):
    diag3 = (meta_row.get("diagnosis_3") or "").strip()
    diag2 = (meta_row.get("diagnosis_2") or "").strip()
    melanoma_diag3 = {"Melanoma Invasive", "Melanoma in situ", "Melanoma, NOS", "Melanoma metastasis"}
    scc_diag3 = {"Squamous cell carcinoma, Invasive", "Squamous cell carcinoma in situ",
                 "Squamous cell carcinoma, NOS"}
    if diag3 in melanoma_diag3:
        return "Melanoma"
    if diag3 == "Basal cell carcinoma":
        return "Basal cell carcinoma"
    if diag3 in scc_diag3:
        return "Squamous cell carcinoma"
    if diag3 in ("Keratoacanthoma", "Paget disease"):
        return "Other malignant"
    if diag2.startswith("Collision") and not diag3:
        return "Unknown"
    return "Unknown"


def to_float(x):
    if x is None or x == "":
        return None
    try:
        return float(x)
    except ValueError:
        return None


def describe(values):
    values = [v for v in values if v is not None]
    if not values:
        return {"count": 0, "mean": None, "median": None}
    return {"count": len(values), "mean": round(st.mean(values), 4), "median": round(st.median(values), 4)}


def load_all():
    manifest = list(csv.DictReader(open(MANIFEST_CSV, newline="", encoding="utf-8")))
    results = list(csv.DictReader(open(RESULTS_CSV, newline="", encoding="utf-8")))
    meta = {r["isic_id"]: r for r in csv.DictReader(open(SOURCE_METADATA, newline="", encoding="utf-8"))}
    return manifest, results, meta


def image_id_from_name(image_name):
    return Path(image_name).stem


def main():
    manifest, results, meta = load_all()
    manifest_by_id = {r["image_id"]: r for r in manifest}

    # one row per image for status/evaluation purposes (all instance rows of
    # an image share the same evaluation_status; the primary/instance-0 row
    # is used for the single-lesion A/B/C/confidence/etc. values)
    rows_by_image = defaultdict(list)
    for r in results:
        rows_by_image[image_id_from_name(r["image_name"])].append(r)

    malignant_ids = [r["image_id"] for r in manifest if r["ground_truth"] == "malignant"]
    print(f"Malignant images in manifest: {len(malignant_ids)}")

    category_of = {}
    for image_id in malignant_ids:
        category_of[image_id] = classify_malignant_category(meta[image_id])

    from collections import Counter
    print("Category counts:", dict(Counter(category_of.values())))

    categories = ["Melanoma", "Basal cell carcinoma", "Squamous cell carcinoma", "Other malignant", "Unknown"]

    def analyze_group(image_ids, label):
        n_total = len(image_ids)
        status_counts = Counter()
        evaluable_rows = []  # single row per SINGLE_LESION_EVALUABLE image
        tp = fn = 0

        for image_id in image_ids:
            image_rows = rows_by_image.get(image_id, [])
            if not image_rows:
                status_counts["MISSING_FROM_RESULTS"] += 1
                continue
            status = image_rows[0]["status"]
            eval_status = image_rows[0]["evaluation_status"]
            status_counts[eval_status if eval_status else status] += 1

            if eval_status == "SINGLE_LESION_EVALUABLE":
                row = image_rows[0]  # exactly one instance by definition
                evaluable_rows.append(row)
                pred = row["provisional_prediction"]
                if pred == "positive":
                    tp += 1
                elif pred == "negative":
                    fn += 1

        sensitivity = tp / (tp + fn) if (tp + fn) else None

        a_vals = [to_float(r["A_value"]) for r in evaluable_rows]
        b_vals = [to_float(r["B_circularity"]) for r in evaluable_rows]
        c_vals = [to_float(r["C_value"]) for r in evaluable_rows]
        conf_vals = [to_float(r["confidence"]) for r in evaluable_rows]
        frac_vals = [to_float(r["lesion_fraction"]) for r in evaluable_rows]

        n_eval = len(evaluable_rows)
        a_concern = sum(1 for r in evaluable_rows if r["A_concern"] == "True")
        b_concern = sum(1 for r in evaluable_rows if r["B_circularity_concern"] == "True")
        c_concern = sum(1 for r in evaluable_rows if r["C_concern"] == "True")
        large_area = sum(1 for r in evaluable_rows if "large_area_fraction" in (r.get("quality_flags") or ""))

        return {
            "category": label,
            "total_selected": n_total,
            "SINGLE_LESION_EVALUABLE": status_counts.get("SINGLE_LESION_EVALUABLE", 0),
            "MULTI_LESION_AMBIGUOUS": status_counts.get("MULTI_LESION_AMBIGUOUS", 0),
            "NO_DETECTION": status_counts.get("NO_DETECTION", 0),
            "FAILED": status_counts.get("FAILED", 0),
            "MISSING_FROM_RESULTS": status_counts.get("MISSING_FROM_RESULTS", 0),
            "TP": tp, "FN": fn,
            "sensitivity": round(sensitivity, 4) if sensitivity is not None else None,
            "A_mean": describe(a_vals)["mean"], "A_median": describe(a_vals)["median"],
            "A_concern_rate": round(100 * a_concern / n_eval, 1) if n_eval else None,
            "B_mean": describe(b_vals)["mean"], "B_median": describe(b_vals)["median"],
            "B_concern_rate": round(100 * b_concern / n_eval, 1) if n_eval else None,
            "C_mean": describe(c_vals)["mean"], "C_median": describe(c_vals)["median"],
            "C_concern_rate": round(100 * c_concern / n_eval, 1) if n_eval else None,
            "mask_area_fraction_mean": describe(frac_vals)["mean"],
            "mask_area_fraction_median": describe(frac_vals)["median"],
            "large_area_fraction_rate": round(100 * large_area / n_eval, 1) if n_eval else None,
            "confidence_mean": describe(conf_vals)["mean"],
        }

    category_results = []
    for cat in categories:
        ids = [i for i in malignant_ids if category_of[i] == cat]
        if not ids:
            continue
        category_results.append(analyze_group(ids, cat))

    # overall malignant (all categories combined) for reference/sanity check
    overall = analyze_group(malignant_ids, "ALL MALIGNANT (combined)")
    category_results.append(overall)

    fieldnames = list(category_results[0].keys())
    with open(OUT_DIR / "malignant_category_comparison.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(category_results)

    print("\n=== Malignant category comparison ===")
    for r in category_results:
        print(f"{r['category']:28s} n={r['total_selected']:4d}  eval={r['SINGLE_LESION_EVALUABLE']:4d}  "
              f"TP={r['TP']:4d} FN={r['FN']:4d}  sens={r['sensitivity']}  "
              f"A_mean={r['A_mean']}  large_area_rate={r['large_area_fraction_rate']}%")

    # --- melanoma subtype breakdown ---
    melanoma_ids = [i for i in malignant_ids if category_of[i] == "Melanoma"]
    subtype_of = {}
    for image_id in melanoma_ids:
        d3 = (meta[image_id].get("diagnosis_3") or "").strip()
        subtype_of[image_id] = d3

    subtypes = sorted(set(subtype_of.values()))
    subtype_results = []
    for subtype in subtypes:
        ids = [i for i in melanoma_ids if subtype_of[i] == subtype]
        subtype_results.append(analyze_group(ids, subtype))

    with open(OUT_DIR / "melanoma_subtype_comparison.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(subtype_results)

    print("\n=== Melanoma subtype comparison ===")
    for r in subtype_results:
        print(f"{r['category']:28s} n={r['total_selected']:4d}  eval={r['SINGLE_LESION_EVALUABLE']:4d}  "
              f"TP={r['TP']:4d} FN={r['FN']:4d}  sens={r['sensitivity']}")

    print(f"\nWrote outputs to {OUT_DIR}")
    return category_results, subtype_results


if __name__ == "__main__":
    main()
