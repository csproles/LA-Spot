"""
Controlled development-cohort comparison of four melanoma-classification
decision layers, all built on the SAME cached YOLO segmentation:

  - "YOLO + Original ABCD" : unmodified Code/MelanomaDeterminingStuff asymmetry
                              function, applied to the existing cached YOLO
                              masks; B/C/D reused unchanged from the existing
                              V2 results (those functions are identical to the
                              classical originals already).
  - "YOLO + ABCD V2"        : cached, unmodified (provisional_prediction column
                              in cohort_v2_results.csv).
  - "YOLO + ABCD V3"        : decision-layer-only re-derivation on the cached
                              A/B/C values, via the existing, unmodified
                              Evaluation_FinalTargeted/metrics_lib.py helpers.
  - "YOLO + V4"             : cached out-of-fold logistic-regression score
                              from Evaluation_V4/CrossValidation, NOT the
                              in-sample frozen-model score (avoids handing V4
                              an unfair advantage from having been fit on the
                              full development set).

No YOLO inference is rerun. No model is retrained. No threshold is tuned.
Only the 2,696-image development cohort is used; the 786-image locked test
set is loaded only to assert there is zero overlap, never scored.
"""

import csv
import importlib.util
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Evaluation_FinalTargeted"))
import metrics_lib as ml  # noqa: E402  (existing, unmodified helper module)

# ---- load the ORIGINAL (non-PCA-aligned) asymmetry function directly from
# Code/, without going through Code's own relative-import package layout ----
_asym_path = ROOT / "Code" / "MelanomaDeterminingStuff" / "asymmetry.py"
_spec = importlib.util.spec_from_file_location("original_asymmetry", _asym_path)
_original_asymmetry_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_original_asymmetry_mod)
score_asymmetry_original = _original_asymmetry_mod.score_asymmetry

DEV_MANIFEST = ROOT / "Evaluation_FinalTargeted" / "Cohort" / "development_manifest.csv"
LOCKED_MANIFEST = ROOT / "Evaluation_FinalTargeted" / "Cohort" / "locked_test_manifest.csv"
FEATURE_TABLE = ROOT / "Evaluation_V4" / "FeatureTable" / "development_feature_table.csv"
OOF_JSON = ROOT / "Evaluation_V4" / "CrossValidation" / "oof_predictions_full.json"
V4_CANDIDATE_KEY = "full_ABC_D_conf_fraction__logistic_regression"
V4_THRESHOLD = 0.25
V3_A_THRESHOLD = 0.14

OUT_DIR = ROOT / "Evaluation_DevComparison"
OUT_DIR.mkdir(exist_ok=True)

PIPELINES = ["YOLO + Original ABCD", "YOLO + ABCD V2", "YOLO + ABCD V3", "YOLO + V4"]


def load_dev_manifest():
    with open(DEV_MANIFEST, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {r["image_id"]: r for r in rows}


def load_locked_ids():
    with open(LOCKED_MANIFEST, newline="", encoding="utf-8") as f:
        return {r["image_id"] for r in csv.DictReader(f)}


def load_v4_oof():
    with open(FEATURE_TABLE, newline="", encoding="utf-8") as f:
        feat_rows = list(csv.DictReader(f))
    with open(OOF_JSON, encoding="utf-8") as f:
        candidate = json.load(f)[V4_CANDIDATE_KEY]
    probs = candidate["oof_probabilities"]
    y_true = candidate["y_true"]
    assert len(feat_rows) == len(probs) == len(y_true), (
        "V4 OOF / feature-table row-count mismatch: "
        f"{len(feat_rows)} vs {len(probs)} vs {len(y_true)}"
    )
    out = {}
    for row, p, y in zip(feat_rows, probs, y_true):
        assert int(row["ground_truth_binary"]) == int(y), (
            f"Ground-truth mismatch for {row['image_id']} between feature table and OOF file"
        )
        out[row["image_id"]] = p
    return out


def compute_original_asymmetry(mask_path_str):
    mask_path = ROOT / mask_path_str
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return score_asymmetry_original(mask)


def gt_of(dev_manifest, image_id):
    return int(dev_manifest[image_id]["ground_truth_binary"])


def confusion_and_metrics(ids, preds, dev_manifest):
    tp = tn = fp = fn = 0
    for iid in ids:
        g = gt_of(dev_manifest, iid)
        pr = preds[iid]
        if g == 1 and pr == 1:
            tp += 1
        elif g == 0 and pr == 0:
            tn += 1
        elif g == 0 and pr == 1:
            fp += 1
        elif g == 1 and pr == 0:
            fn += 1
    return ml.compute_metrics(tp, tn, fp, fn)


def main():
    dev_manifest = load_dev_manifest()
    dev_ids = set(dev_manifest.keys())
    locked_ids = load_locked_ids()
    assert dev_ids.isdisjoint(locked_ids), "Development and locked-test IDs overlap -- aborting"
    print(f"Development cohort: {len(dev_ids)} images. Locked test loaded only for overlap check "
          f"({len(locked_ids)} images, never scored).")

    all_results = ml.load_merged_results()
    evaluable_rows, status_counts, n_images = ml.build_split_rows(all_results, dev_ids)
    assert n_images == len(dev_ids), (
        f"Expected all {len(dev_ids)} dev images accounted for in cached V2 results, got {n_images}"
    )
    print("Detection-status breakdown (shared across all 4 pipelines, same YOLO segmentation):",
          status_counts)

    by_image = {}
    for r in all_results:
        image_id = ml.image_id_from_name(r["image_name"])
        if image_id not in dev_ids:
            continue
        by_image.setdefault(image_id, []).append(r)

    v4_oof = load_v4_oof()
    v4_ids = set(v4_oof.keys())
    evaluable_ids = {ml.image_id_from_name(r["image_name"]) for r in evaluable_rows}
    mismatch = evaluable_ids.symmetric_difference(v4_ids)
    if mismatch:
        print(f"WARNING: {len(mismatch)} images differ between the V2/V3-evaluable set and the "
              f"V4 OOF set. Sample: {sorted(mismatch)[:10]}")
    else:
        print(f"V4 OOF population exactly matches the {len(evaluable_ids)}-image "
              f"SINGLE_LESION_EVALUABLE set used by V2/V3/Original.")

    per_image_records = []
    preds = {p: {} for p in PIPELINES}
    scores = {p: {} for p in PIPELINES}
    arm1_failed_ids = set()

    for row in evaluable_rows:
        image_id = ml.image_id_from_name(row["image_name"])
        b_concern = row["B_circularity_concern"] == "True"
        c_concern = row["C_concern"] == "True"
        override_fired = ml.infer_critical_override(row)

        v2_pred = ml.v2_predict(row)
        preds["YOLO + ABCD V2"][image_id] = v2_pred

        v3_pred = ml.predict_with_A_threshold(row, V3_A_THRESHOLD, override_fired)
        preds["YOLO + ABCD V3"][image_id] = v3_pred

        if image_id in v4_oof:
            v4_score = v4_oof[image_id]
            preds["YOLO + V4"][image_id] = 1 if v4_score >= V4_THRESHOLD else 0
            scores["YOLO + V4"][image_id] = v4_score

        a_result = compute_original_asymmetry(row["mask_path"])
        arm1_a_value = ""
        if a_result is None:
            arm1_failed_ids.add(image_id)
        else:
            arm1_a_value = a_result["value"]
            new_a_concern = a_result["concern"]
            concerns = sum([new_a_concern, b_concern, c_concern])
            arm1_pred = 1 if override_fired else (1 if concerns >= 2 else 0)
            preds["YOLO + Original ABCD"][image_id] = arm1_pred

        common = dict(
            image_id=image_id,
            ground_truth=row["ground_truth"],
            ground_truth_binary=row["ground_truth_binary"],
            evaluation_status="SINGLE_LESION_EVALUABLE",
            B_circularity=row["B_circularity"],
            C_value=row["C_value"],
            D_px=row["D_px"],
            yolo_confidence=row["confidence"],
            mask_area_fraction=row["lesion_fraction"],
        )

        rec = dict(common, pipeline="YOLO + Original ABCD", A=arm1_a_value, decision_score="")
        rec["predicted_class"] = (
            ("malignant" if preds["YOLO + Original ABCD"][image_id] == 1 else "benign")
            if image_id in preds["YOLO + Original ABCD"] else ""
        )
        per_image_records.append(rec)

        rec = dict(common, pipeline="YOLO + ABCD V2", A=row["A_value"], decision_score="")
        rec["predicted_class"] = "malignant" if v2_pred == 1 else "benign"
        per_image_records.append(rec)

        rec = dict(common, pipeline="YOLO + ABCD V3", A=row["A_value"], decision_score="")
        rec["predicted_class"] = "malignant" if v3_pred == 1 else "benign"
        per_image_records.append(rec)

        rec = dict(common, pipeline="YOLO + V4", A=row["A_value"])
        if image_id in scores["YOLO + V4"]:
            rec["decision_score"] = scores["YOLO + V4"][image_id]
            rec["predicted_class"] = "malignant" if preds["YOLO + V4"][image_id] == 1 else "benign"
        else:
            rec["decision_score"] = ""
            rec["predicted_class"] = ""
        per_image_records.append(rec)

    # Informational rows for NO_DETECTION / MULTI_LESION_AMBIGUOUS images (excluded from scoring,
    # identical detection outcome across all 4 pipelines by construction).
    for image_id, rows in by_image.items():
        status = rows[0]["evaluation_status"] or rows[0]["status"]
        if status == "SINGLE_LESION_EVALUABLE":
            continue
        gt_row = dev_manifest[image_id]
        primary = next((r for r in rows if r.get("is_primary_instance") == "True"), rows[0])
        for p in PIPELINES:
            per_image_records.append(dict(
                image_id=image_id,
                ground_truth=gt_row["ground_truth"],
                ground_truth_binary=gt_row["ground_truth_binary"],
                evaluation_status=status,
                pipeline=p,
                predicted_class="",
                A=primary.get("A_value", ""),
                B_circularity=primary.get("B_circularity", ""),
                C_value=primary.get("C_value", ""),
                D_px=primary.get("D_px", ""),
                decision_score="",
                yolo_confidence=primary.get("confidence", ""),
                mask_area_fraction=primary.get("lesion_fraction", ""),
            ))

    fieldnames = ["image_id", "ground_truth", "ground_truth_binary", "evaluation_status",
                  "pipeline", "predicted_class", "A", "B_circularity", "C_value", "D_px",
                  "decision_score", "yolo_confidence", "mask_area_fraction"]
    with open(OUT_DIR / "dev_predictions_per_image.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in per_image_records:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    def build_metric_rows(ids_by_pipeline):
        rows_out = []
        for p in PIPELINES:
            ids = ids_by_pipeline[p]
            m = confusion_and_metrics(ids, preds[p], dev_manifest)
            auc = ""
            if p == "YOLO + V4" and ids:
                from sklearn.metrics import roc_auc_score
                y = [gt_of(dev_manifest, i) for i in ids]
                s = [scores[p][i] for i in ids]
                if len(set(y)) > 1:
                    auc = round(roc_auc_score(y, s), 4)
            rows_out.append(dict(
                pipeline=p, **m, roc_auc=auc,
                evaluable_coverage=len(ids),
                NO_DETECTION=status_counts.get("NO_DETECTION", 0),
                MULTI_LESION_AMBIGUOUS=status_counts.get("MULTI_LESION_AMBIGUOUS", 0),
                total_dev_images=len(dev_ids),
            ))
        return rows_out

    natural_ids = {p: set(preds[p].keys()) for p in PIPELINES}
    natural_rows = build_metric_rows(natural_ids)

    common_ids = set.intersection(*[natural_ids[p] for p in PIPELINES])
    common_ids_by_pipeline = {p: common_ids for p in PIPELINES}
    common_rows = build_metric_rows(common_ids_by_pipeline)

    metric_fields = ["pipeline", "TP", "TN", "FP", "FN", "n", "sensitivity", "specificity",
                      "precision", "f1", "accuracy", "balanced_accuracy", "roc_auc",
                      "evaluable_coverage", "NO_DETECTION", "MULTI_LESION_AMBIGUOUS",
                      "total_dev_images"]

    with open(OUT_DIR / "dev_metrics_natural_population.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=metric_fields)
        w.writeheader()
        for r in natural_rows:
            w.writerow(r)

    with open(OUT_DIR / "dev_metrics_common_subset.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=metric_fields)
        w.writeheader()
        for r in common_rows:
            w.writerow(r)

    print(f"\nArm-1 (Original ABCD) mask-read failures: {len(arm1_failed_ids)} -> {sorted(arm1_failed_ids)}")
    print(f"Common subset evaluable by all 4 pipelines: {len(common_ids)} images "
          f"(natural per-pipeline sizes: {[len(natural_ids[p]) for p in PIPELINES]})")
    print("\n--- Natural population ---")
    for r in natural_rows:
        print(f"{r['pipeline']:24s} n={r['n']:4d} TP={r['TP']:3d} TN={r['TN']:4d} FP={r['FP']:3d} "
              f"FN={r['FN']:3d} sens={r['sensitivity']:.3f} spec={r['specificity']:.3f} "
              f"bal_acc={r['balanced_accuracy']:.3f} auc={r['roc_auc']}")
    print("\n--- Common subset ---")
    for r in common_rows:
        print(f"{r['pipeline']:24s} n={r['n']:4d} TP={r['TP']:3d} TN={r['TN']:4d} FP={r['FP']:3d} "
              f"FN={r['FN']:3d} sens={r['sensitivity']:.3f} spec={r['specificity']:.3f} "
              f"bal_acc={r['balanced_accuracy']:.3f} auc={r['roc_auc']}")

    return dict(natural_rows=natural_rows, common_rows=common_rows, preds=preds, scores=scores,
                arm1_failed_ids=arm1_failed_ids, status_counts=status_counts,
                common_ids=common_ids, dev_manifest=dev_manifest)


if __name__ == "__main__":
    main()
