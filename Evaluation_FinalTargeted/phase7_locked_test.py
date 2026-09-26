"""PHASE 7 — Locked final test. Run ONCE, after V3 is completely frozen.

V3 only changes the DECISION LAYER (see pipeline_v3/decision.py) — it does
not change segmentation or ABCD extraction, so no new YOLO/ABCD computation
is needed here. Both V2's and V3's predictions are derived from the SAME
already-computed, already-saved A/B/C/D values for every locked-test image;
this script only applies two different decision functions to that one
shared dataset and reports both.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics_lib import (ROOT, load_merged_results, load_split_ids, build_split_rows,
                         compute_metrics, infer_critical_override, v2_predict)

sys.path.insert(0, str(ROOT))
from pipeline_v3.decision import v3_predict_from_row  # frozen, see pipeline_v3/decision.py

TEST_MANIFEST = ROOT / "Evaluation_FinalTargeted" / "Cohort" / "locked_test_manifest.csv"
OUT_DIR = ROOT / "Evaluation_FinalTargeted" / "LockedTest"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = load_merged_results()
    test_ids = load_split_ids(TEST_MANIFEST)
    evaluable_rows, status_counts, n_images = build_split_rows(all_results, test_ids)
    for r in evaluable_rows:
        r["_override_fired"] = infer_critical_override(r)

    def confusion(predict_fn):
        tp = tn = fp = fn = 0
        for r in evaluable_rows:
            gt = int(r["ground_truth_binary"])
            pred = predict_fn(r)
            if gt == 1 and pred == 1: tp += 1
            elif gt == 0 and pred == 0: tn += 1
            elif gt == 0 and pred == 1: fp += 1
            elif gt == 1 and pred == 0: fn += 1
        return compute_metrics(tp, tn, fp, fn)

    v2_metrics = confusion(v2_predict)
    v3_metrics = confusion(lambda r: v3_predict_from_row(r, r["_override_fired"]))

    report = {
        "phase": "Phase 7 - Locked test (run once)",
        "total_locked_test_images": n_images,
        "coverage": status_counts,
        "V2": v2_metrics,
        "V3": v3_metrics,
    }
    print(json.dumps(report, indent=2))
    with open(OUT_DIR / "phase7_locked_test_results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


if __name__ == "__main__":
    main()
