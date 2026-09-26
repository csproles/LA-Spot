"""PHASE 5 — Controlled decision-layer comparison, DEVELOPMENT DATA ONLY.

Candidates (all built ONLY from existing, already-saved A/B/C values —
no new measurements, no deep learning, no YOLO/segmentation change):

  A) Current binary concern-count rule (A>0.20, unchanged) — the V2 baseline
  B) Binary concern-count rule with a revised A threshold, chosen from the
     Phase 4 sweep results (set via --a-threshold, must be one of the
     values actually tested in Phase 4 — no new search here)
  C) A simple, interpretable normalized continuous ABC score:
       norm_A = A_value / 0.20
       norm_B = B_value / 0.50
       norm_C = max(C_value / 0.35, 1.0 if C_concern else 0.0)
         (C's concern can also fire via a categorical dangerous-color path
         not reconstructable from saved data as a continuous value; when it
         fired, norm_C is floored at 1.0 so the continuous score still
         reflects that a concern was reached, rather than silently losing
         that signal.)
       score = norm_A + norm_B + norm_C
       predicted positive if score >= CUTOFF
     CUTOFF is fixed at 2.0 (the direct continuous analogue of "2 of 3
     criteria at their own threshold") as the primary candidate, with 1.8
     and 1.6 reported alongside as modest, principled alternatives — not a
     fine-grained search.

All three candidates preserve the critical-color override exactly as
inferred in Phase 4 (metrics_lib.infer_critical_override).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics_lib import (ROOT, load_merged_results, load_split_ids, build_split_rows,
                         compute_metrics, to_float, infer_critical_override,
                         predict_with_A_threshold, CURRENT_A_THRESHOLD)

DEV_MANIFEST = ROOT / "Evaluation_FinalTargeted" / "Cohort" / "development_manifest.csv"
OUT_DIR = ROOT / "Evaluation_FinalTargeted" / "DecisionModelAnalysis"

A_NORM_THRESHOLD = 0.20
B_NORM_THRESHOLD = 0.50
C_NORM_THRESHOLD = 0.35


def continuous_score(row):
    a = to_float(row["A_value"]) or 0.0
    b = to_float(row["B_circularity"]) or 0.0
    c_val = to_float(row["C_value"])
    c_concern = row["C_concern"] == "True"
    norm_a = a / A_NORM_THRESHOLD
    norm_b = b / B_NORM_THRESHOLD
    norm_c = max((c_val / C_NORM_THRESHOLD) if c_val is not None else 0.0, 1.0 if c_concern else 0.0)
    return norm_a + norm_b + norm_c


def predict_continuous(row, cutoff, override_fired):
    if override_fired:
        return 1
    return 1 if continuous_score(row) >= cutoff else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-threshold", type=float, default=0.14,
                        help="Candidate B's revised A threshold, chosen from Phase 4 results.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = load_merged_results()
    dev_ids = load_split_ids(DEV_MANIFEST)
    evaluable_rows, status_counts, n_images = build_split_rows(all_results, dev_ids)
    for r in evaluable_rows:
        r["_override_fired"] = infer_critical_override(r)

    candidates = {
        "A_current_binary_020": lambda r: predict_with_A_threshold(r, CURRENT_A_THRESHOLD, r["_override_fired"]),
        f"B_binary_revised_A_{args.a_threshold}": lambda r: predict_with_A_threshold(r, args.a_threshold, r["_override_fired"]),
        "C_continuous_cutoff_2.0": lambda r: predict_continuous(r, 2.0, r["_override_fired"]),
        "C_continuous_cutoff_1.8": lambda r: predict_continuous(r, 1.8, r["_override_fired"]),
        "C_continuous_cutoff_1.6": lambda r: predict_continuous(r, 1.6, r["_override_fired"]),
    }

    results = {}
    for name, fn in candidates.items():
        tp = tn = fp = fn_ = 0
        for r in evaluable_rows:
            gt = int(r["ground_truth_binary"])
            pred = fn(r)
            if gt == 1 and pred == 1: tp += 1
            elif gt == 0 and pred == 0: tn += 1
            elif gt == 0 and pred == 1: fp += 1
            elif gt == 1 and pred == 0: fn_ += 1
        m = compute_metrics(tp, tn, fp, fn_)
        results[name] = m
        print(f"{name}: TP={tp} TN={tn} FP={fp} FN={fn_} sens={m['sensitivity']:.3f} "
              f"spec={m['specificity']:.3f} prec={m['precision']:.3f} f1={m['f1']:.3f} "
              f"acc={m['accuracy']:.3f} bal_acc={m['balanced_accuracy']:.3f}")

    with open(OUT_DIR / "phase5_decision_candidates_development.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nWrote results to {OUT_DIR}")
    return results


if __name__ == "__main__":
    main()
