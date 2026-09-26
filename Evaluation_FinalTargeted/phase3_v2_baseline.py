"""PHASE 3 — Targeted V2 baseline on DEVELOPMENT data, using the unchanged
existing V2 decision rule (provisional_prediction from risk_level)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics_lib import (ROOT, load_merged_results, load_split_ids, build_split_rows,
                         confusion_from_predictions, compute_metrics, v2_predict)

DEV_MANIFEST = ROOT / "Evaluation_FinalTargeted" / "Cohort" / "development_manifest.csv"
OUT_DIR = ROOT / "Evaluation_FinalTargeted" / "Development"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = load_merged_results()
    dev_ids = load_split_ids(DEV_MANIFEST)
    print(f"Development set size: {len(dev_ids)}")

    evaluable_rows, status_counts, n_images = build_split_rows(all_results, dev_ids)
    tp, tn, fp, fn = confusion_from_predictions(evaluable_rows, v2_predict)
    metrics = compute_metrics(tp, tn, fp, fn)

    report = {
        "phase": "Phase 3 - V2 targeted baseline (development set)",
        "total_development_images": n_images,
        "coverage": status_counts,
        "metrics": metrics,
    }
    print(json.dumps(report, indent=2))
    with open(OUT_DIR / "phase3_v2_baseline_development.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    main()
