# Development-cohort pipeline comparison — reuse log

Branch: `pipeline-testing/research`. No file under `Code/`, `revised_abcd/`,
`pipeline_v3/`, `pipeline_v4/`, or `Evaluation_FinalTargeted/metrics_lib.py`
was modified. `user-shree` and the web app were not touched. The locked test
set (786 images, `Evaluation_FinalTargeted/Cohort/locked_test_manifest.csv`)
was loaded only to assert zero ID overlap with the development cohort — it
was never scored, never read for feature values, and played no role in any
decision made here.

## Cohort

- Development cohort: `Evaluation_FinalTargeted/Cohort/development_manifest.csv`,
  2,696 images (797 malignant / 1,899 benign), from the existing grouped
  patient/lesion split (`Evaluation_FinalTargeted/split_dev_test.py`, seed
  20260918). Not regenerated.
- Detection-status breakdown, identical across all four pipelines because
  they share the same cached YOLO segmentation:
  SINGLE_LESION_EVALUABLE = 2,090, MULTI_LESION_AMBIGUOUS = 421,
  NO_DETECTION = 185.

## What was reused as-is (zero new inference)

| Pipeline | Source of predictions | New computation |
|---|---|---|
| YOLO + ABCD V2 | `Evaluation_FinalTargeted/Cohort/cohort_v2_results.csv`, `provisional_prediction` column | none |
| YOLO + ABCD V3 | Same file's stored A/B/C values, decision recombined via the existing `Evaluation_FinalTargeted/metrics_lib.py::predict_with_A_threshold` (A-threshold 0.14) and `infer_critical_override` | none — pure re-derivation of an already-frozen decision rule on cached features |
| YOLO + V4 | `Evaluation_V4/CrossValidation/oof_predictions_full.json`, candidate `full_ABC_D_conf_fraction__logistic_regression`, out-of-fold probabilities | none |

The V4 out-of-fold array was verified to be in exact row-for-row order with
`Evaluation_V4/FeatureTable/development_feature_table.csv` (2,090 rows,
0 dropped, ground-truth columns matched exactly), so no realignment was
needed. **The out-of-fold score was used deliberately instead of scoring
development images with the frozen model** (`pipeline_v4/frozen_model.pkl`),
because that model was fit on the entire development set — scoring dev
images with it would be in-sample and would overstate V4's performance
relative to V2/V3/Original, which are fixed rules with no such advantage.

`Evaluation_FinalTargeted/Cohort/needs_processing_manifest.csv` (1,978 rows)
turned out to be fully superseded — every image_id in it is already present,
with a result, in `cohort_v2_results.csv`. No image needed fresh YOLO
inference.

## What was newly computed

Only the **YOLO + Original ABCD** arm required new computation, and it did
not touch YOLO:

1. For each of the 2,090 single-lesion-evaluable development images, the
   already-cached YOLO mask (`mask_path` column in `cohort_v2_results.csv`,
   physically verified to exist for every row) was loaded.
2. Asymmetry was recomputed using the unmodified, unaligned
   `Code/MelanomaDeterminingStuff/asymmetry.py::score_asymmetry` — the
   literal original classical-pipeline function — rather than V2's
   PCA-aligned `revised_abcd/revised_asymmetry.py`. This is the one place
   V2 differs from what "the original ABCD decision, fed a YOLO mask"
   would actually produce; B, C, and D are unchanged between the two
   already (confirmed: those functions are shared, unmodified files).
3. The decision was recombined with the existing rule (≥2 of A/B/C concerns
   → positive, with the same critical-color-override recovery method
   `metrics_lib.py` already uses for V3) at the original A-threshold of
   0.20.
4. 0 of 2,090 mask reads failed.

No YOLO model was run. No model was retrained. No threshold was tuned or
selected based on these results.

## Outputs

- `dev_predictions_per_image.csv` — one row per (image, pipeline): image_id,
  ground_truth, evaluation_status, predicted_class, A, B_circularity,
  C_value, D_px, decision_score (V4 only), yolo_confidence,
  mask_area_fraction. Includes informational rows (blank prediction) for
  MULTI_LESION_AMBIGUOUS and NO_DETECTION images, since those are excluded
  from scoring but still recorded per image.
- `dev_metrics_natural_population.csv` — TP/TN/FP/FN and derived metrics per
  pipeline, each pipeline's own evaluable set (2,090 for all four here,
  since segmentation is shared).
- `dev_metrics_common_subset.csv` — identical to the above in this run,
  because all four pipelines share segmentation and therefore share the
  exact same 2,090-image evaluable population. The natural/common-subset
  split becomes meaningful again if a differently-segmented arm (e.g. the
  classical Otsu segmentation) is added later.

## Script

`Evaluation_DevComparison/run_dev_comparison.py`, run with the repo's own
virtualenv (`.venv/Scripts/python.exe`; the ambient `python3` lacked
`scikit-learn`).
