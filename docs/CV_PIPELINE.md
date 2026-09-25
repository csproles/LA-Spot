# Computer-vision pipeline: history, architecture, and evaluation

This document explains how the image-analysis pipeline evolved from the
original classical prototype to the current frozen **V5** model, what each
version changed and why, and the evaluation evidence behind that decision.
It is written for a developer joining the project who needs to understand
*what changed and why* without re-deriving it from git history.

**Current status: V5 is frozen and is the active pipeline** wired into the
Flask API (`v5_detector.py` / `V5Detector`, imported by `main.py`). The
original classical pipeline, V2, V3, and V4 are described below for context;
none of them are invoked by the running application.

## Final architecture (V5)

```
uploaded image
  -> frozen YOLO instance segmentation
  -> raster lesion mask (multiple instances kept separate, never unioned)
  -> ABCD + enhanced interpretable feature extraction (17 features total)
  -> frozen V5 logistic-regression decision model
  -> fixed threshold 0.25
  -> LOWER VISUAL CONCERN / ELEVATED VISUAL CONCERN
```

When YOLO detects more than one lesion instance in a photo, every instance
is kept and reported separately (`num_lesion_instances`,
`multi_lesion_detected` in the API response) -- they are never merged into
one mask. The highest-confidence instance is selected as the "primary"
instance and is the one actually scored; the others are shown to the user
(outlined, not analyzed) via `MultiLesionNotice.razor`, so a multi-lesion
photo never silently looks like only one spot was found.

## 1. Original pipeline (pre-YOLO, classical)

Historical description. The prototype's command-line runner and its lesion
segmentation, asymmetry and diameter scoring have been removed from the tree
(they remain in git history). What the running application still uses from
`Code/` is the preprocessing (vignette, denoise, bilateral filter, hair
removal) and the border and colour scoring, called by the V5 pipeline above.

```
image -> preprocessing/hair removal -> LAB/Otsu lesion segmentation
       -> original ABCD -> original rule-based concern logic
```

- **Preprocessing**: vignette removal, salt-and-pepper denoise, bilateral
  filter, then hair removal (blackhat + inpainting).
- **Segmentation**: LAB color-space distance from a sampled border-ring skin
  tone, Otsu-thresholded, morphologically cleaned, largest connected
  component kept.
- **A (asymmetry)**: crop around the lesion's mass centroid, compare the
  crop against its own horizontal and vertical mirror image; higher mismatch
  = more asymmetric.
- **B (border)**: contour circularity (`4*pi*area/perimeter^2`); border
  irregularity = `1 - circularity`.
- **C (color)**: pixelated LAB-distance coefficient of variation from a
  sampled skin-tone baseline, plus explicit pink/red, blue-gray, white, and
  black "dangerous color" pixel-fraction detection.
- **D (diameter)**: minimum enclosing circle, converted to millimeters using
  the *measured width of vellus (fine body) hair visible in the photo* as a
  physical scale reference -- falls back to "not measurable" if no hair is
  detected.
- **Decision**: a hardcoded rule -- count how many of A/B/C/D were flagged
  (each against its own fixed threshold); 2 or more flagged, or a single
  "dangerous color" exceeding 50% of the lesion, means HIGH; otherwise LOW.
  Not a trained model.

## 2. YOLO segmentation transition

The LAB/Otsu segmentation was replaced by a frozen Ultralytics YOLO instance
segmentation model because it produces substantially better lesion masks
(see Segmentation benchmark, below: mean IoU 0.44 -> 0.78 against official
ISIC ground truth on the same images). This is the single biggest
architecture change in the pipeline's history -- everything from V2 onward
uses YOLO for segmentation; only the decision layer changed after that.

## 3. V2 -- YOLO segmentation + revised ABCD

`revised_abcd/pipeline_v2.py`. Key revisions over the original:

- **PCA-aligned asymmetry**: the lesion mask is rotated to align with its
  own principal axis before the mirror-overlap comparison, instead of
  folding along the photo's fixed x/y axes -- removes a confound between
  lesion *orientation in the photo* and genuine shape asymmetry.
- **Segmentation-quality handling**: per-instance quality flags (fragmented
  mask, implausibly large area fraction, low YOLO confidence) computed
  alongside the mask.
- **Per-instance masks, single-image inference**: YOLO runs one image at a
  time (never batched), because Ultralytics' batch square-resize distorted
  masks for some images; every detected instance gets its own mask and its
  own full ABCD scoring pass -- no `np.max` union across instances.
- **Multi-lesion state**: images with more than one detected instance are
  flagged (`MULTI_LESION_AMBIGUOUS`) and excluded from aggregate accuracy
  metrics, since ISIC provides one diagnosis per image with no per-lesion
  localization -- there's no way to know which lesion a label refers to.
- **No-detection state**: images where YOLO finds nothing are flagged
  (`NO_DETECTION`) rather than silently scored as benign.
- **Removal of hair-based physical diameter conversion**: no defensible
  pixel-to-mm calibration exists for this data source (confirmed via the
  ISIC Archive metadata and image EXIF), so D is always pixels-only;
  `D_mm` is always `None` and never contributes to the decision.
- **Decision**: kept the *same* hardcoded rule as the original (2-of-4
  concerns, critical-color override) -- only the segmentation and A's
  alignment changed, not the decision logic.

## 4. V3 -- retuned asymmetry threshold

`pipeline_v3/decision.py`. The *only* change from V2: the asymmetry concern
threshold moved from 0.20 to 0.14 (selected via a development-only sweep).
Everything else -- segmentation, B/C/D computation, the 2-of-4-concerns
combination rule, the critical-color override -- is identical to V2.

## 5. V4 -- frozen logistic-regression decision model

`pipeline_v4/decision_model.py`. The first version to replace the
hand-tuned rule-count decision layer with a model trained on labeled data:
a scikit-learn `StandardScaler -> LogisticRegression` pipeline, fit on the
development cohort, frozen (never refit after evaluation).

**6 features, in order**: `A_value, B_circularity, C_value, D_px,
confidence (YOLO detection confidence), lesion_fraction (mask-area
fraction)`. **Threshold: 0.25** (selected via a development-only sweep,
maximizing balanced accuracy subject to both sensitivity and specificity
being at least 0.55).

## 6. V5 -- FINAL, FROZEN (current)

`pipeline_v5/`. V4's 6 features plus 11 additional interpretable features
identified via a controlled ablation study across asymmetry, border, color,
and geometry candidates (see `pipeline-testing/research`'s
`Evaluation_FeatureEngineering/` for the full study -- not copied into this
branch; see Evaluation section below for the results that justified V5).

### Exact 17 features, in exact model order

Pulled directly from `pipeline_v5/decision_model.py::V5_CONFIG["features"]`
(order matters -- the fitted `StandardScaler`'s `mean_`/`scale_` arrays are
positional, not name-keyed):

1. `A_value`
2. `B_circularity`
3. `C_value`
4. `D_px`
5. `confidence`
6. `lesion_fraction`
7. `color_entropy`
8. `lab_a_std`
9. `lab_b_std`
10. `red_fraction`
11. `bluegray_fraction`
12. `dark_fraction`
13. `skin_contrast`
14. `solidity`
15. `turning_angle_std`
16. `eccentricity`
17. `D_px_normalized`

Features 1-6 are unchanged from V4 (produced by `revised_abcd/pipeline_v2.py`,
which is itself unchanged since V2). Features 7-17 are new in V5
(`pipeline_v5/feature_extraction.py::extract_v5_new_features`): color
distribution statistics (7-13), border-shape measures (14-15), and
scale-invariant geometry (16-17).

### Model loading and scoring

- `pipeline_v5/decision_model.py::load_frozen_pipeline()` unpickles
  `pipeline_v5/frozen_model.pkl` -- a fitted
  `sklearn.pipeline.Pipeline(StandardScaler, LogisticRegression)`, fit once
  on the full development cohort, never refit after evaluation (including
  after locked-test results were seen).
- `v5_predict_from_row(row, pipeline)` builds the 17-value feature vector in
  the exact order above, calls `pipeline.predict_proba(...)`, and compares
  the result to the threshold.
- **Threshold: 0.25** (unchanged from V4; selected on V5's own
  development-only cross-validated out-of-fold probabilities, using the
  same selection rule as V4 -- maximize balanced accuracy subject to
  sensitivity and specificity both being at least 0.55). Coincidentally
  equal to V4's threshold, not chosen to match it.
- **The model's output is a decision score, not a melanoma or cancer
  probability.** It is a calibrated-ish logistic-regression output over a
  screening-relevant feature set, evaluated against a labeled research
  cohort -- not a clinical risk estimate for an individual.
- **YOLO's detection confidence (feature 5, `confidence`) is a detection
  confidence, not a melanoma probability** -- it measures how sure YOLO is
  that it found a lesion-shaped object, and is one of 17 inputs to the
  decision model, never surfaced to the user as a risk number on its own.
- **No further threshold, feature, or model tuning should be performed
  using the locked test set.** The locked test exists specifically as a
  one-time, held-out check (see Evaluation, below) -- reusing it to tune
  anything invalidates that check's purpose. Any future retuning belongs on
  the development cohort, with a fresh one-time locked-test confirmation
  afterward, exactly as was done for V5 itself.

## Evaluation

Full evaluation code, per-image results, and larger research artifacts live
on the `pipeline-testing/research` branch (`Evaluation_SegmentationBenchmark/`,
`Evaluation_DevComparison/`, `Evaluation_FeatureEngineering/`,
`Evaluation_V5Candidate/`) -- intentionally not copied into `user`. This
section is the concise, citable summary.

**These are prototype research results on specific ISIC-derived evaluation
cohorts, not clinical validation.** They describe how well this pipeline
performs on the labeled data it was evaluated against, not how it performs
on the general population or in any individual case.

### Segmentation quality: YOLO vs. original LAB/Otsu

Independent benchmark, 100-image reproducible sample from the official
ISIC 2018 Challenge Task 1 test set (expert-annotated ground truth, disjoint
from the classification cohort below):

| Method | Mean IoU | Mean Dice |
|---|---|---|
| Original LAB/Otsu | 0.442 | 0.542 |
| YOLO | 0.783 | 0.852 |

YOLO won 91 of 100 images by IoU (paired comparison, same images, same
ground truth).

### Classification: development cohort (2,696 images, patient/lesion-grouped, balanced accuracy at each version's own threshold)

| Version | Balanced accuracy |
|---|---|
| Original classical (LAB/Otsu + original ABCD) | 0.569 |
| V2 | 0.610 |
| V3 | 0.614 |
| V4 | 0.718 |
| **V5** | **0.722** |

### Classification: locked test (615 evaluable images, run once)

The final, one-time, held-out confirmation for V5:

| Metric | Value |
|---|---|
| Evaluable cases | 615 |
| Sensitivity | 0.710 |
| Specificity | 0.713 |
| Precision | 0.417 |
| F1 | 0.526 |
| Accuracy | 0.712 |
| Balanced accuracy | 0.712 |
| ROC-AUC | 0.798 |
| Confusion matrix | TP 98, TN 340, FP 137, FN 40 |

## Known integration TODOs

These are current, open issues in the web application integration --
documented here rather than silently presented as resolved. **None of them
require or should prompt any change to the frozen model, its 17 features,
the 0.25 threshold, or the YOLO segmentation model** -- they are UI/response
wiring issues to verify and fix in the surrounding application code only.

- ~~**Results page sometimes shows "Visual concern not available" instead of
  LOWER/ELEVATED for a successful V5 result.**~~ **Fixed (2026-09-23).** Root
  cause: `RiskBandPanel`'s and `RiskScoreBadge`'s `string?` concern parameters
  were bound as `OverallVisualConcern="Results.OverallVisualConcern"` --
  missing the `@` prefix. Blazor can't tell a literal string apart from a C#
  expression for `string`-typed component parameters, so the component
  received the literal 29-character text "Results.OverallVisualConcern" as
  its value on every single check, not the real verdict -- confirmed by
  inspecting the compiled Razor output (`AddComponentParameter(..., "Results.
  OverallVisualConcern")`, a quoted literal, vs. `Score`'s correctly-compiled
  `TypeCheck<Double>(Results.RiskScore)`). This was never a race condition or
  a reload-vs-fresh issue: it reproduced 100% of the time, everywhere the app
  shows a risk verdict. Fixed at all 9 call sites (CheckResultsStep, Results
  page, HistoryCard, HistoryComparePanel, RecentCheckRow, SpotCard,
  SpotDetailHeader, SpotTimeline, SpotTimelineEntry) by adding `@`.
- **Multi-lesion rendering needs verification.** Confirm all YOLO instances
  the detector returns are consistently reflected in the browser (overlay
  image, instance count, notice text) across fresh-analysis and
  saved/reloaded views.
- **NO_DETECTION needs final browser verification** to ensure no stale
  decision score or ABCD evidence from a previous state is displayed
  alongside the "no lesion located" message.
- **Diameter UI must show pixel and/or relative-to-image size only.** Verify
  no surface (including any newer chat/explanation feature) claims or
  implies millimeters without a physical scale reference.
- **Review remaining medical/risk wording app-wide** so the prototype does
  not present itself as a melanoma diagnosis anywhere, including newer
  features added concurrently with this integration (e.g. a chat feature
  that summarizes a check's result should describe V5's own LOWER/ELEVATED
  verdict, not re-derive a separate "risk band" label from the raw score).
