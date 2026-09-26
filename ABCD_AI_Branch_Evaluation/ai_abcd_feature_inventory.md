# `ai-abcd` Branch — Inspection & Feature Inventory

Read-only inspection of `origin/ai-abcd` (accessed via `git show` and a detached
`git worktree` at `../ai-abcd-worktree`, never merged/checked out onto `user`). This
document is written BEFORE any evaluation numbers are produced, per instructions.

## What `ai-abcd` actually is

The branch's new work lives entirely in one new top-level folder: `bulk_analysis/`.
Everything else on the branch (`pipeline_v5/`, `revised_abcd/`, `Code/`,
`MelanomaDetection/`) was **diffed file-by-file against `user`'s current versions and
found byte-identical** — including `pipeline_v5/frozen_model.pkl` (SHA-256 match) and
the YOLO checkpoint. `bulk_analysis/` contains two independent tools, both of which
explicitly reuse the app's existing segmentation/measurement code rather than
reimplementing anything:

### 1. `run_bulk_analysis.py` — NOT a different pipeline

Its own README states it directly: "reuses the app's model code by importing it
directly... you always get the exact same scoring the app gives, not a
reimplementation." Confirmed by inspection: it imports `pipeline_v5/decision_model.py`
and `revised_abcd/pipeline_v2.py` unmodified. **This is literally frozen V5, wrapped in
a folder-batch CLI.** Running it on our development cohort would reproduce EXP0's
already-known numbers exactly (ROC-AUC 0.8008, balanced accuracy 0.7222, threshold
0.25) — it was not separately re-executed, since doing so would burn compute to
reproduce numbers we already have and have already verified. This is reported plainly
rather than silently treated as "ai-abcd's result."

### 2. `run_clinical_abcd.py` / `clinical_abcd.py` — the actual new contribution

A **rule-based, non-learned** "clinical decision" layer. It reuses the exact same
underlying measurements (`A_value` from `revised_asymmetry.score_asymmetry_revised`;
`border_irreg`/`B_circularity` from the unchanged `Code/MelanomaDeterminingStuff/
border.py::score_border`; `n_significant_defects` from `revised_border.
score_border_experimental`; `color_cv`/`C_value` plus the four color-fraction values
from the unchanged `Code/MelanomaDeterminingStuff/color.py::score_color`) — all
confirmed identical to what `user`'s pipelines already compute and cache — and applies
a **different, independently-chosen set of fixed thresholds** to them, with a
**different output shape**: one concern flag per A/B/C/D criterion, not a combined
score or verdict.

## How its features are calculated (exact thresholds, from `clinical_abcd.py`)

| Criterion | Input(s) (same underlying quantity as our pipelines) | Rule | Concern if |
|---|---|---|---|
| **A** | `A_value` (identical formula/code to our `A_value`) | Two-tier instead of V5's single 0.20 cutoff | `A_value > 0.15` = mild; `> 0.35` = marked (both count as `concern=True`) |
| **B** | `border_irreg` (= our `B_circularity`, same formula) + `n_significant_defects` (= our `B_experimental_n_defects`, already cached, not part of the frozen 17 features) | OR of two sub-rules | `border_irreg > 0.55` **OR** `n_significant_defects >= 3` |
| **C** | `color_cv` (= our `C_value`, same formula) + the 4 color-fraction values (= our `red_fraction`/`bluegray_fraction`/`white_fraction`/`dark_fraction`, same `score_color()` outputs) | OR of three sub-rules | ≥3 distinct colors present (any fraction ≥5%, +1 for baseline brown) **OR** `color_cv > 0.35` **OR** any single color ≥20% of lesion |
| **D** | `D_px` (identical formula) | N/A — always | `concern` is literally the string `"N/A"`, always, for every image. No diameter cutoff is ever applied. |

**No feature is calculated differently at the measurement level** — every raw number
`clinical_abcd.py` consumes is byte-identical in formula and code to what our
`dev_feature_table_v2.csv` already contains. Only the **decision thresholds on top of
those numbers** differ from V5's own frozen thresholds (V5's `A_concern`/
`B_circularity_concern`/`C_concern` — which, per `revised_abcd/pipeline_v2.py`'s own
docstring, are separately frozen and not even the thresholds the logistic-regression
model uses to make its ELEVATED/LOWER call — V5's decision is a 17-feature regression,
not a threshold rule at all).

## Segmentation / preprocessing

**Identical.** Same YOLO checkpoint (SHA-256-verified), same
`run_single_image_inference` (single-image, `conf=0.25`, unbatched), same
preprocessing chain (`preprocess_image`: vignette removal → denoise → bilateral filter
→ hair removal). Confirmed via file diff, not assumed.

## Classifier / decision rule

**`clinical_abcd.py` has no classifier, no trained model, and no combined
decision at all.** `judge_instance()` returns four independent
`{concern, label, ...}` dicts (A/B/C/D) and stops — there is no code anywhere in this
module or `run_clinical_abcd.py` that combines them into one score, one label, or one
TP/TN-style verdict. The module's own docstring frames this as intentional: "no single
number... one decision per criterion, the way a clinician's note would read." Its
thresholds are stated explicitly as **not fit/tuned on any dataset** — "reasoned"
mappings from general dermatology teaching onto this pipeline's specific outputs, with
an explicit disclaimer that "none of these cutoffs come from a validated diagnostic
study of THIS pipeline's exact pixel-based metrics."

**Practical consequence for evaluation** (flagged now, before any numbers are run):
producing a single TP/TN/FP/FN table for `clinical_abcd.py` requires an aggregation
rule across A/B/C that **does not exist in `ai-abcd`'s own code**. Any such rule used
in this evaluation is an addition made *for evaluation purposes only*, clearly
labeled as not part of `ai-abcd`'s design — per instructions, its A/B/C/D judgment
logic itself is not rewritten or re-thresholded to make this possible; only a
transparent, clearly-separate aggregation step is added on top, and results are
reported under more than one reasonable aggregation choice so no single arbitrary
choice is presented as "the" ai-abcd verdict. See `ai_abcd_performance.csv` for exact
detail.

## Threshold used

- `run_bulk_analysis.py`: 0.25 (frozen V5's own threshold — unchanged, since it's the
  same model).
- `clinical_abcd.py`: no single threshold — four independent per-criterion cutoffs
  (0.15/0.35 for A; 0.55 and 3 for B; 0.05/0.20/3 for C; none for D), listed in the
  table above. Explicitly not swept/tuned on any development or validation split —
  they are fixed a priori from literature, per the module's own docstring.

## Continuous score availability

`clinical_abcd.py` produces **booleans** (`True`/`False`/`None`) per criterion, never
a continuous probability or risk score. **A true ROC-AUC cannot be fairly computed
from it** — this is stated plainly in `ai_abcd_performance.csv`/`.md` rather than
worked around. An ordinal "how many of A/B/C fired" count (0–3) is reported
separately as a supplementary, clearly-labeled *pseudo-score* for a pseudo-AUC — not
presented as equivalent to V5/Improved's calibrated logistic-regression probabilities.

## Preprocessing/data differences that could affect a "fair" comparison

None found in the pipeline code itself (everything upstream of the judgment layer is
byte-identical). The one real difference worth flagging: `clinical_abcd.py`'s
`judge_border`/`judge_color` consume `n_significant_defects` and the individual color
fractions, which **are cached in `Evaluation_FinalTargeted/Cohort/cohort_v2_results.csv`
but are NOT part of `dev_feature_table_v2.csv`'s 17-model-feature columns** — they had
to be joined in from the cohort results file for this evaluation (same population,
same cached values, no recomputation) rather than being already present in the
Improved-Experimental feature table. This is a data-plumbing detail, not a pipeline
difference.
