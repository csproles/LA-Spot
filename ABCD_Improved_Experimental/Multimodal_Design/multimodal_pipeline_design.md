# Multimodal (CNN embedding + structured ABCD) Classification — Design Proposal

**Status: proposal only. Nothing in this document has been executed.** No CNN was
run, no new dependency was installed, no image was processed. This is a
development-only design, grounded in this project's actual data volumes, existing
methodology, and current environment — written for review before any implementation
begins. The locked 615-image test set is not referenced anywhere below and must not
be until a development-only candidate is selected and explicitly approved, exactly
as with every other experimental track in `ABCD_Improved_Experimental/`.

## Grounding facts (checked, not assumed)

- Development cohort: **n=2090** (1451 benign / 639 malignant), same `group_id`
  patient/lesion grouping used throughout this project's GroupKFold(5) work.
- Current best structured-only baseline: **Improved ABCD FINAL**, 14 features,
  balanced accuracy 0.7344, ROC-AUC 0.8016, CV std 0.0096 (`ABCD_Improved_Experimental/
  final_recommendation.md`). This is the number the multimodal pipeline needs to beat.
- Environment (`.venv`): `torch 2.6.0+cpu`, `torchvision 0.21.0+cpu`, `timm 1.0.29`
  are **already installed** (torch is already a live-app dependency via Ultralytics/
  YOLO). **No CUDA** — everything below is CPU-only. `xgboost`, `lightgbm`,
  `catboost`, `shap` are **not installed** and would need to be added.
- Original images are reachable at the same `image_path`s already used throughout
  this project (`Evaluation_FinalTargeted/Cohort/development_manifest.csv`); cached
  YOLO masks already exist for the full development cohort.

## 1. Recommended CNN backbone(s)

**Start with a single backbone, not an ensemble** — n=2090 is small enough that
model-selection noise across multiple backbones would be hard to distinguish from
real signal, and the goal right now is "does adding a CNN modality help at all,"
not "which of five CNNs is marginally best."

- **Primary recommendation: EfficientNet-B0**, ImageNet-pretrained (via `timm`,
  already installed). Rationale: strong accuracy/parameter-efficiency tradeoff
  (5.3M params — much less prone to overfitting on 2090 images than a large model),
  widely validated specifically on dermoscopy/ISIC classification in the published
  literature, and cheap enough to run a full frozen-embedding extraction pass on CPU
  in minutes rather than hours.
- **Secondary/fallback: ResNet-50**, ImageNet-pretrained. More parameters (25.6M) and
  a slightly higher overfitting risk on this dataset size, but the most
  well-understood architecture for Grad-CAM-style explainability (the original
  Grad-CAM paper and most follow-on tooling use ResNet as the reference case) — a
  reasonable second choice specifically if explainability tooling proves easier to
  get working reliably on it than on EfficientNet.
- Not recommended for a first pass: anything larger (EfficientNet-B4+, ResNet-101+,
  ViT-Base) — more capacity than 2090 images can responsibly fine-tune, and frozen
  large-backbone embeddings mostly just add irrelevant dimensionality for the
  downstream classifier to overfit on.

## 2. Where to extract the embedding

- **Input**: the lesion **crop**, not the full original photo — take the YOLO mask's
  bounding box (already cached), pad by a fixed margin (e.g. 15–20% of the box's
  larger dimension) so peri-lesional skin context is retained, then resize to the
  backbone's expected input (224×224) with its standard ImageNet preprocessing
  (mean/std normalization per `timm`'s pretrained-config for the chosen backbone).
  **This is a deliberate, secondary benefit worth flagging**: cropping to the lesion
  directly counters the photo-framing/zoom confound already documented for
  `lesion_fraction`/`D_px_normalized` in the main audit — the CNN never sees how much
  of the original frame the lesion occupied, only a normalized, resolution-fixed crop
  around it.
- **Do not mask out the background inside the crop** for the first experiment — keep
  the full rectangular crop (lesion + immediate surrounding skin), not a
  mask-multiplied cutout. Border-transition sharpness and peri-lesional erythema are
  themselves diagnostically relevant and a masked-out image would create an
  artificial hard edge the network has never seen in its ImageNet pretraining.
  (Testing a masked variant is a reasonable Stage-2+ ablation, not a Stage-1 default.)
- **Embedding layer**: the backbone's standard post-global-average-pool feature
  vector, taken *before* its original ImageNet classification head —
  1280-dimensional for EfficientNet-B0, 2048-dimensional for ResNet-50. In `timm`
  this is `model.forward_features(x)` followed by the model's own pooling
  (`model.global_pool` / `model.forward_head(x, pre_logits=True)`), not a
  hand-picked intermediate conv layer — using the layer the backbone was actually
  trained to summarize into is the standard, best-validated choice for
  transfer-learning feature extraction.
- **Frozen vs. fine-tuned, staged**:
  - **Stage 1 (do this first): fully frozen backbone**, ImageNet weights, used purely
    as a fixed feature extractor. No training occurs on the backbone at all — every
    embedding can be computed **once**, offline, for all 2090 images, with zero
    leakage risk from the backbone itself (it was never fit on any of our patients'
    data, so there is nothing to leak across folds).
  - **Stage 2+ (only if Stage 1 shows real signal): partial fine-tuning** of the
    backbone's last block/stage at a low learning rate, refit **separately inside
    each GroupKFold training fold** (5 independently fine-tuned copies of the last
    block) — expensive on CPU (see Training strategy, below) and only worth pursuing
    once frozen embeddings have demonstrated the CNN modality adds something the
    structured features don't already capture.

## 3. Normalization and concatenation

- **CNN embedding**: standardize (zero mean, unit variance) **fit on the training
  fold only**, applied to the validation fold — the same per-fold-fit discipline
  `cv_harness.py` already uses for the structured `StandardScaler`. Optionally
  L2-normalize each embedding vector before standardization (common practice in
  transfer-learning fusion setups; prevents a few high-magnitude dimensions from
  dominating).
- **Dimensionality mismatch is the central design risk here**: 1280 CNN dimensions
  vs. 14 structured features is a ~90:1 imbalance. Concatenating raw would let the
  CNN modality numerically swamp the structured features for most downstream
  classifiers, undermining the explicit goal of keeping ABCD features
  independently attributable.
  - **Mitigation (recommended for Stage 1)**: reduce the CNN embedding via **PCA
    fit on the training fold only** (never on the pooled 2090-image set — that would
    leak validation-fold structure into the transform) down to roughly 64–128
    components, chosen by an explained-variance target (e.g. 90–95%) re-derived
    per fold, not a single global number. This brings the two modalities to a more
    comparable order of magnitude (64–128 CNN dims vs. 14 structured dims) while
    keeping the downstream classifier's effective parameter count sane relative to
    ~1670 training-fold samples.
  - A trainable projection head (a small linear layer collapsing 1280→128, learned
    jointly) is a reasonable Stage 2+ alternative once frozen PCA has been tried,
    but adds real training complexity (another leakage-sensitive fitted component)
    that isn't justified before a simpler baseline is established.
- **Concatenation**: simple vector concatenation of (PCA-reduced CNN embedding) ⧺
  (standardized 14-feature structured vector) into one feature vector per image.
  This is **feature-level early fusion**, not a jointly-trained end-to-end network —
  the right choice here specifically because it keeps every structured feature
  individually addressable in the downstream classifier's own feature-importance
  output, which is the explainability requirement driving this whole design.

## 4. Downstream classifier

- **Try first: XGBoost** (or LightGBM as an equally reasonable first choice — pick
  whichever installs more easily in this environment; both have mature, fast, exact
  SHAP support via `TreeExplainer`). Rationale: gradient-boosted trees handle
  concatenated mixed-scale/mixed-modality vectors well without requiring careful
  per-feature scaling assumptions, are robust to the redundant/correlated dimensions
  a PCA-reduced CNN embedding will contain, and have the best-supported
  explainability tooling of the three candidates named in the request.
- **CatBoost**: reasonable second candidate, particularly strong if any categorical
  structured features are added later (e.g. an encoded `quality_flags` string) —
  not the first choice here since the current structured feature set is entirely
  numeric.
- **Also run, as a mandatory sanity-check baseline, not an afterthought**: plain
  `StandardScaler + LogisticRegression` on the same concatenated (PCA-reduced CNN +
  structured) vector, using the *exact* existing `cv_harness.py` harness. This costs
  almost nothing to add and answers a question a tree-model-only comparison can't:
  is any observed gain coming from the new CNN modality itself, or from switching to
  a more flexible classifier family? Without this row, "XGBoost beat frozen
  V5/Improved" would be confounded between two changes at once.
- Given ~1670 training-fold samples vs. potentially 80–140 concatenated features,
  regularize aggressively from the start: shallow trees (`max_depth` 3–4), meaningful
  L1/L2 (`reg_alpha`/`reg_lambda`), `subsample`/`colsample_bytree` < 1, and
  early stopping against an *inner* validation split (see below) rather than a large
  fixed `n_estimators`.

## 5. Training/validation strategy — leakage discipline

- **Outer evaluation loop stays identical to the existing methodology**: the same
  `GroupKFold(5)` on `group_id`, the same population, so results are directly
  comparable to and appendable into `pipeline_performance_comparison.csv`.
- **The frozen backbone (Stage 1) introduces no leakage risk by construction**: since
  it is never fit on any of our images (ImageNet weights only), embeddings for all
  2090 images can be extracted **once, offline**, before any fold splitting happens
  — there is nothing patient-specific for a frozen backbone to "leak." This is a
  genuine advantage of starting with Stage 1 over jumping straight to fine-tuning.
- **Everything fit on our data must still be refit per fold**, exactly like the
  existing 17/14-feature logistic regressions:
  - PCA on the CNN embedding → fit on that fold's training rows only.
  - `StandardScaler` on the structured features → fit on that fold's training rows
    only (already the existing pattern).
  - The downstream XGBoost/LightGBM/LogisticRegression → fit fresh per fold on that
    fold's training rows only, OOF predictions collected on each held-out fold.
- **Hyperparameter selection must not be tuned directly against the outer GroupKFold
  OOF result** — repeatedly checking outer-fold balanced accuracy while adjusting
  XGBoost hyperparameters is itself a (subtle, easy to miss) form of overfitting to
  the evaluation procedure. Recommended fix: carve out an **inner** validation split
  (either a nested GroupKFold within each outer training fold, or a single held-out
  dev-tuning slice of the training folds, grouped the same way) used only for
  early-stopping/hyperparameter search; the outer GroupKFold(5) OOF numbers are
  computed only with hyperparameters already fixed from the inner loop, and are used
  purely for final reporting/comparison, mirroring how V5's own threshold was
  selected on development-only OOF probabilities and never re-tuned after the fact.
- **If/when Stage 2 fine-tuning is attempted**, it must fine-tune a fresh copy of the
  backbone's last block inside each of the 5 outer training folds independently (5
  separately fine-tuned models) — this is the expensive part: on CPU, even a small
  fine-tuned slice over ~1670 images for several epochs, five times over, is a real
  wall-clock cost (plausibly hours, not minutes) and should only be attempted once
  Stage 1's frozen-embedding results justify it.

## 6. Metrics to compare against Improved ABCD FINAL

Reuse the exact metric set already established, so every new row can be appended
directly to `pipeline_performance_comparison.csv` without inventing a parallel
reporting format: **TP/TN/FP/FN, accuracy, balanced accuracy (primary comparison
metric, consistent with every prior experiment in this project), sensitivity,
specificity, precision, F1, ROC-AUC, CV mean/std balanced accuracy across the same 5
folds.**

Additional checks specific to this comparison:
- **Paired bootstrap CI on the balanced-accuracy/ROC-AUC delta** between the
  multimodal candidate's OOF predictions and Improved FINAL's OOF predictions on the
  same images — reusing the exact `paired_bootstrap_ci` pattern already implemented
  in `Evaluation_FeatureEngineering/run_ablation_cv.py` and
  `Evaluation_V5Candidate/LockedTest/run_locked_test_evaluation.py`. This is how
  every prior "is this improvement real" question in this project has been answered,
  and should be answered the same way here rather than trusting a single point
  estimate.
- **Calibration**: a reliability diagram and Brier score for the final probability
  output. This matters specifically because gradient-boosted-tree raw probabilities
  are frequently poorly calibrated even when their *ranking* (AUC) is good, and a
  medical-use setting cares about the probability number itself, not just
  discrimination — worth checking whether Platt scaling / isotonic calibration
  (fit per-fold, same leakage discipline as everything else) measurably improves
  calibration without hurting discrimination.

## 7. Explainability for both modalities

- **Structured features**: SHAP `TreeExplainer` on the fitted XGBoost/LightGBM model
  gives exact, fast, per-image, per-feature contributions for every one of the 14
  structured features — a direct, richer upgrade over the current logistic
  regression's global standardized coefficients (Part 1/2 of the main audit), since
  SHAP gives a *per-patient* explanation ("for this case, `solidity` contributed
  −0.08 to the score, `turning_angle_std` contributed +0.11...") rather than only a
  population-level coefficient. This is directly usable for patient-facing
  explanation text, consistent with this project's existing policy against
  presenting raw model internals as a diagnosis.
- **CNN features — two complementary pieces, not one**:
  1. **Modality-level SHAP aggregation**: sum SHAP contributions across all CNN-
     embedding dimensions (post-PCA) vs. all structured-feature dimensions, per
     image, to report an interpretable split like "visual/deep-learning signal
     contributed ~40% of this case's score, structured ABCD features contributed
     ~60%." This is honest about what's achievable: individual PCA component SHAP
     values are not themselves human-meaningful, but the aggregate split is.
  2. **Grad-CAM (or Grad-CAM++) on the CNN backbone**, applied to the lesion crop,
     producing a heatmap of which image regions most influenced the backbone's
     embedding. This explains the CNN backbone's own activations, not literally the
     downstream XGBoost split decision, but combined with the modality-level SHAP
     split above gives a reasonably complete two-part explanation ("here's roughly
     how much the visual signal mattered, and here's roughly where in the image it
     was looking"). Grad-CAM output fits naturally alongside the existing
     `VisualReview` contour/mask-overlay materials already established in this
     project's audit workflow — same presentation pattern, new content.
  3. **Honest limitation to state explicitly, not gloss over**: because the CNN
     embedding is PCA-reduced before concatenation, there is no direct, exact
     traceability from a specific SHAP-flagged PCA component back to a specific
     image region — only Grad-CAM (applied to the backbone directly, independent of
     the downstream classifier) can answer "where was the CNN looking." Any
     patient-facing explanation combining both should say so plainly rather than
     implying a false one-to-one mapping between the two explainability tools.

## Staged experimental plan

| Stage | What | Cost/risk | Gate to proceed |
|---|---|---|---|
| **0** | Build lesion crops (cached YOLO boxes + padding) for all 2090 dev images | Low (uses already-cached masks) | — |
| **1** | Extract frozen EfficientNet-B0 embeddings for all 2090 images (one offline pass, no fitting, no leakage) | Low–moderate (CPU-only, but one-time, no training) | — |
| **2** | Sanity-check baselines: (a) LogisticRegression on structured-only (=reproduce Improved FINAL, already have), (b) LogisticRegression on PCA-reduced CNN-embedding-only, (c) LogisticRegression on concatenated — same cheap harness, isolates whether the CNN modality has ANY signal before introducing a new classifier family | Low (reuses existing `cv_harness.py`) | Only proceed to Stage 3 if (b) or (c) shows real signal (outside CV-std noise) vs. structured-only |
| **3** | Swap in XGBoost/LightGBM on the concatenated vector, with inner-loop hyperparameter selection | Moderate (new dependency, more moving parts) | Only proceed to Stage 4 if this beats Improved FINAL by more than its own CV std, confirmed by paired bootstrap CI |
| **4** | Partial backbone fine-tuning (5 independently fine-tuned copies, one per outer fold) | High (CPU wall-clock cost, more leakage-discipline surface area) | Only attempt if Stage 1–3's frozen-embedding results are close-but-not-quite, i.e. justified by evidence |
| **5** | Explainability tooling (SHAP + Grad-CAM) on the selected best configuration | Moderate | Applied once a development-only candidate is actually selected |

**Explicit stop point**: after the best development-only configuration is identified
through Stages 0–5, this design (like every other experimental track in this
project) stops there and is reported as a development-only candidate pending
approval — the locked 615-image test set is not touched until you explicitly say so.

## New dependencies this would require

`xgboost` (or `lightgbm`), `shap` — not currently installed in `.venv`. `torch`/
`torchvision`/`timm` are already present. No GPU is available in this environment;
everything above is scoped assuming CPU-only execution, which is the main reason
Stage 1 (frozen, one-time, no backprop) is proposed before Stage 4 (fine-tuning,
repeated backprop across folds) rather than jumping straight to end-to-end training.
