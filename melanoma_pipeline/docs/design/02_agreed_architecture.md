# Agreed Architecture — As Built

This is the definitive, as-implemented record of the multimodal (CNN + validated
ABCD) melanoma pipeline in `melanoma_pipeline/`, capturing every decision made after
`01_original_design_proposal.md` was written. Read that file first for the full
reasoning (backbone tradeoffs, fusion strategy, explainability plan); this file
records what was actually *decided and built*, and why it sometimes differs from
that original proposal.

## Context that shaped the final decisions

- **Scale**: ~402,000 images (ISIC 2024 / SLICE-3D-scale), extremely class-imbalanced
  (a small fraction malignant). This is far larger than the n=2090 development cohort
  the original ABCD audit and "Improved ABCD FINAL" baseline were built on.
- **Compute**: GPU resources are available for this project (confirmed explicitly),
  which changes the calculus from the original proposal's CPU-only default —
  fine-tuning, not just frozen-embedding extraction, is in scope.
- **Segmentation**: only ~2–3k images have (or will have) pixel-level masks; the
  existing frozen YOLO checkpoint (already benchmarked at IoU 0.783 / Dice 0.852
  against expert ISIC ground truth) is reused as-is rather than replaced, pending a
  fresh benchmark on the actual new data source if/when it's needed.

## CNN backbone: ConvNeXt-Base (`convnext_base.fb_in22k_ft_in1k`)

The original proposal recommended EfficientNet-B0 under a CPU-only assumption.
With GPU confirmed available and ~402k images in scope, the recommendation moved to
**ConvNeXt-Base**, specifically because:
- Fine-tuning (not just frozen extraction) is now feasible, and ConvNeXt-Base's main
  strength — using large-scale pretraining well (ImageNet-22k-pretrained checkpoints,
  via `timm`) — only pays off once you can actually fine-tune.
- 402k images is enough scale to justify the larger capacity (88M params vs. B0's
  5.3M), provided class imbalance is handled deliberately (it is — see focal loss
  and `scale_pos_weight`, below).
- EfficientNetV2-M was the runner-up (faster to train, lighter on GPU memory) and
  remains a reasonable fallback if VRAM or iteration-speed constraints bite in
  practice — swapping is a one-line change to `config.MODEL_NAME` (and confirming
  `config.EMBEDDING_DIM` matches the new backbone's actual feature dimension, per
  the README's verification command).

**Fine-tuning strategy actually implemented: full fine-tuning, no layer freezing**
(`train_cnn.py::MelanomaCNN` — every backbone parameter is trainable). This is a
step beyond the original proposal's staged "frozen first, fine-tune only if
justified" plan — full fine-tuning was chosen directly given confirmed GPU access
and the ~402k-image scale, which together make the overfitting risk that motivated
staging much smaller than it was under the original n=2090/CPU-only assumption.

## Downstream classifier: CatBoost (not XGBoost/LightGBM)

The original proposal suggested trying XGBoost/LightGBM first, with CatBoost as a
secondary candidate. **CatBoost is what was actually specified and implemented**,
with `scale_pos_weight` computed per-fold from that fold's own class ratio to handle
the severe malignant/benign imbalance. `shap.TreeExplainer` works identically well
against CatBoost's tree ensembles as it would against XGBoost/LightGBM, so the
explainability plan (SHAP, see below) is unaffected by this choice.

## Fusion: PCA-reduced CNN embedding ++ validated ABCD features, per fold

- **CNN embedding**: 1024-dim (ConvNeXt-Base's penultimate pooled feature vector,
  via `timm.create_model(..., num_classes=0)` — confirm this dimension empirically
  with the command in the README rather than trusting this number blindly).
- **ABCD/structured features**: the real, validated V5 feature set — NOT a
  simplified/placeholder extractor. `features.py` reuses this project's existing,
  frozen pipeline exactly as `MelanomaDetection/MelanomaDetection.Python/
  v5_detector.py::V5Detector.process_image` calls it: YOLO segmentation
  (`revised_abcd.pipeline_v2.process_image`) → the unchanged base-6 features
  (`A_value, B_circularity, C_value, D_px, confidence, lesion_fraction`) →
  `pipeline_v5.feature_extraction.extract_v5_new_features`'s 11 additional features.
  17 features total, in the exact order `pipeline_v5.decision_model.V5_CONFIG
  ["features"]` defines — **the feature count is derived from that live config
  object at import time (`features.ABCD_FEATURE_NAMES` / `N_ABCD_FEATURES`), not
  hardcoded**, so this module can't silently drift out of sync with the production
  feature set. An earlier scaffold version of `features.py` used a simplified,
  from-scratch Otsu-threshold segmentation + 11 generic OpenCV features instead —
  that version has been fully replaced; see `features.py`'s own module docstring
  for the reasoning (the Otsu approach was independently benchmarked at IoU 0.442
  vs. YOLO's 0.783 in this project's own prior audit, i.e. known to be worse).
- **Fusion mechanism**: PCA-reduce the CNN embedding (`config.PCA_N_COMPONENTS`,
  default 128, clipped to `min(configured, n_train_samples, n_features)` per fold)
  down toward the same order of magnitude as the 17 ABCD features, THEN concatenate.
  **The PCA is fit exactly once per fold, on that fold's training embeddings only**,
  and applied (transform-only) to that fold's validation embeddings —
  `train_catboost.py::fit_transform_pca`. This was called out explicitly as a
  requirement (keep PCA and other learned preprocessing within training folds) and
  is the same per-fold-fit discipline every other learned component in this
  pipeline already follows (the CNN itself is fine-tuned fresh per fold in the same
  sense — each fold's `MelanomaCNN` instance only ever sees that fold's training
  images; CatBoost is fit fresh per fold too).

## Metric: `utils.compute_paauc` — the corrected, verified ISIC-2024 reference construction

Went through three iterations before landing on the correct implementation (full
history is in `utils.py`'s own docstring, not repeated here):
1. First version computed the wrong ROC region entirely (low-FPR instead of
   high-TPR).
2. Second version computed the right region but normalized incorrectly — it
   averaged TPR over the kept region's own (classifier-dependent) FPR width rather
   than computing a true partial-AUC integral, and could score above the
   theoretical maximum of 1.0 for a strong classifier.
3. **Third version (current)** implements the actual ISIC reference construction:
   flip both labels and scores (`y_flipped = 1 - y_true`, `scores_flipped =
   1 - y_score`), compute `roc_curve` on the flipped problem, truncate at
   `max_fpr = 1 - MIN_TPR` with linear boundary interpolation, integrate via
   `sklearn.metrics.auc`. Returns the **raw** partial-AUC area by default (this is
   literally what the ISIC-2024 Kaggle leaderboard reports, typically ~0.15–0.20 for
   a competitive model — NOT rescaled to [0,1]); pass `normalize=True` for a
   [0,1]-scaled version if that's more convenient for a specific comparison.
   Verified by direct computation (not just derived on paper) against three known
   reference points — perfect classifier (raw 0.1999…, normalized 1.0), random/
   diagonal classifier (raw ≈0.02, normalized ≈0.10, converging as sample size
   grows), and a completely reversed classifier (raw 0.0) — plus tied-score and
   exact-boundary edge cases. `config.MIN_TPR = 0.80` (renamed from an earlier,
   incorrect `FPR_THRESHOLD` constant).

## Explainability: SHAP now, Grad-CAM documented but not (yet) implemented

`explain.py` (new — not part of the original 8-file scaffold spec) implements the
structured-feature and modality-level SHAP explainability described in the original
proposal's Section 7:
- `shap.TreeExplainer` on the fitted CatBoost model (exact, not sampled/approximate)
  gives per-image, per-feature SHAP attributions for every one of the 17 named ABCD
  features — usable for patient-facing explanation text, consistent with this
  project's existing policy against presenting raw model internals as a diagnosis.
- A modality-level split (summed |SHAP| across CNN-PCA-component columns vs. named
  ABCD-feature columns, as a percentage) gives an interpretable "how much of this
  case's score came from the visual/deep-learning signal vs. the structured ABCD
  signal" summary, honestly scoped: individual PCA-component SHAP values are not
  themselves human-meaningful (a component is a rotated linear combination of 1024
  CNN activations), only the aggregate split is.
- **Grad-CAM** (backbone-level "where was the CNN looking" heatmaps) is described in
  the original proposal but **not implemented in this branch** — it wasn't part of
  this round's explicit ask, and doing it well deserves its own focused pass (choice
  of CAM variant, target layer selection for ConvNeXt's block structure, and how to
  present it alongside the existing `VisualReview`-style overlay materials this
  project already uses elsewhere). Flagged here as the clear next step, not silently
  dropped.

## What's genuinely new/changed vs. the original 8-file scaffold spec

For anyone diffing against the original scaffold request:
- `features.py`: fully replaced (real YOLO+validated extraction; dynamic feature
  count/order; sequential rather than joblib-parallel image processing, since a
  loaded neural net shouldn't be copied across worker processes the way pure
  NumPy/OpenCV work safely was).
- `config.py`: added environment-variable path overrides (`.env.example`),
  `PCA_N_COMPONENTS`, renamed `FPR_THRESHOLD` → `MIN_TPR`.
- `train_catboost.py`: added the per-fold PCA step; feature dimension is now
  derived (`PCA components actually used (per fold) + N_ABCD_FEATURES`), not a
  hardcoded `1024 + 11 = 1035`.
- `utils.py`: `compute_paauc` corrected (see above); `np.trapz` → `np.trapezoid`
  fixed as an unrelated but real bug (removed in NumPy 2.0+).
- `explain.py`: new file, not in the original spec.
- This `docs/design/` folder, `.env.example`, and the top-level `README.md`: new,
  not in the original spec.
