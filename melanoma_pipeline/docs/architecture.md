# Architecture: What This Pipeline Implements

A single, current reference for the hybrid CNN + validated ABCD melanoma pipeline:
what it actually does, right now, in the code. (This replaces two earlier separate
documents, an initial design proposal and a later "what changed" record, which
were confusing to read side by side. Anything from those worth keeping is folded in
below; anything superseded is dropped rather than carried forward as noise.)

## Overview

```
lesion image
  ├──► ConvNeXt-Base (fine-tuned, all layers trainable) ──► 1024-dim embedding
  │                                                              │
  │                                                        PCA (fit on that
  │                                                        fold's TRAIN
  │                                                        embeddings only)
  │                                                              │
  │                                                     up to 128-dim reduced
  │                                                        embedding
  │                                                              │
  └──► YOLO segmentation ──► validated ABCD/structured           │
       (frozen checkpoint)   feature extraction (17 features)    │
                                    │                             │
                                    └──────────► concatenate ◄────┘
                                                      │
                                              CatBoost classifier
                                          (scale_pos_weight per fold)
                                                      │
                                      malignant-probability score
                                       + SHAP explanations (per
                                       structured feature + CNN-vs-
                                       ABCD modality split)
```

Two feature sources, fused at the feature level (not end-to-end joint training),
specifically so the structured ABCD features stay individually attributable for
SHAP explanations rather than being absorbed into an opaque joint representation.

## 1. CNN backbone: ConvNeXt-Base

`convnext_base.fb_in22k_ft_in1k` (via `timm`): ImageNet-22k-pretrained, then
fine-tuned on ImageNet-1k by its original authors, as the starting checkpoint.
Confirmed 1024-dim penultimate pooled feature vector (`timm.create_model(...,
num_classes=0)`; verify with the command in the README rather than trusting this
number blindly).

**Why this over the alternatives considered**: EfficientNet-B0 and ResNet-50 are
better choices under CPU-only, small-dataset constraints (fewer parameters, lower
overfitting risk). That was the right call earlier in this project's dataset-scale
history. Once GPU access was confirmed and the target dataset grew to ISIC-2024
scale (~402k images), the calculus flipped: fine-tuning (not just frozen-embedding
extraction) became practical, and ConvNeXt-Base's main strength, using large-scale
pretraining well, only pays off once you can actually fine-tune. EfficientNetV2-M
was the runner-up (faster to train, lighter on GPU memory) and remains a reasonable
fallback under VRAM/iteration-speed pressure. Swapping is a one-line change to
`config.MODEL_NAME` (and re-confirming `config.EMBEDDING_DIM` for the new backbone).

**Fine-tuning setting: full fine-tuning, no layers frozen**
(`train_cnn.py::MelanomaCNN`: every backbone parameter is trainable, fresh per
fold). Justified specifically by confirmed GPU access + the ~402k-image scale, which
together make the overfitting risk that would otherwise argue for a staged
frozen-then-fine-tune approach much smaller than it would be on a small CPU-only
dataset.

## 2. Segmentation: YOLO (frozen, unchanged)

The project's existing frozen YOLO instance-segmentation checkpoint, reused as-is,
independently benchmarked elsewhere in this project at mean IoU 0.783 / Dice 0.852
against expert ISIC ground truth. Not retrained here. Only ~2–3k images in this
project have (or will have) pixel-level masks, which is part of why retraining
segmentation wasn't pursued as part of this pipeline. See the "Prompt" section of
`melanoma_pipeline/README.md` for the fuller reasoning behind sticking with YOLO
over an Attention U-Net/TransUNet alternative at that label budget.

## 3. Validated ABCD/structured features: 17 features

Real, validated feature extraction, **not** a simplified placeholder. `features.py`
calls this package's own consolidated `abcd/` module (relocated, unmodified copy of
the project's frozen YOLO+ABCD pipeline; see `abcd/__init__.py`), the same
sequence the live application's own detector uses: YOLO segmentation → the
unchanged base-6 features (`A_value, B_circularity, C_value, D_px, confidence,
lesion_fraction`) → 11 additional features (`abcd/new_features.py`). 17 features
total, in the order `abcd.feature_config.V5_ALL_FEATURES` defines. **The feature
count is derived from that list at runtime**, not hardcoded, so this module can't
silently drift out of sync. An early scaffold version of `features.py` used a
from-scratch Otsu-threshold segmentation + 11 generic OpenCV features instead; that
version has been fully replaced (the Otsu approach was independently benchmarked at
IoU 0.442 vs. YOLO's 0.783 in this project's own prior audit: known to be worse).
Full feature definitions: `melanoma_pipeline/README.md` §4.

## 4. CNN input: currently the full image, not a lesion crop

**Known gap, stated plainly rather than glossed over**: the original design intent
for this pipeline was to feed the CNN a YOLO-cropped lesion patch (bounding box +
padding), for two reasons: focusing the network on the lesion itself, and
countering the photo-framing/zoom confound already documented for
`lesion_fraction`/`D_px_normalized` elsewhere in this project's audit work (a CNN
trained on full images can't see how much of the frame the lesion occupied, so it
can't accidentally learn "zoomed-in photos are more concerning" the way a
framing-sensitive numeric feature might).

**What's actually implemented**: `dataset.py::ISICDataset` loads the full original
image and applies generic (not lesion-aware) augmentation (`RandomResizedCrop` for
training, `Resize`+`CenterCrop` for validation), matching this pipeline's literal
scaffold specification, which described generic augmentation rather than
YOLO-box-based cropping. This is a real, unresolved gap between the original design
intent and the current code, not a deliberate revision. Adding lesion-aware cropping
(reusing the same mask `features.py` already computes) is a concrete, low-cost next
step if/when full-image training doesn't perform as well as hoped.

## 5. Fusion: per-fold PCA, then concatenate

PCA-reduces the CNN embedding (`config.PCA_N_COMPONENTS`, default 128, clipped to
`min(configured, n_train_samples, n_features)` per fold) toward the same order of
magnitude as the 17 ABCD features, then concatenates. **The PCA is fit exactly once
per fold, on that fold's training embeddings only**, and applied (transform-only) to
that fold's validation embeddings (`train_catboost.py::fit_transform_pca`). This
was an explicit requirement (keep PCA and other learned preprocessing within
training folds) and is the same per-fold-fit discipline the CNN and CatBoost
training both already follow. Note: PCA is fit directly on the raw embedding (no
separate standardization/L2-normalization step beforehand), a simplification
relative to the earliest design sketch, which suggested standardizing first; worth
revisiting if PCA component scaling turns out to matter in practice.

Why PCA at all: without it, ~1024 CNN dimensions would numerically swamp the 17
ABCD features for most downstream classifiers, undermining the goal of keeping
structured features individually attributable in SHAP output.

## 6. Classifier: CatBoost

`CatBoostClassifier`, `scale_pos_weight` computed per fold from that fold's own
class ratio (handles the malignant/benign imbalance without needing the dataset
pre-balanced). `shap.TreeExplainer` works identically well against CatBoost as it
would against XGBoost/LightGBM (both considered earlier), so the choice of CatBoost
specifically doesn't affect the explainability plan below.

**Caveat worth knowing**: `model.fit(X_train, y_train, eval_set=(X_val, y_val),
use_best_model=True)`: CatBoost's early stopping selects the best training
iteration using each fold's own *validation* set directly. This is a mild form of
tuning against the outer evaluation fold (not full hyperparameter search, just
picking when to stop), not a fully separate inner-validation split. For the
strictest possible leakage discipline this would ideally use a nested/inner split
instead. Flagged as a possible refinement, not fixed here.

## 7. Cross-validation / leakage discipline

`GroupKFold(n_splits=5)`, grouped by `patient_id`. The same patient's images never
appear in both the training and validation portion of the same fold (see
`melanoma_pipeline/README.md`'s "Patient-separated splits"). The CNN, the PCA step,
and CatBoost are all fit fresh per fold, on that fold's training data only; nothing
is fit on pooled/all-fold data.

## 8. Evaluation metric: pAUC above TPR ≥ 0.80

`utils.compute_paauc` implements the ISIC-2024 reference partial-AUC construction
(label/score flip, `roc_curve` on the flipped problem, truncate at `max_fpr = 1 −
MIN_TPR` with boundary interpolation, integrate via `sklearn.metrics.auc`). Went
through two incorrect iterations before landing here. Full correction history is
in `utils.py`'s own docstring. Returns the raw partial-AUC area by default (what the
ISIC-2024 leaderboard itself reports), or a [0,1]-normalized version via
`normalize=True`. `config.MIN_TPR = 0.80`.

## 9. Explainability: SHAP implemented, Grad-CAM not yet

`explain.py` implements:
- `shap.TreeExplainer` on the fitted CatBoost model: exact, per-image, per-feature
  attributions for every one of the 17 named ABCD features, usable for patient-facing
  explanation text (consistent with this project's existing policy against
  presenting raw model internals as a diagnosis).
- A modality-level split (summed |SHAP| across CNN-PCA-component columns vs. named
  ABCD-feature columns): an interpretable "how much of this case's score came from
  the visual signal vs. the structured ABCD signal" summary. Individual PCA
  component SHAP values are not themselves human-meaningful; only this aggregate
  split is.

**Grad-CAM** (backbone-level "where was the CNN looking" heatmaps) is **not
implemented**. It would need its own focused pass: CAM variant choice, target-layer
selection for ConvNeXt's block structure, and how to present it alongside this
project's existing `VisualReview`-style overlay materials. Flagged as the clear next
step for explainability, not silently dropped.

## Summary of known gaps vs. original design intent

1. **CNN input is the full image, not a YOLO-cropped lesion patch** (Section 4):
   the most consequential gap; affects both interpretability and the
   framing/zoom-confound concern the crop was meant to address.
2. **CatBoost's early stopping uses each fold's own validation set directly**
   (Section 6): a mild leakage-discipline softening versus a fully separate inner
   split.
3. **PCA is fit on raw (non-standardized) embeddings** (Section 5): a
   simplification, not necessarily a problem, but worth testing if results seem off.
4. **Grad-CAM is not implemented** (Section 9).

None of these are bugs. The pipeline runs and is methodologically sound as built,
but they're real differences from the fullest version of the original design, and
are recorded here so they don't get lost or silently assumed to be already handled.
