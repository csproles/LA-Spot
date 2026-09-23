# Hybrid Melanoma Detection Pipeline (CNN embedding + validated ABCD features)

A multimodal melanoma classifier: a fine-tuned CNN backbone (ConvNeXt-Base) supplies
a deep visual embedding, this project's validated ABCD/structured lesion features
(YOLO segmentation + the frozen V5 feature formulas, consolidated into this package's
own `abcd/` module) supply an interpretable signal, and a downstream CatBoost
classifier combines both — with SHAP explainability on top so the structured
features stay individually attributable per patient.

**This package is standalone.** It does not import from, or require the presence
of, any other folder in the wider repository — `abcd/` is this package's own
consolidated copy of the validated YOLO segmentation and ABCD feature-extraction
code (see `abcd/__init__.py` for exactly what was consolidated from where).

**Status**: fully scaffolded, consolidated, and syntax/import-checked (including
from a location with no other repository folders present at all — see
"Verification," below). **No training has been run, no dataset has been
downloaded, and no CNN checkpoints exist yet.**

Design background: `docs/design/01_original_design_proposal.md` (the original
proposal) and `docs/design/02_agreed_architecture.md` (what was actually decided,
and why it differs from the proposal in a few places).

---

## 1. Final architecture

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

- **CNN backbone**: `convnext_base.fb_in22k_ft_in1k` (via `timm`) — ConvNeXt-Base,
  ImageNet-22k-pretrained then fine-tuned on ImageNet-1k by its original authors, as
  the starting checkpoint. Confirmed **1024-dim** penultimate pooled feature vector
  (empirically verified — see "Verification," below — not assumed).
- **Fine-tuning setting**: **full fine-tuning, no layers frozen**
  (`train_cnn.py::MelanomaCNN` — every backbone parameter is trainable). This was a
  deliberate choice given confirmed GPU access and the ~402k-image target dataset
  scale, versus a smaller/CPU-only setting where a staged frozen-first approach would
  have been safer. See `docs/design/02_agreed_architecture.md` for the full
  reasoning and what would change this recommendation.
- **Segmentation**: the project's existing frozen YOLO instance-segmentation
  checkpoint, reused as-is (independently benchmarked elsewhere in this project at
  mean IoU 0.783 / Dice 0.852 against expert ISIC ground truth). Not retrained here.
- **Structured features**: 17 validated ABCD features (see Section 4).
- **Fusion**: PCA-reduce the CNN embedding (`config.PCA_N_COMPONENTS`, default 128,
  clipped to `min(configured, n_train_samples, n_features)` per fold) — fit ONCE PER
  FOLD on that fold's training embeddings only, then applied (transform-only) to
  that fold's validation embeddings — then concatenate with the 17 ABCD features.
- **Classifier**: CatBoost (`CatBoostClassifier`), `scale_pos_weight` computed from
  each fold's own class ratio (handles class imbalance without needing the dataset
  pre-balanced).
- **Cross-validation**: `GroupKFold(n_splits=5)`, grouped by patient ID — see
  "Patient-separated splits" in Section 3.
- **Evaluation metric**: pAUC above TPR ≥ 0.80 (`config.MIN_TPR`), the ISIC-2024
  reference partial-AUC construction — see Section 4.
- **Explainability**: SHAP (`explain.py`) — per-structured-feature attributions plus
  a CNN-embedding-vs-ABCD-features modality split. (Grad-CAM is documented as a
  next step, not yet implemented — see `docs/design/02_agreed_architecture.md`.)

---

## 2. Environment setup

### GPU/CPU

This pipeline is designed to run with a GPU (full CNN fine-tuning on a large dataset
is not practical on CPU alone) but will run on CPU automatically if no GPU is
detected (`torch.cuda.is_available()` — see `train_cnn.py`/`extract_embeddings.py`).
Install a CUDA-matched PyTorch build for your device — see
https://pytorch.org/get-started/locally/ for the exact install command for your
CUDA version; `requirements.txt` intentionally does not pin a specific PyTorch
build for this reason.

### Setup

```bash
cd melanoma_pipeline
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Developed/verified against Python 3.12, PyTorch 2.6, NumPy 2.5, scikit-learn 1.9
(verification environment was CPU-only; install a CUDA build separately for actual
training).

### Configuration (all paths configurable, nothing hardcoded)

Copy `.env.example` to `.env` and edit, or export the same environment variables
directly, or edit `config.py`'s defaults — all three are equally supported. At
minimum:

```
MELANOMA_DATA_DIR=/path/to/isic2024
MELANOMA_IMAGE_DIR=/path/to/isic2024/images
MELANOMA_METADATA_CSV=/path/to/isic2024/metadata.csv
YOLO_WEIGHTS_PATH=/path/to/yolo_melanoma_seg.pt   # see "External assets" below
```

### Expected ISIC 2024 file layout

```
<MELANOMA_DATA_DIR>/
├── images/               # <MELANOMA_IMAGE_DIR> -- one file per isic_id
│   ├── ISIC_0123456.jpg
│   ├── ISIC_0123457.jpg
│   └── ...
└── metadata.csv           # <MELANOMA_METADATA_CSV>
```

`metadata.csv` must have at least these columns (the public ISIC 2024/SLICE-3D
challenge metadata already uses these exact names):

| column | meaning |
|---|---|
| `isic_id` | image identifier; must match an image filename (any of `.jpg`/`.jpeg`/`.png`) under `MELANOMA_IMAGE_DIR` |
| `patient_id` | used for patient-separated GroupKFold splitting (`config.GROUP_COL`) |
| `target` | 0 = benign, 1 = malignant (`config.TARGET_COL`) |

If your source uses different column names, rename them to match, or override
`config.GROUP_COL`/`config.TARGET_COL`.

**Practical note**: the Kaggle distribution of ISIC 2024 ships images inside a
single HDF5 archive (`train-image.hdf5`), not individual files. Extract it to
individual files under `MELANOMA_IMAGE_DIR` first (a short one-off `h5py` script —
not included here, since it's specific to whichever exact archive format you end up
with).

---

## 3. Exact commands, in order

Run every command from inside `melanoma_pipeline/`.

```bash
# Stage 1 — validated ABCD/structured feature extraction (YOLO segmentation +
# the consolidated V5 feature formulas in abcd/ -- NOT a placeholder).
# Writes outputs/abcd_features.npy, outputs/abcd_index.npy, outputs/abcd_feature_names.npy.
python features.py

# Stage 2 — ConvNeXt-Base fine-tuning under GroupKFold(5), patient-separated.
# Writes outputs/checkpoints/fold_{0..4}_best.pt.
python train_cnn.py

# Stage 3 — CNN embedding extraction from each fold's best checkpoint.
# Writes outputs/embeddings/fold_{k}_{train,val}_{embeddings,labels,isic_ids}.npy.
python extract_embeddings.py

# Stage 4 — per-fold PCA (fit on that fold's training embeddings only) +
# concatenate with the validated ABCD features + train CatBoost.
# Writes outputs/catboost_fold_{0..4}.cbm and prints per-fold + OOF pAUC.
python train_catboost.py

# Stage 5 — SHAP explainability (per-feature attributions + CNN-vs-ABCD
# modality split + population-level feature importance), per fold.
# Writes outputs/shap_fold_{0..4}_{per_image,modality_split,feature_importance}.csv.
python explain.py
```

Each stage reads only what the previous stage wrote to `outputs/` — re-run any
later stage without repeating earlier ones, as long as their outputs are on disk.

### Patient-separated splits

Every stage's cross-validation uses `GroupKFold(n_splits=5)` grouped by
`config.GROUP_COL` (`"patient_id"`) — the same patient's images never appear in
both the training and validation portion of the same fold, preventing a model from
"cheating" by recognizing the same patient's skin/lesion across the train/
validation boundary. PCA (`train_catboost.py`) and the CNN fine-tuning
(`train_cnn.py`) are both fit fresh per fold, on that fold's training data only.

---

## 4. Features, dimensions, labels, and metrics

### The 17 validated ABCD/structured features (`abcd.feature_config.V5_ALL_FEATURES`)

Feature order/count is **derived from this list at runtime**, not hardcoded — if it
ever changes, `features.py`'s output shape changes with it automatically.

| # | Feature | What it measures |
|---|---|---|
| 1 | `A_value` | Shape asymmetry: PCA-principal-axis-aligned mask, folded against its own horizontal/vertical mirror, `1 − mean(IoU_h, IoU_v)`. |
| 2 | `B_circularity` | Border irregularity: `1 − 4π·Area/Perimeter²` (despite the name, this is irregularity, not circularity — higher = more irregular). |
| 3 | `C_value` | Color variegation: coefficient of variation of per-pixel LAB distance from a sampled peri-lesional skin baseline. |
| 4 | `D_px` | Diameter of the minimum enclosing circle around the lesion contour, in **raw pixels** — never calibrated to physical mm (no defensible pixel→mm calibration exists for this data source). |
| 5 | `confidence` | YOLO's own detection confidence for the segmented instance (a segmentation-quality signal, not a clinical ABCD criterion). |
| 6 | `lesion_fraction` | Fraction of the full image frame covered by the lesion mask (potentially framing/zoom-sensitive, not a pure size measurement — see design docs). |
| 7 | `color_entropy` | Shannon entropy of a 16×16 2D histogram over the LAB a/b plane, within the lesion mask. |
| 8 | `lab_a_std` | Std of the LAB "a" (green–red) channel within the lesion mask. |
| 9 | `lab_b_std` | Std of the LAB "b" (blue–yellow) channel within the lesion mask. |
| 10 | `red_fraction` | Fraction of lesion pixels reading pink/red relative to the sampled skin baseline. |
| 11 | `bluegray_fraction` | Fraction of lesion pixels reading blue-gray relative to the sampled skin baseline. |
| 12 | `dark_fraction` | Fraction of lesion pixels reading very dark/black relative to the sampled skin baseline. |
| 13 | `skin_contrast` | LAB-space distance between the lesion's mean color and the mean color in a thin ring immediately surrounding it. |
| 14 | `solidity` | `contour_area / convex_hull_area` — lower = more concave/irregular border. |
| 15 | `turning_angle_std` | Std of the discrete turning angle along the contour, resampled to 100 equal arc-length points — a local border-roughness measure. |
| 16 | `eccentricity` | Elongation of the best-fit ellipse around the lesion outline. |
| 17 | `D_px_normalized` | `D_px / sqrt(image_height × image_width)` — a scale-normalized diameter proxy, still not a physical measurement. |

Exact formulas/code: `abcd/legacy_scoring.py` (`B_circularity`, `C_value` and the
color-fraction features), `abcd/asymmetry.py` (`A_value`), `abcd/diameter.py`
(`D_px`), `abcd/new_features.py` (the 11 additional features), `abcd/pipeline.py`
(the glue that assembles all 17 per image).

### Concatenated feature dimensions

`PCA components actually used this fold (≤ config.PCA_N_COMPONENTS, default 128)`
**+** `17` (the validated ABCD features) = the CatBoost model's input dimension for
that fold. The PCA component count is **not fixed at exactly 128** — it's clipped to
`min(configured, n_train_samples_in_this_fold, 1024)`, so a small fold can use fewer
components; the exact count actually used is saved per fold to
`outputs/catboost_fold_{k}_feature_names.npy` (used by `explain.py` so SHAP always
reads the right column layout, since it isn't guaranteed identical across folds).

### Label interpretation

`target = 0` → benign, `target = 1` → malignant (`config.TARGET_COL`). CatBoost's
`predict_proba(X)[:, 1]` is the model's estimated probability of the malignant
class — a research/screening-prototype score, not a clinical diagnosis.

### Evaluation metric: pAUC above TPR ≥ 0.80

`utils.compute_paauc` implements the ISIC-2024 reference partial-AUC construction:
flip both labels and scores (`y_flipped = 1 − y_true`, `scores_flipped = 1 − y_score`,
a lossless re-expression — `AUC(1−y, 1−s) == AUC(y, s)` always), compute `roc_curve`
on the flipped problem, truncate at `max_fpr = 1 − MIN_TPR` with linear boundary
interpolation, integrate via `sklearn.metrics.auc`. Returns the **raw** partial-AUC
area by default (this is literally what the ISIC-2024 Kaggle leaderboard reports,
typically ~0.15–0.20 for a competitive model — not rescaled to [0,1]); pass
`normalize=True` for a [0,1]-scaled version. Verified against known reference
values (perfect classifier: raw 0.1999…/normalized 1.0; random classifier: raw
≈0.02/normalized ≈0.10; completely reversed classifier: raw 0.0) plus tied-score and
exact-boundary edge cases — full correction history (two earlier, wrong iterations)
is in `utils.py`'s own docstring. `config.MIN_TPR = 0.80`.

Also reported alongside pAUC in `train_cnn.py`/`train_catboost.py`'s per-fold
output: standard accuracy-family metrics are NOT separately computed by this
pipeline's training scripts (pAUC is the sole training/validation metric, matching
the ISIC-2024 reference task) — if you want accuracy/precision/recall/F1/ROC-AUC as
well, compute them from the same saved OOF probabilities (`outputs/
shap_fold_{k}_per_image.csv` and the raw embeddings/labels under `outputs/
embeddings/` have everything needed).

---

## 5. External assets you need before running anything

### 5a. The ISIC 2024 dataset — not included

See "Expected ISIC 2024 file layout" in Section 2. Known public sources (verify
current access/registration requirements yourself):
- ISIC Challenge archive: https://challenge.isic-archive.com/
- Kaggle competition mirror: https://www.kaggle.com/competitions/isic-2024-challenge

### 5b. The YOLO segmentation checkpoint — NOT included in this branch, obtain separately

Unlike an earlier version of this branch, **this checkpoint is not committed here**
(this branch was trimmed to a minimal training package and no longer includes the
web application folder it used to live in). You have three ways to get it — use
whichever you actually have access to:

1. **The original Ultralytics training-run output**, if you have access to the
   machine/environment this project trained on: `runs/segment/
   melanoma_yolo26n_seg/weights/best.pt` (the `weights/` folder Ultralytics writes
   during training — `best.pt` is the checkpoint with the best validation metric,
   NOT `last.pt`, which is a *different* checkpoint, the final training epoch
   regardless of whether it was the best one). **Verified byte-for-byte identical**
   (SHA-256 `68131680...7882f5b`) to the `yolo_melanoma_seg.pt` this whole pipeline
   was built and evaluated against — this is the most direct source if you can reach
   it, since it requires no git access at all, just a file copy:
   ```bash
   cp /path/to/runs/segment/melanoma_yolo26n_seg/weights/best.pt \
     melanoma_pipeline/weights/yolo_melanoma_seg.pt
   ```
   After copying, you can confirm you have the exact right file with:
   ```bash
   python -c "import hashlib; print(hashlib.sha256(open('melanoma_pipeline/weights/yolo_melanoma_seg.pt','rb').read()).hexdigest())"
   # expect: 6813168001fe796cbaed8d1ce66f5539bd172a4f67e124edc50af017a7882f5b
   ```
2. **From this same repository's `user` branch** (if you have access to the full
   repository, not just this branch), without checking that whole branch out:
   ```bash
   git show user:MelanomaDetection/MelanomaDetection.Python/models/yolo_melanoma_seg.pt \
     > melanoma_pipeline/weights/yolo_melanoma_seg.pt
   ```
3. **Directly from your project team**, if you only have this standalone branch —
   ask whoever gave you access to this repository for a copy of
   `yolo_melanoma_seg.pt`, and place it at `melanoma_pipeline/weights/
   yolo_melanoma_seg.pt` (or point `YOLO_WEIGHTS_PATH` at wherever you put it).
   Ask them to confirm the SHA-256 above if you want to be certain it's the exact
   checkpoint this pipeline was built against, not a retrained/updated one.

`features.py` raises a clear, actionable error (naming this exact section) if the
checkpoint isn't found where `config.get_yolo_weights_path()` expects it.

### 5c. ConvNeXt-Base pretrained weights — downloaded automatically

`timm` downloads `convnext_base.fb_in22k_ft_in1k` from its own hub the first time
`train_cnn.py` runs (needs internet once; cached locally afterward). Pre-download
on a machine with internet access and copy the cache over if your training device
is offline.

---

## 6. Verification already performed (without training or downloading data)

```bash
# Every file compiles (syntax check)
python -m py_compile config.py utils.py dataset.py features.py train_cnn.py extract_embeddings.py train_catboost.py explain.py abcd/*.py

# Confirmed standalone: every module imports successfully from a location with
# NO other repository folders present at all (proves abcd/ has no hidden
# dependency on this repo's other folders).

# ConvNeXt-Base's real embedding dimension (confirms EMBEDDING_DIM = 1024,
# without downloading any pretrained weights -- shape depends only on the
# architecture definition):
python -c "import timm, torch; m = timm.create_model('convnext_base.fb_in22k_ft_in1k', pretrained=False, num_classes=0); print(m(torch.randn(1,3,224,224)).shape)"
```

If you get a different `EMBEDDING_DIM` on your device/timm version, update
`config.py` — everything downstream derives its array sizing from that constant.

---

## 7. Repository layout

```
melanoma_pipeline/
├── config.py                 # single source of truth for paths/hyperparameters
├── .env.example               # environment-variable path template (copy to .env)
├── utils.py                    # compute_paauc, FocalLoss, checkpoint I/O, seeding
├── dataset.py                   # ISICDataset, albumentations transforms, DataLoader factory
├── features.py                  # validated ABCD extraction entry point
├── abcd/                         # consolidated, standalone YOLO+ABCD source (see abcd/__init__.py)
│   ├── preprocessing.py           # vignette/denoise/bilateral/hair-removal chain
│   ├── legacy_scoring.py           # score_border, score_color (unchanged legacy formulas)
│   ├── asymmetry.py                 # PCA-aligned asymmetry
│   ├── border_experimental.py        # convexity-defect metric (recorded, not in the 17 features)
│   ├── diameter.py                    # diameter + (unreachable in this path) hair calibration
│   ├── segmentation_quality.py         # mask quality flags
│   ├── yolo_inference.py                # single-image YOLO inference wrapper
│   ├── new_features.py                   # the 11 additional V5 features
│   ├── feature_config.py                  # the 17-feature name/order list
│   └── pipeline.py                         # glue: preprocess_image / score_instance / process_image
├── train_cnn.py                # ConvNeXt-Base fine-tuning under GroupKFold(5)
├── extract_embeddings.py        # per-fold CNN embedding extraction
├── train_catboost.py             # per-fold PCA + concatenation + CatBoost training
├── explain.py                     # SHAP explainability
├── weights/                        # place yolo_melanoma_seg.pt here (gitignored)
├── requirements.txt
├── docs/design/
│   ├── 01_original_design_proposal.md
│   └── 02_agreed_architecture.md
└── README.md                        # this file
```

`data/`, `outputs/`, and `weights/*` (except `.gitkeep`) are gitignored — see the
repo root `.gitignore`.

---

## 8. Known limitation / documented next step

Grad-CAM (backbone-level "where in the image was the CNN looking" heatmaps) is
described in the design docs but **not implemented**. `explain.py` currently covers
SHAP-based structured-feature and modality-level attribution only. See
`docs/design/02_agreed_architecture.md`'s explainability section for what a
Grad-CAM addition would need.

---

## Prompt

This section preserves the exact prompts that shaped this pipeline's design and
implementation, in chronological order, for reference.

### 1. Original multimodal design request

> I want to explore a multimodal melanoma classification architecture because YOLO + ABCD alone is not performing strongly enough for a medical-use setting, and a CNN-only model would have limited interpretability.
>
> Design a pipeline that combines:
>
> 1. A CNN backbone to extract deep visual features from the lesion image.
> 2. Our current structured lesion features, including ABCD metrics and other engineered features.
> 3. Concatenate the CNN feature vector with the numerical ABCD/engineered-feature vector.
> 4. Feed the combined vector into a downstream classifier such as XGBoost, CatBoost, or another suitable model.
>
> The goal is to improve predictive performance while preserving explainability by keeping the ABCD/domain features available for feature importance and patient-facing explanations.
>
> Please propose:
> - recommended CNN backbone(s)
> - where to extract the CNN embedding
> - how to normalize and concatenate the two feature types
> - which downstream classifier to test first
> - training/validation strategy that avoids leakage
> - metrics to compare against our current Improved ABCD pipeline
> - how to provide explainability for both the CNN and structured features
>
> Do not modify the locked test set or use it for model selection. Start with a development-only experimental design.

### 2. CNN backbone follow-up

> which cnn architecture is better for this project and why

### 3. Dataset scale + segmentation question

> so we are going to use 402,000 around images from the isic dataset 2024 for dataset, does the yolo segmentation model currently being used is good to use for this multimodal deisgn we are about to implement? or attention u-net or transunet would be better to perform lesion segmentation

### 4. Label-scarcity clarification

> we don't have pixel level segmentation masks more than 2-3k images we processed so we need to generate that separately, in this case which segmentation would be good to use?

### 5. Backbone variant follow-up

> so for the cnn what about ConvNeXt-Base or EfficientNetV2-M?

### 6. GPU availability

> we have gpu resources

### 7. The scaffold specification (verbatim)

> does this script good to build the cnn model, i am not training the dataset right now in this device, just setting everything up for cnn first, according to our rest of the multimodal plan Build a hybrid melanoma detection pipeline. Do not train anything, do not download data.
> Just scaffold the full codebase so it is ready to run when the dataset is available.
>
> Project structure to create:
> melanoma_pipeline/
> ├── config.py
> ├── dataset.py
> ├── features.py
> ├── train_cnn.py
> ├── extract_embeddings.py
> ├── train_catboost.py
> ├── utils.py
> └── requirements.txt
>
> --- config.py ---
> Single source of truth. Include:
> - DATA_DIR, IMAGE_DIR, METADATA_CSV, OUTPUT_DIR, CHECKPOINT_DIR, EMBEDDING_DIR (all pathlib Paths)
> - N_FOLDS = 5, SEED = 42, GROUP_COL = "patient_id"
> - IMAGE_SIZE = 224, BATCH_SIZE = 8, GRAD_ACCUM_STEPS = 4 (effective batch = 32)
> - EPOCHS = 10, LR = 2e-4, WEIGHT_DECAY = 1e-4
> - MODEL_NAME = "convnext_base.fb_in22k_ft_in1k"
> - FOCAL_GAMMA = 2.0, FOCAL_ALPHA = 0.25
> - CATBOOST_ITERATIONS = 1000, CATBOOST_LR = 0.05, CATBOOST_DEPTH = 6
> - TARGET_COL = "target", FPR_THRESHOLD = 0.2 (for paAUC)
>
> --- dataset.py ---
> - ISICDataset class (torch Dataset): loads image path + label from a dataframe slice
> - Albumentations augmentation pipelines: train_transforms and val_transforms
>   Train: RandomResizedCrop, HorizontalFlip, VerticalFlip, ColorJitter, ShiftScaleRotate,
>          CoarseDropout, Normalize(ImageNet mean/std), ToTensorV2
>   Val: Resize, CenterCrop, Normalize, ToTensorV2
> - get_dataloader(df, fold_indices, mode, config) helper that returns a DataLoader
>
> --- features.py ---
> - extract_abcd_features(image_path) function using OpenCV:
>   1. Load image, convert to HSV
>   2. Otsu threshold on V channel to get lesion mask
>   3. Asymmetry: compare top/bottom and left/right halves of the masked region, return
>      normalized difference score
>   4. Border irregularity: perimeter^2 / (4 * pi * area) — compactness ratio
>   5. Color variance: std of R, G, B channels within the mask
>   6. Diameter estimate: equivalent diameter from contour area
>   7. Extended: mean H, S, V within mask; lesion-to-image area ratio
> - Returns a numpy array of shape (11,) with all features
> - extract_features_for_dataframe(df, image_dir, n_jobs=-1) that runs the above in
>   parallel using joblib and returns a numpy array of shape (N, 11)
> - Add a __main__ block that loads metadata CSV, extracts all features, saves to
>   OUTPUT_DIR/abcd_features.npy and a matching index to OUTPUT_DIR/abcd_index.npy
>
> --- utils.py ---
> - compute_paauc(y_true, y_score, fpr_threshold=0.2) using sklearn.metrics.roc_curve,
>   returns partial AUC normalized to [0,1]
> - FocalLoss class (nn.Module): BCE-based focal loss, takes gamma and alpha from config
> - seed_everything(seed) utility
> - save_checkpoint(model, optimizer, epoch, fold, path) and load_checkpoint(path, model, optimizer)
> - AverageMeter class for tracking loss/metric during training
>
> --- train_cnn.py ---
> - Uses GroupKFold on GROUP_COL to create 5 folds
> - For each fold:
>   - Instantiate ConvNeXt-Base from timm with pretrained weights, replace head with
>     Linear(1024, 1) + Sigmoid
>   - Fine-tune all layers (no freezing)
>   - Mixed precision with torch.cuda.amp.GradScaler
>   - Gradient accumulation over GRAD_ACCUM_STEPS
>   - FocalLoss from utils.py
>   - AdamW optimizer with cosine LR scheduler
>   - After each epoch, evaluate val fold with compute_paauc
>   - Save best checkpoint per fold to CHECKPOINT_DIR/fold_{k}_best.pt
> - At end, print per-fold paAUC and mean paAUC
>
> --- extract_embeddings.py ---
> - For each fold, load best checkpoint
> - Remove the final Linear head, use model as feature extractor (output 1024-dim)
> - Run full train+val set through extractor
> - Save embeddings to EMBEDDING_DIR/fold_{k}_train_embeddings.npy and val equivalent
> - Save corresponding labels and isic_ids alongside
>
> --- train_catboost.py ---
> - For each fold:
>   - Load CNN embeddings (1024-dim) from EMBEDDING_DIR
>   - Load ABCD features (11-dim) from OUTPUT_DIR/abcd_features.npy, align by isic_id
>   - Concatenate → (1035,) feature vector per sample
>   - Train CatBoostClassifier with scale_pos_weight computed from fold's class ratio
>   - Evaluate on val fold with compute_paauc
>   - Save model to OUTPUT_DIR/catboost_fold_{k}.cbm
> - Print per-fold paAUC and OOF mean paAUC
>
> --- requirements.txt ---
> torch, torchvision, timm, albumentations, opencv-python-headless, scikit-learn,
> catboost, joblib, pandas, numpy, tqdm, pathlib
>
> General rules:
> - No training runs, no data downloads, no __main__ execution except features.py and
>   the explicit extraction scripts
> - Every file must be importable without side effects
> - Use tqdm for all loops that touch data
> - Inline comments on: focal loss math, gradient accumulation reset logic,
>   embedding extraction (why hooks vs head removal), paAUC normalization
> - All paths via config.py, never hardcoded
> - Python 3.10+, type hints throughout

### 8. First pAUC correction

> Fix compute_paauc in utils.py.
>
> The current implementation computes pAUC where FPR ≤ threshold, which is wrong.
> The ISIC 2024 metric is pAUC where TPR ≥ 0.80 — the area under the ROC curve
> in the high-sensitivity region.
>
> Replace the function with this logic:
> 1. Use sklearn.metrics.roc_curve to get fpr, tpr, thresholds arrays
> 2. Find the subset of points where tpr >= 0.80
> 3. Compute the area under that subset using np.trapz(tpr_subset, fpr_subset)
> 4. Normalize by the maximum possible area in that region: (1.0 - min_fpr_in_subset) * (1.0 - 0.80)
>    so the result is in [0, 1]
> 5. Update the docstring to clearly state: "pAUC above TPR=0.80, normalized to [0,1]"
> 6. Update the FPR_THRESHOLD constant in config.py — rename it to MIN_TPR = 0.80
>    and update every reference to it across all files
>
> Also update train_cnn.py and train_catboost.py wherever compute_paauc is called
> to pass min_tpr=0.80 instead of fpr_threshold.

### 9. Second pAUC correction (the actual ISIC reference construction)

> Please correct this again. The current calculation is average TPR over a selected FPR interval, not the requested pAUC.
>
> Implement the ISIC reference approach:
>
> 1. Flip binary labels and probability scores: y_flipped = 1 - y_true; scores_flipped = 1 - y_score.
> 2. Compute roc_curve(y_flipped, scores_flipped).
> 3. Set max_fpr = 1 - min_tpr.
> 4. Truncate this transformed ROC at max_fpr, inserting a linearly interpolated boundary point.
> 5. Compute the raw area using sklearn.metrics.auc.
> 6. Return raw area by default; optionally normalize by dividing by max_fpr.
>
> For min_tpr=0.80:
>
> * Perfect classifier: 0.20 raw / 1.00 normalized.
> * Theoretical random diagonal: 0.02 raw / 0.10 normalized.
> * Completely reversed ranking: 0.00.
>
> Do not divide by the observed FPR width. Correct the docstrings and remove the misleading McClish explanation. Keep MIN_TPR and the updated call sites. Verify against the reference implementation, including tied scores and boundary interpolation, before using this metric for model selection.

### 10. Branch creation and full-repo consolidation

> Create a separate Git branch named `cnn-with-abcd` for the complete multimodal pipeline, then commit and push it to the existing remote.
>
> Inspect the current repository and our design instructions. Include and finish implementing everything needed for my professor to run the pipeline on another device: ConvNeXt setup and training, existing YOLO segmentation and masks, validated ABCD/structured-feature extraction, CNN embedding extraction, feature concatenation, CatBoost training, evaluation, and SHAP explanations. Replace any Otsu scaffold or placeholder features with the actual existing extractor. Derive feature dimensions from real outputs.
>
> Include all required source files, helpers, dependency specifications, configuration templates, the design document, and a setup/design prompt documenting the agreed architecture. Add a README with exact commands to run all stages, patient-separated splits, configurable paths, and instructions for obtaining ISIC 2024 data and required model weights. Keep PCA and other learned preprocessing within training folds, and preserve the corrected pAUC implementation.
>
> Exclude datasets, large weights, virtual environments, credentials, caches, and generated outputs from Git. Clearly document how to transfer or download required external assets.
>
> Check imports, dependencies, and syntax without training or downloading the dataset. Preserve unrelated work, stage only relevant files, review the staged diff for secrets and large files, commit, and push with upstream tracking. Do not merge into the original branch or force-push.
>
> Finish with the branch link, commit hash, external files my professor needs, and exact starting commands. Clearly report any unresolved blockers.

### 11. Design-doc clarification

> what's the need of 2 files in melanoma_pipeline/docs/design?

### 12. Standalone-package cleanup (this pass)

> Clean up the `cnn-with-abcd` branch into a minimal, standalone training package for my professor. He will download ISIC 2024 himself and will not run the web app.
>
> Inspect all imports and dependencies first. Include everything required for the complete pipeline: ConvNeXt setup/training → embedding extraction + existing YOLO-based ABCD/structured-feature extraction → concatenation → CatBoost training → evaluation and explanations. Preserve the agreed architecture, validated feature calculations, patient-separated splits, training-fold-only preprocessing, and corrected pAUC implementation. Replace any remaining Otsu scaffold or placeholder features.
>
> Consolidate required YOLO/ABCD source code and helpers inside `melanoma_pipeline/`, updating imports so it runs independently of the other repository folders.
>
> Keep only:
>
> * `melanoma_pipeline/` with all required pipeline code
> * Dependency specifications and portable configuration
> * README, relevant multimodal design document, .gitignore, and required licenses/attributions
>
> Remove the web app, application backend, sample images, unrelated legacy code, .claude settings, and unrelated documentation from this branch after checking dependencies. Exclude datasets, large model weights, generated outputs, credentials, caches, and virtual environments. Document how to obtain every required external asset, including our trained YOLO weights.
>
> In the README, include:
>
> 1. Final architecture, exact ConvNeXt variant, and frozen/fine-tuning settings.
> 2. Environment setup, GPU/CPU configuration, expected ISIC 2024 file layout, and configurable paths.
> 3. Exact commands to run every stage in order.
> 4. Feature definitions, concatenated dimensions, label interpretation, and evaluation metrics.
> 5. A section titled "Prompt" containing the prompts I provided for the CNN and all related pipeline setup and corrections.
>
> Work only on `cnn-with-abcd`. Preserve unrelated uncommitted work and leave other branches unchanged. Verify syntax, imports, and command-line entry points without training or downloading datasets. Review the staged changes for missing dependencies, secrets, and large files, then commit and push normally with upstream tracking. Do not merge or force-push.
>
> Report the branch link, commit hash, final folder tree, starting commands, required external assets, and any unresolved blockers.
