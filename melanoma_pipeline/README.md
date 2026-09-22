# Hybrid Melanoma Detection Pipeline (CNN embedding + validated ABCD features)

A multimodal melanoma classifier: a fine-tuned CNN backbone (ConvNeXt-Base) supplies
a deep visual embedding, this project's existing, validated ABCD/structured lesion
features (YOLO segmentation + the frozen V5 feature pipeline) supply an interpretable
signal, and a downstream CatBoost classifier combines both — with SHAP explainability
on top so the structured features stay individually attributable per patient.

Design background: `docs/design/01_original_design_proposal.md` (the original
proposal) and `docs/design/02_agreed_architecture.md` (what was actually decided and
built, and why it differs from the proposal in a few places — read this one first if
you only read one).

**Status**: fully scaffolded and syntax/import-checked. **No training has been run,
no dataset has been downloaded, and no CNN checkpoints exist yet in this repository.**
Everything below tells you exactly what to obtain and run to produce them.

---

## 1. External assets you need before running anything

### 1a. The ISIC 2024 dataset — NOT included in this repository

This pipeline expects:
- A folder of individual lesion images (`.jpg`/`.jpeg`/`.png`, one file per
  `isic_id`).
- A metadata CSV with at least these columns: `isic_id`, `patient_id`, `target`
  (0 = benign, 1 = malignant). The public ISIC 2024 (SLICE-3D) challenge metadata
  already uses these exact column names — if your source uses different names,
  rename them to match (or override `config.TARGET_COL`/`config.GROUP_COL`).

Known public sources for this data (verify current access/registration
requirements yourself — competition data access terms change over time and this
README can't guarantee current availability):
- ISIC Challenge archive: https://challenge.isic-archive.com/
- The 2024 challenge was also hosted as a Kaggle competition:
  https://www.kaggle.com/competitions/isic-2024-challenge

**Practical note, not a formality**: the Kaggle distribution of this dataset ships
images inside a single HDF5 archive (`train-image.hdf5`), not as a folder of
individual image files. `dataset.py`/`features.py` in this package expect a folder
of individual files under `config.IMAGE_DIR`. If your copy is HDF5-packed, extract
it to individual files first (a short one-off script using `h5py` — not included
here, since it's a one-time data-prep step specific to whichever exact archive
format you end up with, not a permanent part of this pipeline).

Once you have images + metadata, set the paths (see "Configuration," below) and
place/point at them — this package never downloads anything itself.

### 1b. The YOLO segmentation checkpoint — already in this repository

Unlike the dataset, the YOLO checkpoint `features.py` uses is **already committed**
in this repo's git history at:
```
MelanomaDetection/MelanomaDetection.Python/models/yolo_melanoma_seg.pt
```
Cloning/checking out this branch gets you this file automatically — no separate
download step. If you ever want to point at a different checkpoint instead, set the
`YOLO_WEIGHTS_PATH` environment variable (see `.env.example`); `features.py` resolves
the checkpoint path via this project's existing `yolo_config.get_yolo_weights_path()`,
so the override behaves identically to how the live web application resolves it.

### 1c. ConvNeXt-Base pretrained weights — downloaded automatically by `timm`

The first time `train_cnn.py` runs, `timm` downloads the `convnext_base.
fb_in22k_ft_in1k` pretrained checkpoint from its own hub (requires internet access on
first run only; cached locally afterward, typically under `~/.cache/huggingface` or
`~/.cache/torch`). If your training device has no internet access, pre-download the
checkpoint on a machine that does and copy the `timm`/`huggingface_hub` cache
directory over.

---

## 2. Environment setup

```bash
cd melanoma_pipeline
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

This was developed/verified against Python 3.12, PyTorch 2.6 (CPU build in the
verification environment; install a CUDA-matched build for actual training —
see https://pytorch.org/get-started/locally/ for the exact command for your
device's CUDA version), NumPy 2.5, scikit-learn 1.9. `requirements.txt`
intentionally does not pin exact versions so you can match your device's CUDA
toolkit; if you hit an incompatibility, pin the versions above as a known-working
baseline.

Optional: `pip install python-dotenv` if you want `.env` to be loaded automatically
by the scripts below — see `.env.example` for the alternative (export environment
variables directly, or just edit `config.py`'s defaults).

---

## 3. Configuration

Copy `.env.example` to `.env` and edit the paths, or export the same environment
variables directly, or edit `config.py`'s defaults — all three are equally
supported (`config.py` reads every path via `os.environ.get(...)` with the same
relative-path fallback either way). At minimum, set:

```
MELANOMA_DATA_DIR=/path/to/isic2024
MELANOMA_IMAGE_DIR=/path/to/isic2024/images
MELANOMA_METADATA_CSV=/path/to/isic2024/metadata.csv
```

All other settings (fold count, seed, batch size, epochs, PCA component count) have
working defaults in `config.py` and can be left alone for a first run.

---

## 4. Running the pipeline — exact commands, in order

Run every command from inside `melanoma_pipeline/` (all paths in `config.py` are
relative to this directory by default).

```bash
# Stage 1 — extract the validated ABCD/structured features for every image
# (YOLO segmentation + this project's existing V5 feature formulas; NOT a
# simplified placeholder — see features.py's module docstring).
# Writes outputs/abcd_features.npy, outputs/abcd_index.npy, outputs/abcd_feature_names.npy.
python features.py

# Stage 2 — fine-tune ConvNeXt-Base under GroupKFold(5), patient-separated
# (see "Patient-separated splits," below). Writes outputs/checkpoints/fold_{0..4}_best.pt.
python train_cnn.py

# Stage 3 — extract CNN embeddings from each fold's best checkpoint.
# Writes outputs/embeddings/fold_{k}_{train,val}_{embeddings,labels,isic_ids}.npy.
python extract_embeddings.py

# Stage 4 — per-fold PCA (fit on that fold's training embeddings only) +
# concatenate with the validated ABCD features + train CatBoost.
# Writes outputs/catboost_fold_{0..4}.cbm and prints per-fold + OOF pAUC.
python train_catboost.py

# Stage 5 — SHAP explainability (per-image feature attributions + CNN-vs-ABCD
# modality split + population-level feature importance), per fold.
# Writes outputs/shap_fold_{0..4}_{per_image,modality_split,feature_importance}.csv.
python explain.py
```

Each stage reads only what the previous stage wrote to `outputs/` — you can re-run
any later stage without repeating earlier ones, as long as their outputs are still
on disk.

### Patient-separated splits

Every stage's cross-validation uses `sklearn.model_selection.GroupKFold(n_splits=5)`
grouped by `config.GROUP_COL` (`"patient_id"` by default) — the same patient's images
never appear in both the training and validation portion of the same fold. This
matters because ISIC-style datasets can have multiple photos of the same patient/
lesion; splitting by image instead of by patient would let a model "cheat" by
recognizing the same patient's skin/lesion across the train/validation boundary
rather than genuinely generalizing. This is the same splitting discipline used
throughout the rest of this project's evaluation work.

---

## 5. Verification already performed (without training or downloading data)

Everything below was actually run and checked, not just asserted:

```bash
# Every file compiles (syntax check)
python -m py_compile config.py utils.py dataset.py features.py train_cnn.py extract_embeddings.py train_catboost.py explain.py

# ConvNeXt-Base's real embedding dimension (confirms config.EMBEDDING_DIM = 1024
# is correct, not assumed)
python -c "import timm; m = timm.create_model('convnext_base.fb_in22k_ft_in1k', pretrained=False, num_classes=0); import torch; print(m(torch.randn(1,3,224,224)).shape)"

# compute_paauc verified against known reference values (perfect/random/reversed
# classifiers, tied scores, exact-boundary cases) -- see utils.py's own docstring
# and this branch's commit history for the verification commands used.
```

If you re-run these on your own device and get a different `EMBEDDING_DIM`, update
`config.py` — everything downstream (`extract_embeddings.py`, `train_catboost.py`)
derives its array sizing from that constant, so a mismatch there is the first thing
to check if you see a shape error.

---

## 6. Repository layout

```
melanoma_pipeline/
├── config.py               # single source of truth for paths/hyperparameters
├── .env.example             # environment-variable path template (copy to .env)
├── utils.py                 # compute_paauc, FocalLoss, checkpoint I/O, seeding
├── dataset.py                # ISICDataset, albumentations transforms, DataLoader factory
├── features.py               # REAL validated ABCD extraction (YOLO + V5 formulas)
├── train_cnn.py               # ConvNeXt-Base fine-tuning under GroupKFold(5)
├── extract_embeddings.py       # per-fold CNN embedding extraction
├── train_catboost.py           # per-fold PCA + concatenation + CatBoost training
├── explain.py                  # SHAP explainability (per-feature + modality split)
├── requirements.txt
├── docs/design/
│   ├── 01_original_design_proposal.md
│   └── 02_agreed_architecture.md   # read this one for what was actually decided
└── README.md                        # this file
```

`data/` and `outputs/` (datasets, checkpoints, embeddings, trained models, SHAP
CSVs) are gitignored — see the repo root `.gitignore` — since they're either large,
regenerable, or both. Nothing under either directory is expected to exist until you
run the commands above.

---

## 7. Known limitation / documented next step

Grad-CAM (backbone-level "where in the image was the CNN looking" heatmaps) is
described in the design docs but **not implemented in this branch** — `explain.py`
currently covers SHAP-based structured-feature and modality-level attribution only.
See `docs/design/02_agreed_architecture.md`'s explainability section for what a
Grad-CAM addition would need.
