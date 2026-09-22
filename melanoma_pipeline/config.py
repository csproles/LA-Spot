"""Single source of truth for the hybrid (CNN embedding + validated ABCD)
melanoma pipeline. Every other module reads paths/hyperparameters from
here -- no path or hyperparameter should be hardcoded anywhere else in this
package.

Path values can be overridden without editing this file via environment
variables (see the `.env.example` template alongside this module, and the
README's "Configuration" section) -- this is what lets the same code run
unmodified on a different device/dataset location. Every `os.environ.get`
call below falls back to the same relative-path defaults as before if the
variable isn't set, so nothing breaks if `.env` is absent.

Nothing in this module has side effects beyond reading already-set
environment variables (no directory creation, no file I/O) so it is always
safe to import. Call `ensure_dirs()` explicitly from an entry-point script
if you need the output directories to exist on disk.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------- paths ----
# Every one of these can be overridden via an environment variable of the
# same name (e.g. set MELANOMA_DATA_DIR in your shell or in a `.env` file
# loaded by your process manager) -- see .env.example. Defaults assume you
# run scripts from inside this `melanoma_pipeline/` directory.
DATA_DIR: Path = Path(os.environ.get("MELANOMA_DATA_DIR", "data"))
IMAGE_DIR: Path = Path(os.environ.get("MELANOMA_IMAGE_DIR", str(DATA_DIR / "images")))
METADATA_CSV: Path = Path(os.environ.get("MELANOMA_METADATA_CSV", str(DATA_DIR / "metadata.csv")))

OUTPUT_DIR: Path = Path(os.environ.get("MELANOMA_OUTPUT_DIR", "outputs"))
CHECKPOINT_DIR: Path = Path(os.environ.get("MELANOMA_CHECKPOINT_DIR", str(OUTPUT_DIR / "checkpoints")))
EMBEDDING_DIR: Path = Path(os.environ.get("MELANOMA_EMBEDDING_DIR", str(OUTPUT_DIR / "embeddings")))

# Path to the YOLO segmentation checkpoint used by features.py. NOT
# committed to git (excluded as a large model weight) -- see the README's
# "External assets" section for exactly how to obtain this project's
# already-trained checkpoint. Defaults to a `weights/` folder inside this
# package; override with the YOLO_WEIGHTS_PATH environment variable to
# point at a checkpoint stored elsewhere instead.
YOLO_WEIGHTS_PATH: Path = Path(os.environ.get("YOLO_WEIGHTS_PATH", "weights/yolo_melanoma_seg.pt"))


def get_yolo_weights_path() -> Path:
    """Resolves the YOLO checkpoint path, raising a clear, actionable error
    (rather than a confusing downstream file-not-found from inside
    `ultralytics`) if it hasn't been placed yet."""
    if not YOLO_WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"YOLO checkpoint not found at {YOLO_WEIGHTS_PATH.resolve()}. "
            f"This project's already-trained checkpoint is not committed to git "
            f"(excluded as a large binary) -- see README.md's 'External assets' "
            f"section for exactly how to obtain it, or set the YOLO_WEIGHTS_PATH "
            f"environment variable to point at a copy stored elsewhere."
        )
    return YOLO_WEIGHTS_PATH


def ensure_dirs() -> None:
    """Create every output directory this package writes to. Call this
    explicitly from a script's __main__/entry point -- never at import
    time, so importing config.py never touches the filesystem."""
    for d in (OUTPUT_DIR, CHECKPOINT_DIR, EMBEDDING_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------- splits -----
N_FOLDS: int = int(os.environ.get("MELANOMA_N_FOLDS", 5))
SEED: int = int(os.environ.get("MELANOMA_SEED", 42))
GROUP_COL: str = "patient_id"   # GroupKFold grouping column -- prevents the
                                 # same patient's images appearing in both
                                 # a fold's train and validation split
                                 # (patient-separated splits; see README).
TARGET_COL: str = "target"      # binary label column (0=benign, 1=malignant)

# ---------------------------------------------------------- image/CNN -----
IMAGE_SIZE: int = 224
BATCH_SIZE: int = int(os.environ.get("MELANOMA_BATCH_SIZE", 8))
GRAD_ACCUM_STEPS: int = 4       # effective batch size = BATCH_SIZE * GRAD_ACCUM_STEPS = 32

EPOCHS: int = int(os.environ.get("MELANOMA_EPOCHS", 10))
LR: float = 2e-4
WEIGHT_DECAY: float = 1e-4

MODEL_NAME: str = "convnext_base.fb_in22k_ft_in1k"  # timm model id
EMBEDDING_DIM: int = 1024  # ConvNeXt-Base's penultimate feature dimension --
                            # empirically confirmed (not just assumed) by
                            # instantiating `timm.create_model(MODEL_NAME,
                            # num_classes=0)` and checking its output shape;
                            # see the README's "Verification" section for
                            # the exact command. Used by extract_embeddings.py
                            # and train_catboost.py to size arrays without
                            # re-instantiating the model every time.

# ------------------------------------------------------------ focal loss --
FOCAL_GAMMA: float = 2.0
FOCAL_ALPHA: float = 0.25

# --------------------------------------------------------------- CatBoost --
CATBOOST_ITERATIONS: int = 1000
CATBOOST_LR: float = 0.05
CATBOOST_DEPTH: int = 6

# ------------------------------------------------- CNN-embedding PCA -----
# train_catboost.py reduces each fold's CNN embedding via PCA BEFORE
# concatenating it with the validated ABCD feature vector, fit on that
# fold's TRAINING rows only (never on pooled/all-fold data, and never
# including validation rows) -- the same per-fold-fit discipline every
# other learned preprocessing step in this pipeline (StandardScaler,
# the CNN itself, CatBoost) already follows. This keeps the ~1024 CNN
# dimensions from numerically swamping the much smaller (17-dim) validated
# ABCD feature vector, and keeps the downstream CatBoost model's effective
# parameter count sane relative to a single fold's training-row count.
# See docs/architecture.md ("Fusion") for the full reasoning.
PCA_N_COMPONENTS: int = int(os.environ.get("MELANOMA_PCA_N_COMPONENTS", 128))

# ------------------------------------------------------------------ eval --
# Partial-AUC evaluation region: "AUC above TPR (sensitivity) >= MIN_TPR" --
# the high-sensitivity ROC region the ISIC-2024 competition metric scores
# on. utils.compute_paauc() implements this via the reference label/score-
# flip construction (verified against known perfect/random/reversed
# reference values -- see that function's docstring for the exact formula
# and its correction history; two earlier versions were wrong).
MIN_TPR: float = 0.80
