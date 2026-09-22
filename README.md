# Hybrid Melanoma Detection — Training Pipeline (`cnn-with-abcd` branch)

This branch is a **minimal, standalone training package** — a research pipeline that
combines a fine-tuned CNN embedding with this project's validated ABCD/structured
lesion features and a CatBoost classifier. It intentionally does **not** include the
production web application, its backend, or the rest of this repository's broader
history; it is trimmed down specifically so it can be handed to someone who only
needs to train and evaluate this model, on their own device, with their own copy of
the ISIC 2024 dataset.

**Everything you need is in `melanoma_pipeline/`.** Start there:

➡️ **[`melanoma_pipeline/README.md`](melanoma_pipeline/README.md)** — architecture,
environment setup, exact commands for every stage, feature definitions, and required
external assets (dataset, YOLO checkpoint).

Also at this repo root:
- **[`NOTICES.md`](NOTICES.md)** — third-party license notices and attributions
  (ISIC sample-image licensing, and the licenses of major dependencies this pipeline
  uses, including a specific note on Ultralytics/YOLO's AGPL-3.0 licensing).
- **`.gitignore`** — excludes the dataset, generated outputs, model weights,
  virtual environments, and credentials from version control (see `melanoma_pipeline/
  README.md` for what to obtain and where to place it instead).

## What was deliberately removed from this branch

The full production repository (Flask/.NET web application, Docker deployment
config, the original classical ABCD pipeline, and various research/evaluation
scripts) exists on this project's other branches. This branch removes all of that
and keeps only what's needed to train and evaluate the hybrid CNN+ABCD model —
`melanoma_pipeline/` is fully self-contained (its own consolidated copy of the
validated YOLO/ABCD feature-extraction code, so it does not depend on any other
folder in this repository being present).
