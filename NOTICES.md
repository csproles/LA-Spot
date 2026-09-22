# Third-Party Notices and Attributions

This repository (on this branch) is a training package for a hybrid CNN + validated
ABCD melanoma-classification pipeline. It does not itself bundle the ISIC 2024
dataset or any other third-party dataset images — you obtain those yourself (see
`melanoma_pipeline/README.md`, "External assets"). This file preserves the
attribution/licensing notices that were carried alongside a small set of sample ISIC
images previously included in this repository (removed from this branch as part of
trimming it down to a minimal training package) and documents the licenses of the
major third-party libraries this pipeline depends on.

## ISIC sample images (previously bundled, now removed from this branch)

An earlier version of this repository bundled a small set of sample ISIC-sourced
images (`Images/Benign/`, `Images/Malignant/`) for local testing. Their attribution
files listed the contributor as:

```
Anonymous
```

and their license was **CC0 1.0 Universal** (public domain dedication) — the full
legal text is the standard Creative Commons CC0 1.0 Universal license, available at
https://creativecommons.org/publicdomain/zero/1.0/legalcode. When you obtain the
ISIC 2024 dataset yourself (per the README), check that dataset's own accompanying
license/attribution metadata — ISIC Archive images are contributed by many
institutions under a mix of licenses (commonly CC-0 or CC-BY-NC), not uniformly CC0,
and the per-image metadata typically identifies which license applies to which image.

## Third-party library licenses (informational, not exhaustive legal advice)

This pipeline's `requirements.txt` depends on the following major libraries. Verify
current license terms directly from each project before any redistribution — this
list is provided for awareness, not as a substitute for your own license review:

| Library | Typical license | Notes |
|---|---|---|
| PyTorch / torchvision | BSD-3-Clause | |
| timm | Apache-2.0 | |
| **Ultralytics (YOLO)** | **AGPL-3.0** (with a separate commercial/Enterprise license available from Ultralytics) | **Flagged specifically**: AGPL-3.0 has real implications if this pipeline (or a service built on it) is distributed or made network-accessible to others — it generally requires making the complete corresponding source available to users of such a service. This is not the case for a local research/training package used only by you, but is worth understanding before any wider deployment or distribution. See https://github.com/ultralytics/ultralytics and Ultralytics' own licensing page for current, authoritative terms. |
| scikit-learn | BSD-3-Clause | |
| CatBoost | Apache-2.0 | |
| SHAP | MIT | |
| albumentations | MIT | |
| OpenCV (opencv-python-headless) | Apache-2.0 | |
| pandas / NumPy | BSD-3-Clause | |

## This project's own trained YOLO checkpoint

The YOLO segmentation checkpoint this pipeline uses (`yolo_melanoma_seg.pt`) was
trained by this project's own team on ISIC-derived data. It is not redistributed in
this branch (excluded as a large binary, per this branch's asset-exclusion policy) —
see `melanoma_pipeline/README.md`'s "External assets" section for how to obtain it.
