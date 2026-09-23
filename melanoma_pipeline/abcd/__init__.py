"""Consolidated, standalone copy of this project's validated YOLO
segmentation + ABCD feature-extraction pipeline.

Every function in this package is copied UNCHANGED (import paths adjusted
to be self-contained; no formula, threshold, or algorithm modified) from
the main repository's `revised_abcd/`, `pipeline_v5/feature_extraction.py`,
and `Code/{ComputerVisionStuff,MelanomaDeterminingStuff}/` modules, so that
`melanoma_pipeline/` can run independently of those directories. Each
module below states its exact original source path in its own docstring
for provenance/traceability.

Do not reintroduce a dependency on the original repository layout here --
if you need to change one of these functions, change it in this package
directly (and note the change, since it will then diverge from the
original repo's own frozen/validated version).
"""
