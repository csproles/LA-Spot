# Segmentation benchmark: LAB/Otsu vs. YOLO, on official ISIC-2018 Task 1 ground truth

**This is an independent segmentation-quality experiment. It is not part of,
and shares zero image IDs with, the 3,482-image melanoma-vs-benign
classification cohort used in `Evaluation_FinalTargeted/` and
`Evaluation_DevComparison/`. No result here is merged into that cohort or
used to select/tune/retrain any classification pipeline.**

Branch: `pipeline-testing/research`. `user-shree` and V2/V3/V4 were not
modified — YOLO inference here uses a vendored, byte-identical, read-only
copy of `revised_abcd/yolo_single_image.py` (see that file's header),
extracted via `git show user-shree:...` without checking out or altering
that branch. No threshold was adjusted after seeing any result, and YOLO was
not retrained.

## Step 1 — locating official ground truth

Queried the public ISIC Archive API (`api.isic-archive.com`, read-only
metadata calls) and confirmed the full catalog of 62 public collections.
Cross-checked every collection involving lesion-boundary annotation against
our own images previously (reported separately): zero overlap between the
3,482-image classification cohort and any official/semi-official
segmentation collection. That ruled out running this benchmark on our own
cohort's images.

The collection used here is **"Challenge 2018: Task 1-2: Test"**
(ISIC Archive collection id **64**), the canonical official ISIC 2018
Challenge Task 1 test set (1,000 images, HAM10000-sourced, citing Codella et
al. 2018 and Tschandl et al. 2018). Its ground truth is the official
`ISIC2018_Task1_Test_GroundTruth.zip`
(`https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1_Test_GroundTruth.zip`,
9,680,101 bytes). **Verified official**: the zip's own bundled
`ATTRIBUTION.txt` reads *"ISIC-2018 Challenge: Task 1: Test Ground Truth by
Anonymous"*, CC0-licensed — confirmed before any image was processed. These
are the original challenge task's expert-annotated lesion-boundary masks,
not a model-generated or "novice" derivative (a separate, disqualified
collection — "ISIC 2018 Novice Segmentation masks", id 531, explicitly
U-Net-generated then reviewed — was found during the search and deliberately
excluded per the "no generated masks as ground truth" instruction).

IMA++ (collection 482, genuine multi-annotator manual segmentation) was also
considered but not used here, since the official 2018 Task 1 test set is the
more standard, single-authoritative-annotation benchmark; overlap with our
cohort was zero for IMA++ as well.

## Step 2 — the fixed 100-image sample

- Full list of all 1,000 Task-1-Test image IDs fetched and cached:
  `task1_2018_test_full_list.json`.
- Sampled with a fixed seed (`SEED = 20260921`, `random.Random(SEED).sample`)
  over the sorted ID list: **100 images**, IDs and provenance saved in
  `segmentation_benchmark_manifest.csv` (image_id, source collection,
  attribution, image URL, ground-truth zip URL/note, seed).
- Confirmed zero overlap between this 100-image sample and the 3,482-image
  classification cohort (dev + locked test combined).
- Each sampled image's official mask was extracted from the verified zip
  (`ground_truth/<id>_gt.png`); all 100 extracted successfully, none missing.
- Each sampled image's original was downloaded individually from its ISIC
  Archive `files.full.url` (not the 2.37 GB full Test-Input zip — only the
  100 needed originals, ~45 MB total).

## Step 3 — methods, run unchanged

- **LAB/Otsu**: the exact `Code/` chain from `Code/main.py` —
  `load_image → remove_vignette → remove_salt_pepper_noise(kernel_size=3) →
  apply_bilateral_filter(diameter=9, sigma_color=75, sigma_space=75) →
  remove_hair(kernel_size=17, threshold=10) → segment_lesion`. No parameter
  was changed from what `Code/main.py` already uses.
- **YOLO**: `revised_abcd/yolo_single_image.py::run_single_image_inference`
  (vendored copy), same frozen checkpoint used by V2/V3/V4
  (`melanoma_yolo26n_seg/weights/best.pt`), `conf=0.25`, single-image calls
  (no batching), `retina_masks=True`. Raster masks used directly — no
  polygon coordinates. Multi-instance images: the existing primary-instance
  policy (highest-confidence instance) was used for the scored comparison;
  instance count was recorded separately, not unioned.

## Step 4 — results (n = 100)

| Method | Mean IoU | Median IoU | Mean Dice | Median Dice | Failures |
|---|---|---|---|---|---|
| Original LAB/Otsu | 0.4417 | 0.4709 | 0.5415 | 0.6403 | 0 |
| YOLO | 0.7829 | 0.8544 | 0.8523 | 0.9215 | 4 |

YOLO failures = 4 no-detection images (0 instances found). LAB/Otsu never
produced an empty mask on this set (0 failures), though several of its
non-empty masks scored 0 IoU (see visual panels — the classical border-ring
skin-sampling step latched onto the wrong region).

**Paired comparison (same 100 images, same ground truth):**

| | IoU | Dice |
|---|---|---|
| YOLO wins | 91 | 91 |
| LAB/Otsu wins | 8 | 8 |
| Ties | 1 | 1 |

Paired difference (YOLO IoU − LAB/Otsu IoU): **mean = +0.3412, median =
+0.2808**.

YOLO multi-instance detections: 10 of 100 images (recorded, not unioned;
primary/highest-confidence instance used for the scored comparison above).

Full per-image data: `segmentation_metrics_per_image.csv` (image_id,
ground-truth area, both methods' predicted area, IoU, Dice, empty/failure
flags, YOLO instance count). Machine-readable summary:
`segmentation_metrics_summary.json`.

## Step 5 — visual review (interpretation only, not used to tune anything)

`visual_panels/` — Original | Official Ground Truth | LAB/Otsu | YOLO, two
examples per category:

- `yolo_substantially_better.png` — ISIC_0036212, ISIC_0023752
- `otsu_substantially_better.png` — ISIC_0023269, ISIC_0021596
- `both_good.png` — ISIC_0016438, ISIC_0015381
- `both_poor.png` — ISIC_0022219, ISIC_0021379

In the "YOLO substantially better" examples, LAB/Otsu's border-ring
skin-color sampling visibly locks onto a crescent-shaped region around the
lesion rather than the lesion itself, producing 0 IoU despite a non-empty
mask.

## What was NOT done

- No threshold, parameter, or preprocessing step was changed after seeing
  ground truth or results.
- YOLO was not retrained or fine-tuned.
- The sample was not expanded beyond 100 images.
- Nothing here was merged into, or used to affect, the 3,482-image
  classification cohort or any classification-pipeline decision.

## Files kept out of git (large/regenerable, matches this repo's existing
convention for other `Evaluation_*` raw-data folders)

`images/`, `ground_truth/`, `otsu_masks/`, `yolo_masks/` — reproducible by
re-running `run_segmentation_benchmark.py` against the committed manifest
and scripts. `__pycache__/` excluded as usual.
