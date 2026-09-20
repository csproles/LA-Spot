"""Multi-lesion evaluation policy.

The previous convention (v2 verification run) used the highest-confidence
YOLO instance as a stand-in for "the lesion the image's benign/malignant
label refers to." That is an unjustified assumption: this ISIC data source
provides exactly one diagnosis per IMAGE, with no per-lesion localization or
identification field tying that diagnosis to a specific detected region
(confirmed in the metadata investigation — local metadata.csv and the live
ISIC Archive API both carry no such field). Confidence is a property of the
detector, not evidence about which lesion a pathologist biopsied.

Policy, per instruction:
  - single detected instance on a labeled image -> the label unambiguously
    refers to that one lesion; normal image-level evaluation applies
  - more than one detected instance on a labeled image -> which lesion (if
    any) the label refers to is UNKNOWN without dataset annotation; the
    image's ground truth is kept, every instance is still fully scored, but
    NONE of those instances may be counted as a TP/TN/FP/FN
  - no detection -> kept as its own category, never dropped
  - unlabeled images -> not applicable to classification evaluation at all

`target_lesion_id` exists so that if the dataset ever provides real
per-lesion annotation, that identified instance can be marked evaluable
without changing this policy's structure — it is unused today because no
such annotation exists for this data source.
"""

EVALUATION_STATUSES = (
    "SINGLE_LESION_EVALUABLE",
    "MULTI_LESION_AMBIGUOUS",
    "NO_DETECTION",
    "FAILED",
    "UNLABELED",
)


def determine_evaluation_status(ground_truth, pipeline_status, num_instances, target_lesion_id=None):
    if ground_truth not in ("benign", "malignant"):
        return "UNLABELED"
    if pipeline_status == "FAILED":
        return "FAILED"
    if pipeline_status == "NO_DETECTION":
        return "NO_DETECTION"
    if target_lesion_id is not None:
        # A specific instance has been identified by dataset annotation as
        # the one the label refers to. Not reachable today (no such
        # annotation exists here) — kept for forward compatibility only.
        return "SINGLE_LESION_EVALUABLE"
    if num_instances == 1:
        return "SINGLE_LESION_EVALUABLE"
    return "MULTI_LESION_AMBIGUOUS"
