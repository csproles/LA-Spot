# LASpot — LLM handoff contract (Phase 9)

## What the CV layer hands the LLM

`pipeline_v3.structured_output.analyze_image(model, image_path)` returns a
JSON-serializable dict. Example, one lesion detected:

```json
{
  "segmentation_status": "SINGLE_LESION_DETECTED",
  "num_lesion_instances": 1,
  "lesion_instances": [
    {
      "lesion_instance_id": 0,
      "is_primary_instance": true,
      "yolo_confidence": 0.86,
      "asymmetry": { "value": 0.093, "concern": false },
      "border": { "value": 0.249, "concern": false },
      "color": { "value": 0.363, "concern": true, "label": "Color: CV=0.363 | blue-gray(15%) Concerning" },
      "diameter_px": 636.0,
      "mask_area_fraction": 0.309,
      "overall_visual_concern": "LOWER VISUAL CONCERN",
      "quality_flags": [],
      "explanation_ready_features": { "...": "raw values + a diameter caveat string" }
    }
  ],
  "primary_instance_overall_visual_concern": "LOWER VISUAL CONCERN",
  "config_version": "v3-<frozen-name>"
}
```

`segmentation_status` is one of `SINGLE_LESION_DETECTED`,
`MULTI_LESION_DETECTED`, `NO_DETECTION`, `FAILED`. The LLM should handle all
four — `NO_DETECTION` and `FAILED` mean there is nothing to summarize except
a prompt to retake the photo; `MULTI_LESION_DETECTED` means more than one
spot was found and the LLM should say so rather than picking one, matching
the evaluation-side policy of never silently collapsing multiple lesions
into a single answer (`MULTI_LESION_AMBIGUOUS`).

## What the LLM must do

Translate the deterministic CV findings (`asymmetry`, `border`, `color`,
`diameter_px`, `mask_area_fraction`, `overall_visual_concern`,
`quality_flags`) into plain language a non-clinician can follow. Explain
*what was measured*, not *what it means medically*.

## What the LLM must never do

- Never output, or paraphrase into, "melanoma", "cancer", "malignant",
  "benign", "you have/don't have skin cancer", or any diagnostic claim.
  Those words exist only in this project's own evaluation code
  (`Evaluation_3500/`, `Evaluation_FinalTargeted/`) that scores predictions
  against labeled research data — never in anything a real user sees.
- Never translate `LOWER VISUAL CONCERN` as "benign", "safe", "no cancer",
  or "no need to see a doctor."
- Never translate `ELEVATED VISUAL CONCERN` as "melanoma", "cancer", or
  "definitely malignant."
- Never state or imply `diameter_px` is a physical measurement in
  millimeters — the pipeline has no validated pixel-to-mm calibration
  (see Limitations in the final report). If a user asks how big the spot
  is in real units, say that isn't measurable from the photo alone.
- Never claim the assessment is validated, clinically tested, or
  equivalent to a professional exam.

## What the LLM should communicate

Regardless of concern level, the app can encourage the user to have any
new, changing, or personally-concerning spot looked at by a clinician —
that recommendation should not depend on `overall_visual_concern`, since
this tool is explicitly not diagnostic in either direction. A `LOWER VISUAL
CONCERN` result is a statement about what the silhouette/color analysis did
*not* flag, not a clearance.

## Evolution (Phase 10) handoff

`pipeline_v3.evolution.compare_records(earlier, later)` returns simple
numeric deltas (asymmetry/border/color/diameter_px/mask_area_fraction) plus
a mandatory `caveat` string. The LLM must surface that caveat whenever it
discusses a change over time — camera distance, angle, crop, and resolution
can all move `diameter_px` without the lesion changing size. Frame any
diameter-based comparison as "the photographed size appears to have
changed," never "the lesion has grown."
