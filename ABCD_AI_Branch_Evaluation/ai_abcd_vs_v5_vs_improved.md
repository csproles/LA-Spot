# ai-abcd vs. Frozen V5 vs. Improved ABCD FINAL

All numbers below come from the SAME development cohort (n=2090, or n=2089 for the
two logistic-regression pipelines after their standard NaN-drop), the SAME
`group_id` GroupKFold(5) folds, and the SAME underlying measurement code (confirmed
byte-identical across branches — see `ai_abcd_feature_inventory.md`). **The locked
615-image test set was not used anywhere in this comparison.**

## Side-by-side table

| Pipeline | Feature_Count | Threshold | TP | TN | FP | FN | Accuracy | Balanced_Accuracy | Sensitivity | Specificity | Precision | F1 | ROC_AUC | CV_Mean_Balanced_Accuracy | CV_STD_Balanced_Accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Frozen V5** | 17 | 0.25 | 492 | 977 | 474 | 146 | 0.7032 | 0.7222 | 0.7712 | 0.6733 | 0.5093 | 0.6135 | 0.8008 | 0.7226 | 0.0078 |
| **Improved ABCD FINAL** | 14 | 0.30 | 460 | 1085 | 366 | 178 | 0.7396 | 0.7344 | 0.7210 | 0.7478 | 0.5569 | 0.6284 | 0.8016 | 0.7348 | 0.0096 |
| **ai-abcd (ANY of A/B/C)** | 3 booleans | n_concerns≥1 | 638 | 5 | 1446 | 1 | 0.3077 | 0.5009 | 0.9984 | 0.0034 | 0.3061 | 0.4686 | N/A* | 0.5009 | 0.0031 |
| **ai-abcd (MAJORITY, 2-of-3)** | 3 booleans | n_concerns≥2 | 582 | 354 | 1097 | 57 | 0.4478 | 0.5774 | 0.9108 | 0.2440 | 0.3466 | 0.5022 | N/A* | 0.5775 | 0.0152 |

\* No true continuous score exists (see below). A supplementary, clearly non-standard
pseudo-AUC from the ordinal 0–3 concern count = **0.6447** for both rows (same
underlying score, only the binarization threshold differs) — not comparable to
V5/Improved's calibrated-probability AUCs and not used in any ranking below.

`ai_abcd_performance.csv` has the machine-readable version of this table plus the
pseudo-AUC column; `ai_abcd_per_image_judgments.csv` has every image's individual
A/B/C concern flags and raw values for spot-checking.

## 1. What `ai-abcd` calculates differently

Nothing at the measurement level — `A_value`, `border_irreg`/`B_circularity`,
`n_significant_defects`, `color_cv`/`C_value`, and the four color-fraction values are
computed by byte-identical code to what V5/Improved already use. What differs is
purely the **decision layer**: instead of a trained 17-feature logistic regression
(V5) or a tuned 14-feature logistic regression (Improved), `ai-abcd`'s
`clinical_abcd.py` applies four independent, literature-derived, never-tuned
threshold rules (one each for A/B/C, D always "N/A") and — critically — **never
combines them into a single verdict at all**. The "ANY"/"MAJORITY" rows above are an
evaluation-only addition to make a comparison possible; `ai-abcd` itself makes no
such combined call.

## 2. Better sensitivity?

**ai-abcd (ANY)**, at 0.9984 — but this is not a meaningful win; see #7.
**ai-abcd (MAJORITY)** is second at 0.9108. Both clear V5 (0.7712) and Improved
(0.7210) by a wide margin, entirely because both aggregation rules are far more
trigger-happy than either logistic regression.

## 3. Better specificity?

**Improved ABCD FINAL**, at 0.7478 — clearly best. V5 is second (0.6733).
**ai-abcd (MAJORITY)** is a distant third (0.2440), and **ai-abcd (ANY)** is
essentially non-functional as a benign-vs-malignant discriminator (0.0034 —
it flags 1446 of 1451 benign lesions).

## 4. Better balanced accuracy?

**Improved ABCD FINAL** (0.7344) > **Frozen V5** (0.7222) > **ai-abcd (MAJORITY)**
(0.5774) > **ai-abcd (ANY)** (0.5009 — statistically indistinguishable from a coin
flip). Neither `ai-abcd` aggregation is competitive with either logistic-regression
pipeline on this metric.

## 5. Better ROC-AUC?

**Not a fair comparison to make.** V5 (0.8008) and Improved (0.8016) both have true
calibrated-probability AUCs from a fitted model. `ai-abcd` has no continuous score;
its 0.6447 pseudo-AUC (from an ordinal 0–3 concern count, the coarsest possible
score) is reported only for context and should not be read as "worse AUC" in the
same sense — it is a different, much cruder kind of number entirely.

## 6. More stable across folds?

By raw CV standard deviation, **ai-abcd (ANY)** has the smallest std (0.0031) — but
this is because it is stuck at ~0.50 balanced accuracy in every fold, not because it
is a well-behaved model; a rule that always says "concerning" is trivially stable.
Among the pipelines that are actually discriminating between classes, **Frozen V5**
(std 0.0078) is marginally more stable than **Improved FINAL** (std 0.0096), and
**ai-abcd (MAJORITY)** is the least stable of the three real discriminators (std
0.0152).

## 7. Could any apparent gain come from leakage, preprocessing, framing, or dataset differences?

No apparent "gain" exists to explain away — `ai-abcd`'s aggregated numbers are worse,
not better, than V5/Improved on the metric that matters most for this comparison
(balanced accuracy). But the ANY-rule's extreme sensitivity (0.9984) is worth
explaining precisely so it isn't mistaken for a real finding: it is **not** leakage,
preprocessing, or framing — every input is the same cached, already-verified value
V5/Improved use. It is a direct, mechanical consequence of **OR-ing three
independently-lenient thresholds together**. Each of A/B/C's individual cutoffs was
chosen (per `clinical_abcd.py`'s own docstring) to be closer to "worth a closer look"
than "diagnostic," e.g. `A_value > 0.15` is *below* the benign population's own
median `A_value` (~0.139–0.141 per the main audit's dev-cohort statistics) — meaning
roughly half of ALL benign lesions already clear the "mild asymmetry" bar on
asymmetry alone, before border or color are even considered. Once any one of three
such lenient gates is enough to flag a case, false positives compound multiplicatively
rather than averaging out. This is exactly what a screening-oriented, per-criterion
"worth a second look" tool is *designed* to do — and exactly why it is not
comparable, as a classifier, to a calibrated logistic regression without the
aggregation caveats stated throughout this document.

## 8. Anything in `ai-abcd` worth incorporating into our improved pipeline?

Yes — two things, neither of which is the aggregation logic itself:

- **`n_significant_defects` (the convexity-defect "notch count" from
  `revised_border.score_border_experimental`)** is not currently one of Improved
  FINAL's 14 features and is not part of frozen V5 either — it already exists,
  cached, in `cohort_v2_results.csv`, unused by either of our logistic regressions.
  `clinical_abcd.py`'s framing of it as "the direct computational analogue of
  clinically described notching/scalloping... a more specific signal than the
  global shape" is a reasonable hypothesis and it costs nothing to test: this is a
  concrete, low-cost candidate for a development-only ablation in
  `ABCD_Improved_Experimental` (add `B_experimental_n_defects` as a 15th feature,
  same GroupKFold(5) harness, see whether it clears the CV-std bar the way the
  other Phase-2 candidates were tested).
- **The two-tier asymmetry framing** (mild vs. marked, at 0.15/0.35) is a reasonable
  alternative way to *read* `A_value` for human-facing explanation text, even though
  it performed poorly as a standalone classifier gate here — worth considering for
  any future clinician-facing report text, separate from the ML decision itself.

Everything else — the fixed, untuned thresholds and the (missing) combination logic
— is not recommended for adoption; it was not designed to be a standalone
classifier and evaluating it as one (which this document had to do, per the task) is
not a fair test of what it was actually built for.

---

## Is `ai-abcd` actually comparable to V5/Improved?

**Only partially, and only after being forced into a shape it wasn't designed for.**
It shares 100% of its segmentation and raw-measurement code with V5/Improved (a
genuinely fair, verified common foundation), but it is not itself a trained
classifier — it's a per-criterion clinical-note generator with no combined verdict
and no continuous score. Every number in the comparison table above that involves
`ai-abcd` depends on an aggregation rule this evaluation added, not one `ai-abcd`
specifies. Under either reasonable aggregation, it is clearly worse than both V5 and
Improved FINAL on balanced accuracy, specificity, and stability — the ANY rule in
particular is not a usable classifier (specificity 0.003). The one part worth
carrying forward is a single existing-but-unused feature (`n_significant_defects`),
as a cheap development-only ablation — not the branch's decision logic as a whole.
