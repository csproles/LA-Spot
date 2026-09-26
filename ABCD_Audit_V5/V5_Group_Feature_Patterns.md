# V5 Group-Level Feature Patterns (TP/TN/FP/FN)

Follow-up to the main V5 ABCD audit. Analysis-only — no change to the model, features, thresholds, or
locked-test results. Built from `V5_locked_test_master.csv` (n=615, independently reconstructed and
verified to reproduce the published TP 98 / TN 340 / FP 137 / FN 40 exactly) and the 42
`VisualReview/*/feature_values.json` case files (a verified subset of the same 615 rows).

**Methodology note on "unusually high/low"**: per-case percentiles are computed against the case's own
**true-class population** (benign = TN+FP, n=477; malignant = TP+FN, n=138) rather than the whole
615-row set — e.g. a false positive's `B_circularity` is compared to the distribution of `B_circularity`
among all *benign* cases (TN+FP), not the whole cohort. This is the framing the audit brief itself asked
for ("more useful than simply saying the raw value is high") and is what every `_Unusually_High/Low`
number in `V5_Group_Feature_Patterns.csv` and `V5_VisualReview_Unusual_Features.csv` uses. ≥90th
percentile = unusually high, ≤10th = unusually low, plus a stricter Tukey-fence (Q3+1.5·IQR /
Q1-1.5·IQR) outlier flag used for case-level reporting. Group-vs-group directional claims (`TN_vs_FP`,
`TP_vs_FN`, etc.) use a Mann-Whitney U test (two-sided, p<0.05) plus a minimum effect size
(|AUC-equivalent − 0.5| ≥ 0.05 for a direction label, ≥ 0.15 for "STRONG"); a full descriptive-stats
table (count/mean/median/std/Q1/Q3/min/max for all 17 features × 7 groups) is in
`V5_Group_Level_Descriptive_Stats.csv`.

---

## 1. False Positive patterns

Ranked by effect size vs. TN (full table, all 17 features: `_fp_ranked_supporting.csv`). FP cases —
benign lesions the model flagged as elevated-concern — are **not random misses**. On almost every
feature that discriminates malignant from benign at all, FP cases sit on the *malignant* side of the
benign distribution:

| Feature | FP Median | TN Median | P(FP > TN) | FP unusually high (%) |
|---|---|---|---|---|
| `lesion_fraction` | 0.230 | 0.028 | **0.89** | 32.8% |
| `D_px_normalized` | 0.687 | 0.232 | **0.89** | 32.1% |
| `turning_angle_std` | 0.320 | 0.216 | 0.83 | 26.3% |
| `B_circularity` | 0.310 | 0.217 | 0.76 | 24.8% |
| `A_value` | 0.170 | 0.135 | 0.68 | 24.1% |
| `lab_b_std` | 5.03 | 4.53 | 0.60 | 17.5% |
| `color_entropy` | 1.43 | 1.24 | 0.65 | 17.5% |
| `solidity` (inverse) | 0.938 | 0.969 | 0.25 (FP *lower*) | (5.8% high / **27.0% low**) |

FP cases cluster most strongly around **framing/diameter** (`lesion_fraction`, `D_px_normalized` — the
two largest effect sizes in the entire table, larger even than the overall malignant-vs-benign
separation) and **border shape** (`turning_angle_std`, `B_circularity`, and correspondingly low
`solidity`). Asymmetry (`A_value`) and color-spread (`lab_a_std`/`lab_b_std`/`color_entropy`) contribute
a real but smaller effect. `skin_contrast` shows no clear FP-vs-TN split at this sample size (p=0.10) —
consistent with it being a modest, population-level effect (Part 4 of the main audit) rather than
something that cleanly separates specific error cases. `dark_fraction`/`bluegray_fraction` show no
FP-specific pattern despite elevated "unusually high" percentages — driven by their step-function floors
(any nonzero value jumps straight past the 90th percentile in a mostly-zero distribution), exactly the
`POSSIBLE_OUTLIER_DRIVEN_PATTERN` risk flagged in the main audit. `confidence` and `eccentricity` show no
FP pattern at all, as expected.

**Interpretation**: false positives are not a "confused/borderline" population — they are specifically
the benign lesions that most resemble the malignant profile on size/framing and border-shape features.
This is a much stronger, more literal confirmation of the `lesion_fraction`/`D_px_normalized`
framing-confound concern from the main audit: these two features alone show the single largest
FP-vs-TN gap of any feature pair, larger than any genuine ABCD signal in the set.

## 2. False Negative patterns

Ranked by effect size vs. TP (full table: `_fn_ranked_supporting.csv`). The mirror image of the FP
pattern: FN cases — malignant lesions the model called lower-concern — sit on the *benign* side of the
malignant distribution across nearly the same feature set:

| Feature | FN Median | TP Median | P(TP > FN) | FN unusually low (%) |
|---|---|---|---|---|
| `lesion_fraction` | 0.065 | 0.320 | **0.84** | 30.0% |
| `D_px_normalized` | 0.365 | 0.785 | **0.84** | 30.0% |
| `turning_angle_std` | 0.241 | 0.346 | 0.77 | 27.5% |
| `C_value` | 0.314 | 0.395 | 0.77 | 22.5% |
| `color_entropy` | 1.183 | 1.678 | 0.75 | 22.5% |
| `lab_a_std` | 2.88 | 3.99 | 0.74 | 25.0% |
| `B_circularity` | 0.234 | 0.338 | 0.73 | 20.0% |
| `solidity` (inverse) | 0.962 | 0.921 | 0.29 (FN *higher*) | (7.5% high / 0.0% low) |

FN cases look **smaller/less-zoomed** (`lesion_fraction`, `D_px_normalized` — again the largest effect
sizes), **smoother-bordered** (low `turning_angle_std`, low `B_circularity`, correspondingly high
`solidity`), **less colorful** (low `C_value`, `color_entropy`, `lab_a_std`, `lab_b_std`), and
**more symmetric** (`A_value` lower, though a smaller effect here than for border/color). None of FN
cases fall in the "unusually high" tail for `lesion_fraction`/`D_px_normalized`/`color_entropy`/`C_value`
at all (0.0%) — the missed melanomas are concentrated at the *low* end of essentially every feature this
model relies on to flag concern.

**Interpretation**: this directly answers the brief's example question — yes, FN cases genuinely do tend
to look more symmetric, smoother-bordered, and less colorful, and this is not a weak or ambiguous
pattern; it is the single most consistent story in this entire analysis. `skin_contrast` again shows no
clear FN-vs-TP split at this sample size (p=0.125), consistent with Part 1.

## 3. True Positive patterns

TP cases are, essentially, the "textbook melanoma" profile: correctly-caught malignant lesions show
significantly *higher* `A_value`, `B_circularity`, `C_value`, `color_entropy`, `lab_a_std`, `lab_b_std`,
`turning_angle_std`, `lesion_fraction`, `D_px_normalized`, and *lower* `solidity` than FN cases (same
feature list as Section 2, opposite framing). The features that most consistently distinguish TP from FN
are the same ones with the largest FN-vs-TP effect sizes above: `lesion_fraction`/`D_px_normalized`
first, then `turning_angle_std`/`B_circularity`/`solidity` (border), then the color group.

## 4. True Negative patterns

TN cases are the mirror: consistently *low* values across the same feature set, and the one feature
where TN shows its own distinct "high" signature is `solidity` (TN median 0.969 vs. FP median 0.938,
P(TN>FP)=0.75) — genuinely smooth, convex borders are the strongest single marker of a correctly-called
benign lesion. `dark_fraction`/`bluegray_fraction`/`red_fraction` are at or near 0 for the overwhelming
majority of TN cases (as for every group), consistent with their sparse/step-function nature rather than
being a TN-specific signal.

## 5. Strongest useful features

Ranked by combined evidence (large, consistent, statistically significant effect across *both* the
FP-vs-TN and FN-vs-TP comparisons, in the clinically-expected direction, replicating the main audit's
independent benign-vs-malignant findings):

1. **`turning_angle_std`** — strong, consistent signal in both error comparisons (TN-vs-FP p<0.0001,
   TP-vs-FN p<0.0001), correct direction throughout, no known implementation concern.
2. **`B_circularity`** — very strong signal in both comparisons; genuinely useful as a *univariate*
   measurement (remember the main audit's caveat: its multivariate model coefficient is sign-flipped due
   to multicollinearity — the raw feature carries real information, the model's use of it is confused).
3. **`solidity`** — the single strongest class-separating feature in the whole set (main audit) and the
   clearest border marker here (TN median 0.969 vs FP 0.938; FN median 0.962 vs TP 0.921) — border
   smoothness vs. irregularity is, by this analysis, the most reliable ABCD-aligned signal V5 has.
4. **`color_entropy`** and **`lab_a_std`/`lab_b_std`** — the color-spread trio shows a strong, consistent
   FP/FN split with no known implementation concern (unlike the color-fraction features).
5. **`A_value`** — a real, statistically strong signal in both comparisons, though weaker than the border
   group and carrying the segmentation-sensitivity caveats from the main audit's Part 6.

## 6. Weakest features

1. **`eccentricity`** — `NO_CLEAR_PATTERN` in every comparison run here (benign-vs-malignant, FP-vs-TN,
   FN-vs-TP all non-significant) — the most thoroughly unsupported feature in the whole 17.
2. **`confidence`** — `NO_CLEAR_PATTERN` throughout, as expected (not an ABCD signal by design).
3. **`bluegray_fraction`** / **`dark_fraction`** — no clear FP/FN pattern; any apparent "unusually high"
   percentage is an artifact of their step-function floors on an otherwise near-all-zero distribution
   (`POSSIBLE_OUTLIER_DRIVEN_PATTERN`), not a real population signal.
4. **`D_px`** (raw pixels) — `HIGH_VARIABILITY` only, no clear direction in any comparison; its
   normalized counterpart (`D_px_normalized`) carries essentially all the usable signal.
5. **`skin_contrast`** — real at the full benign-vs-malignant level (main audit) but too weak to produce
   a significant FP-vs-TN or FN-vs-TP split at this sample size — a modest, population-level effect, not
   a case-discriminating one.

## 7. Features likely affected by segmentation/framing

- **`lesion_fraction` / `D_px_normalized`** — by far the largest effect sizes for BOTH the FP-vs-TN and
  FN-vs-TP comparisons (P(FP>TN)=0.89, P(TP>FN)=0.84) — stronger than any other feature pair, including
  the "genuine" ABCD signals. Combined with the main audit's rho=0.989 redundancy finding, this is the
  single most important pattern in this whole analysis: the model's two best-looking features by raw
  effect size are the two most likely to be measuring photo framing/zoom rather than biology.
- **`A_value`, `D_px`, `B_circularity`, `solidity`, `turning_angle_std`** — all inherit the largest-
  contour-only, no-small-lesion-floor, edge-clamped-crop risks documented in the main audit's Parts 6/7.
  The case-level flags (`V5_VisualReview_Unusual_Features.csv`) should be cross-checked against
  `Possible_Segmentation_Influence` for lesions in the smallest 5% of their true class's area — a
  concrete, checkable trigger for "is this pattern real or a segmentation artifact."
- **`red_fraction` / `bluegray_fraction` / `dark_fraction`** — sparse, step-function-driven; any observed
  FP/FN "pattern" is disproportionately influenced by a handful of nonzero cases (`Zero_Value_Fraction_All`
  0.63–0.72 for these three).

## 8. Top 5 findings

1. **FP and FN cases are near-mirror images of each other** on nearly the entire feature set — FPs look
   like "benign lesions borrowing the malignant profile," FNs look like "malignant lesions borrowing the
   benign profile." This is a clean, strongly-supported, symmetric story, not scattered noise.
2. **`lesion_fraction`/`D_px_normalized` show the single largest effect sizes anywhere in this analysis**
   — larger than genuine border/color signals — which is exactly what you'd expect if a chunk of the
   model's apparent performance is riding on photo framing rather than true diameter or lesion size. This
   is the most consequential single finding for follow-up work.
3. **Border-shape features (`solidity`, `turning_angle_std`, `B_circularity`) form the most reliable
   genuinely-ABCD-aligned signal**, consistent across every comparison run, with no sparse-data or
   step-function caveats — the strongest candidates for "this is real."
4. **The color-fraction features (`red/bluegray/dark_fraction`) contribute essentially nothing
   case-specific** beyond a handful of outlier-driven percentile flags — confirming, with fresh evidence,
   the main audit's concern about their step-function floors.
5. **`eccentricity` and `confidence` show zero pattern anywhere in this analysis** — not in the overall
   class comparison, not in FP-vs-TN, not in FN-vs-TP — the cleanest "no signal" result in the dataset.
