# Part 12 — V5 ABCD Improvement Recommendations

**None of these are implemented here.** V5's model, features, thresholds, and locked-test results are
unchanged. Every recommendation below is a candidate for a NEW experimental version, evaluated only on
development/validation data, per the instructions governing this audit.

Each entry: current feature → observed problem → evidence → potential improved method → whether it
should be tested → what a dev-only experiment would need. Ordered roughly by how directly actionable
the evidence is (cheapest/most-supported first).

---

## 1. `B_circularity` — misleading name, sign-flipped in the multivariate model

**Observed problem**: stored value is actually `1 - circularity` (border *irregularity*), not
circularity, despite the name. Its standalone (univariate) association with melanoma is positive and
consistent across both the development cohort (AUC 0.683) and the locked test set (AUC 0.680) — the
correct, clinically-expected direction. But the frozen model's standardized coefficient for this
feature is **negative** (-0.621, rank 3 of 17 by magnitude) — directly contradicting its own univariate
signal.

**Evidence**: `V5_ABCD_Feature_Inventory.csv` (row `B_circularity`); `V5_Feature_Failure_Analysis.csv`
(locked-test AUC 0.680, direction `higher_in_malignant`); `Evaluation_V5Candidate/v5_coefficients.csv`
(coefficient -0.62096); `Evaluation_FeatureEngineering/redundancy_report.md` (`B_circularity` is a
perfect monotonic transform of the excluded `isoperimetric_ratio`, rho=1.00, and correlates with
`solidity`/`turning_angle_std`, both also in the model) — the sign flip is a textbook multicollinearity
artifact, not a data problem.

**Potential improved method**: (a) rename the stored value to `B_irregularity` for clarity (pure
code-quality fix, zero statistical effect); (b) as a real modeling experiment, test whether dropping
`B_circularity` (keeping `solidity` + `turning_angle_std`, which already carry overlapping
border-irregularity signal without the sign-flip problem) changes locked-test-equivalent performance on
development data; (c) literature (Part 11, B2/B3) offers two concrete alternative/complementary border
measures — Lee/McLean/Atkins' area-based Irregularity Index (a refinement of what `solidity` already
approximates) and a DFT-magnitude descriptor of the already-computed 100-point resampled contour
(low implementation cost, reuses existing infrastructure) — either could replace or supplement
`B_circularity` with something that doesn't share its collinearity problem.

**Should be tested?** Yes — this is the single cheapest, best-evidenced candidate in this list (a) is
essentially free, (b)/(c) are standard ablation experiments using code the model-fitting pipeline
already has.

**Dev-only experiment needed**: Re-run `Evaluation_FeatureEngineering`-style GroupKFold(5) CV, comparing
(i) current 17 features, (ii) 17 features minus `B_circularity`, (iii) 17 features with `B_circularity`
replaced by a DFT-magnitude border descriptor, on OOF ROC-AUC / balanced accuracy / sensitivity /
specificity at a re-swept threshold. Development cohort only; no locked-test access.

---

## 2. `lesion_fraction` / `D_px_normalized` — near-duplicate features, possible photo-framing confound

**Observed problem**: Spearman rho=0.989 between the two (already documented by the project's own prior
research as "the single most important redundancy finding"). Both show nearly identical, strong
locked-test AUCs (0.745 vs. 0.745) — both are essentially "how much of the photo frame the lesion
occupies," which conflates true lesion size with photography framing/zoom convention, especially in a
multi-source dataset like ISIC.

**Evidence**: `Evaluation_FeatureEngineering/redundancy_report.md`; `V5_Feature_Failure_Analysis.csv`
(both AUC 0.745, both flagged category D); `V5_Part9_Diameter_Implementation_Audit.md` (`D_px_normalized`
section).

**Potential improved method**: This is fundamentally a *dataset* question, not a formula question — no
single literature citation fixes a framing confound. Two dev-only paths: (a) keep only one of the two
(the model already spends coefficient weight on both for overlapping information — dropping one is a
straightforward complexity-reduction test); (b) if ISIC per-source metadata (contributing
institution/imaging protocol) is available, test whether `lesion_fraction`/`D_px_normalized`'s
association with ground truth *survives* controlling for image source — if the signal collapses once
source is controlled for, that's strong evidence it's a framing artifact, not a real diameter/size
signal.

**Should be tested?** Yes, but this is evidence-gathering, not a drop-in replacement — (a) is a quick
experiment; (b) is more involved (requires ISIC metadata joins not currently in this pipeline's data) and
should be scoped as a separate investigation before any feature change.

**Dev-only experiment needed**: (a) same CV setup as above, drop `D_px_normalized` (keep
`lesion_fraction`, since it's the pre-existing V4 feature with a longer track record) and compare
OOF metrics; (b) a data-only analysis (no model refit) checking whether `lesion_fraction`/
`D_px_normalized` distributions differ by ISIC source collection independent of ground truth label.

---

## 3. `skin_contrast` — reproducible but clinically counter-intuitive direction

**Observed problem**: lower `skin_contrast` (less contrast with immediately surrounding skin)
associates with melanoma, not higher — and this direction is consistent and statistically significant
on both the development cohort (AUC 0.429) and, independently, the locked test set (AUC 0.435,
p=0.020). This contradicts the naive intuition that a melanoma should stand out more from surrounding
skin.

**Evidence**: `V5_Feature_Failure_Analysis.csv` (`skin_contrast` row); `V5_Part8_Color_Implementation_Audit.md`.

**Potential improved method**: This is flagged as needing **human visual review before any formula
change is even considered** (see Part 5/13) — the pattern is real and reproducible, so the priority is
understanding *why* (e.g., larger/more diffuse-bordered malignant lesions in this cohort push the
sampling ring further from any sharp edge; or a genuine dermoscopic pattern about melanocytic lesion
color blending) before deciding whether the current ring-based formula needs to change at all. Literature
(Part 11, B-adjacent finding) offers one concrete alternative if a change is later warranted: Kaya et
al.'s multi-scale, texture-homogeneity-based border-cutoff-abruptness measure, a generalization of the
single-ring LAB-distance approach `skin_contrast` currently uses.

**Should be tested?** Not yet as a formula change — first needs the human visual review this audit's
Part 5 material was built for. If the visual review confirms a plausible mechanism (e.g., ring radius
scaling with lesion size systematically undersamples true peri-lesional skin for large lesions), a
formula change (e.g., a fixed-radius ring, or the multi-scale approach above) becomes a well-motivated
dev-only experiment.

**Dev-only experiment needed**: (only after visual review) compare `skin_contrast`'s current
area-scaled-ring formula against a fixed-radius-ring variant, and/or the Kaya-style multi-scale texture
measure, on development-cohort OOF AUC contribution.

---

## 4. `dark_fraction`, `bluegray_fraction`, `eccentricity` — weak/near-chance discrimination

**Observed problem**: `eccentricity` shows chance-level univariate signal on both cohorts (dev AUC
0.503, locked AUC 0.505, p=0.86) yet is retained with a non-trivial multivariate coefficient (-0.100),
apparently picked up purely via correlation with excluded geometry features. `dark_fraction` (locked AUC
0.528, p=0.25, full IQR overlap) and `bluegray_fraction` (locked AUC 0.557, small effect) are both
heavily shaped by hard-coded step-function floors (0.12/0.15) that produce near-identical (0.0) medians
for both classes.

**Evidence**: `V5_Feature_Failure_Analysis.csv`; `V5_Part10_Weak_Feature_Ranking.md`;
`V5_Part8_Color_Implementation_Audit.md` (step-function floor explanation).

**Potential improved method**: For the color-fraction features, Umbaugh/Moss/Stoecker's data-driven
color-clustering approach (Part 11, C1) directly targets the "hand-tuned fixed thresholds" root cause —
replacing fixed red/bluegray/dark thresholds with a per-image adaptive cluster-based description. For
`eccentricity`, given it shows no standalone signal on two independent cohorts, the direct dev-only test
is simply: does removing it change performance at all (a near-zero-cost ablation, since if it's truly
adding nothing beyond what correlated features already contribute, dropping it reduces complexity for
free).

**Should be tested?** Yes for `eccentricity` (cheap, well-evidenced removal candidate). For the
color-fraction features, the clustering-based replacement is a larger change and should be scoped as its
own ablation arm rather than assumed to be an improvement.

**Dev-only experiment needed**: Ablation CV comparing (i) current 17 features, (ii) 17 minus
`eccentricity`, (iii) `red/bluegray/dark_fraction` replaced by a k-means color-cluster-count feature
(Part 11, C1) — same GroupKFold(5), same threshold-selection rule as the original V4/V5 methodology.

---

## 5. `color_entropy` — literature suggests a specific, low-cost variant worth testing

**Observed problem**: no problem observed in this pipeline's own data (`color_entropy` performs
consistently: dev AUC 0.650, locked AUC 0.638) — this recommendation is opportunity-driven, not
failure-driven.

**Evidence**: `V5_ABCD_Literature_Review.md`, entry C3 (Martínez-Ortega & Martinez-Jaramillo, Cureus
2026) — a peer-reviewed paper specifically arguing that global (single joint-histogram) Shannon entropy
under-quantifies polychromia relative to channel-specific entropy, under real-world imaging conditions.

**Potential improved method**: compute per-channel (L, a, b individually) 1D entropy in addition to the
existing joint 2D (a,b) entropy, reusing the LAB pixel arrays `color_entropy`/`lab_a_std`/`lab_b_std`
already extract.

**Should be tested?** Yes, as a low-cost addition/comparison — implementation difficulty is rated "low"
in the literature review since it reuses existing LAB infrastructure. Note the paper's framing is
smartphone/"real-world" imaging, so its critique may be less applicable to ISIC's calibrated dermoscopic
images specifically — this should be validated on ISIC data, not assumed to transfer.

**Dev-only experiment needed**: add per-channel entropy as one or more new candidate features; run the
same ablation-CV procedure as the original `Evaluation_FeatureEngineering` study to see whether it adds
independent signal beyond the existing `color_entropy`/`lab_a_std`/`lab_b_std` trio (a real risk, given
`lab_a_std`/`lab_b_std` already capture per-channel spread — per-channel entropy may turn out to be
redundant with them, exactly the kind of redundancy the original ablation study was careful to check for
elsewhere).

---

## 6. `A_value`, `D_px` — segmentation-sensitivity risks needing case-level confirmation before any change

**Observed problem**: aggregate statistics look fine for both (`A_value` locked AUC 0.655; `D_px` locked
AUC 0.576) — the concern is specific, code-confirmed edge cases: `A_value`'s crop window is silently
off-center for lesions near the image border (Part 6), and both `A_value` and `D_px` inherit
largest-contour-only logic with no small-contour floor or false-positive-region guard (Parts 6/7/9).

**Evidence**: `V5_Part6_Asymmetry_Implementation_Audit.md`, `V5_Part7_Border_Implementation_Audit.md`,
`V5_Part9_Diameter_Implementation_Audit.md`.

**Potential improved method**: not a formula change — a robustness fix: (a) reject/flag alignment when
the mask's bounding box comes within some margin of the image border, rather than silently clamping the
crop; (b) add a minimum-contour-size floor consistent with the one color features already have (20px)
and asymmetry already has (5px, arguably too low) to border/diameter features, which currently have none.

**Should be tested?** These are correctness/robustness fixes, not accuracy experiments in the usual
sense — they should first be confirmed as actually occurring in this dataset (via the Part 5 visual
review of edge-adjacent and tiny-lesion cases) before deciding whether to implement a guard, since a
guard that never fires is unnecessary complexity.

**Dev-only experiment needed**: after visual confirmation, a before/after comparison on development-
cohort OOF metrics with the edge-clamp and small-contour guards added, to confirm the fix doesn't
regress performance (it shouldn't, since it only changes behavior on cases the current code already
handles unreliably).

---

## Not recommended for further work right now

- **`solidity`, `turning_angle_std`, `lab_a_std`, `lab_b_std`, `color_entropy`, `A_value`**: consistently
  performing, no code-level defect found — no action needed beyond the opportunistic C3 literature test
  above for `color_entropy`.
- **`confidence`**: correctly near-chance by design (a YOLO quality covariate, not meant to discriminate)
  — no change warranted.
- **Physical mm calibration for `D`**: the existing hair-based calibration code
  (`revised_abcd/revised_diameter.py`) is real, careful engineering but was found to have essentially
  zero usable evidence in this dataset (0/40 images had a hair-shaped majority, per its own docstring).
  Part 11's D2 (GraphDerm ruler-detection calibration) is the one literature finding that could plausibly
  unlock physical units, but it requires visible ruler markings in the source images and is an arXiv
  preprint, not peer-reviewed — this should be scoped as a data-availability investigation (do any ISIC
  images in this cohort actually contain rulers?) before any modeling work, not started as a modeling
  project.
