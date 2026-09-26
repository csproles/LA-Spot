# Multicollinearity Report — B_circularity / solidity / turning_angle_std

Phase 5 of the modeling brief. Development population only (n=2090 dev-cohort rows from
`dev_feature_table_v2.csv`); locked test set never opened.

Per the task instructions, this uses the **original** `B_circularity`, `solidity`, `turning_angle_std`
columns rather than their `_pc` counterparts — the prior agent already confirmed `solidity_pc` and
`turning_angle_std_pc` are byte-for-byte identical to the originals for all 2090 images, and
`B_circularity_pc` differs from `B_circularity` only by the original CSV's 3-decimal rounding (max
abs diff 0.0005, mean 0.00025, re-verified directly here), so using the originals vs. `_pc` versions
makes no material difference to this analysis.

## Pairwise correlation matrix (n=2090, no missing values in these 3 columns)

### Pearson

| | B_circularity | solidity | turning_angle_std |
|---|---|---|---|
| **B_circularity** | 1.0000 | -0.8704 | 0.7594 |
| **solidity** | -0.8704 | 1.0000 | -0.6394 |
| **turning_angle_std** | 0.7594 | -0.6394 | 1.0000 |

### Spearman

| | B_circularity | solidity | turning_angle_std |
|---|---|---|---|
| **B_circularity** | 1.0000 | -0.8550 | 0.7437 |
| **solidity** | -0.8550 | 1.0000 | -0.6958 |
| **turning_angle_std** | 0.7437 | -0.6958 | 1.0000 |

All three border/shape features are strongly, monotonically correlated with each other (|rho| in the
0.64-0.87 range). This is expected: `B_circularity = 1 - 4*pi*area/perimeter^2`, `solidity =
area/convex_hull_area`, and `turning_angle_std` (std of discrete turning angle over a 100-point
resampled contour) are all different mathematical lenses on the same underlying property — how much
a contour deviates from a smooth convex shape. They are not literally redundant (none of the pairwise
correlations is +-1.00, unlike e.g. `isoperimetric_ratio` vs `B_circularity` which the prior ablation
study found to be an exact rho=1.00 duplicate and pruned), but they share a large amount of variance.

## Context: frozen V5's own standardized coefficients

From `ABCD_Audit_V5/_source_from_research_branch/Evaluation_V5Candidate/v5_coefficients.csv` (cited,
not refit):

| feature | standardized coefficient | abs rank |
|---|---|---|
| solidity | -0.72784 | 1 (largest magnitude) |
| D_px_normalized | +0.71966 | 2 |
| B_circularity | -0.62096 | 3 |
| turning_angle_std | +0.09386 | 13 |

`solidity` and `B_circularity` are V5's #1 and #3 most influential features by magnitude, both
negative. `turning_angle_std` is comparatively weak (rank 13) and, notably, has the **opposite sign**
from `B_circularity` despite `turning_angle_std` and `B_circularity` being *positively* correlated
with each other (Pearson +0.76) — a classic multicollinearity symptom (redundant predictors can end
up "splitting" or inverting each other's coefficient signs relative to what a univariate model would
show).

## B_circularity's univariate direction vs. its model coefficient (the "sign-flip")

`B_circularity`'s point-biserial correlation with `ground_truth_binary` on its own (2090 rows, no
other features): **r = +0.2991** (p = 1.8e-44) — i.e., on its own, higher `B_circularity` (more
border irregularity, since `B_circularity = 1 - 4*pi*area/perimeter^2` grows as the boundary deviates
from a smooth circle) is clearly, strongly associated with **malignant** ground truth, exactly the
direction the ABCD "Border irregularity" criterion predicts clinically.

But in **frozen V5's own multivariate model**, `B_circularity`'s coefficient is **-0.62096** —
negative, i.e. the multivariate model has it pointing in the *opposite* direction from its own
univariate signal. This is the "sign-flip" flagged in the prior audit
(`ABCD_Audit_V5/V5_Part7_Border_Implementation_Audit.md` and `V5_ABCD_Weak_Feature_Ranking.md`).

## VIF within the FINAL feature set

The FINAL recommended feature set (14 features; see `final_recommendation.md`) still contains all
three of `B_circularity`, `solidity`, `turning_angle_std` unchanged (none of EXP2's weak-feature
pruning or later experiments touched this trio — none of them were ever flagged as individually weak;
`turning_angle_std`'s univariate weakness is masked by its strong pairwise correlation with the other
two, which is exactly what VIF is designed to detect). `statsmodels` is not installed in `.venv`, so
VIF was computed manually with scikit-learn's `LinearRegression`
(`VIF_i = 1 / (1 - R^2)`, where `R^2` comes from regressing feature `i` on all 13 other FINAL-set
features; computed on the 2089 FINAL-population rows with no missing values in any FINAL feature):

| feature | R² (vs. other 13 FINAL features) | VIF |
|---|---|---|
| B_circularity | 0.8558 | **6.94** |
| solidity | 0.8363 | **6.11** |
| turning_angle_std | 0.6765 | **3.09** |

Using the common rule-of-thumb bands (VIF < 5 = low concern, 5-10 = moderate, >10 = severe):
`B_circularity` and `solidity` sit in the **moderate** range (~6-7) — driven almost entirely by their
mutual correlation and, to a lesser extent, their correlation with `turning_angle_std` and other
FINAL-set border/shape/color features. `turning_angle_std` is lower (~3.1), consistent with it having
weaker pairwise correlation with the other two (Pearson 0.64-0.76 vs. -0.87 between `B_circularity`
and `solidity`). None reach the "severe" (>10) band, but the moderate VIFs for `B_circularity`/
`solidity` confirm real, non-trivial shared variance — enough to plausibly explain coefficient
instability/sign inversion in a linear model, without being so extreme that the model is
uninterpretable.

## Does the FINAL configuration change B_circularity's sign-flip behavior?

**No — the sign-flip persists.** A plain `LogisticRegression(max_iter=1000, random_state=20260918)`
was fit on a `StandardScaler`-transformed version of the FINAL 14-feature set, on all 2089 available
FINAL-population dev rows (1 row dropped for missing `skin_contrast_pc`), purely for coefficient
inspection (this fit is not used for any of the CV/threshold numbers reported elsewhere — same
convention as `build_v5_candidate.py`'s reference-model fit). `B_circularity`'s coefficient in this
FINAL-feature-set fit is **-0.7190** — still negative, still contradicting its own univariate
positive (+0.2991) direction, and if anything slightly *larger in magnitude* than frozen V5's -0.62096.

Full FINAL-model standardized coefficients (all 14 features, sorted by |coefficient|):

| feature | standardized coefficient |
|---|---|
| D_px_normalized | +0.8237 |
| solidity | -0.7389 |
| B_circularity | -0.7190 |
| lab_b_std_pc | +0.2897 |
| C_value_pc | +0.2786 |
| A_value_pc | +0.2656 |
| lab_a_std_pc | +0.2630 |
| entropy_L | -0.1998 |
| red_fraction_pc | +0.1348 |
| turning_angle_std | +0.1156 |
| entropy_b | +0.1125 |
| skin_contrast_pc | -0.1113 |
| confidence | -0.0901 |
| entropy_a | +0.0519 |
| *intercept* | -1.0505 |

**Interpretation**: none of the changes tested in this modeling phase (primary-component-consistent
recomputation, weak-feature pruning, channel-specific entropy, size-feature redundancy resolution,
split major/minor asymmetry, fractal-dimension border feature) were targeted at, or had any material
effect on, the B_circularity/solidity/turning_angle_std multicollinearity itself — that trio was never
one of the "weak" features under test (their combined signal, via `solidity` and `B_circularity`
specifically, is in fact among the model's strongest). Fixing the sign-flip would require either (a)
choosing to keep only one of `B_circularity`/`solidity` (dropping the more redundant one), (b) an
explicit interaction/composite feature engineered to combine them, or (c) switching to a model family
less sensitive to correlated linear predictors (e.g., a regularized model with an L1/L2 penalty tuned
specifically for this, or a tree-based model) — none of which were in scope for this modeling brief
(which specifically asked to characterize, not resolve, this multicollinearity). This is flagged here
as a known, unresolved limitation of the FINAL candidate, not silently glossed over.
