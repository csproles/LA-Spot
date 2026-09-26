# V5 ABCD Feature Set — Literature Review (Part 11)

**Status: research reference only.** This document surveys peer-reviewed / established
literature on classical, interpretable computer-vision methods for quantifying the
dermoscopic ABCD(E) criteria (Asymmetry, Border, Color, Diameter), specifically to
identify possible *future* improvements or alternatives to the formulas used in the
frozen V5 feature set. **Nothing in this document modifies the current V5 pipeline.**
The V5 model (`StandardScaler + LogisticRegression`, 17 features, threshold 0.25,
trained/frozen on ISIC dermoscopic images) is treated as a fixed baseline throughout;
every entry below is evaluated *against* that baseline, not as a replacement decision.

End-to-end CNN/deep-learning melanoma classifiers are intentionally out of scope —
the search targeted classical feature-engineering methods for A, B, C, and D
individually, since that is the design space this pipeline occupies. All citations
below were located via web search on 2026-09-22; where a detail (e.g., exact page
range or author list) could not be independently confirmed from more than one source,
that uncertainty is flagged explicitly rather than presented as fact.

Current V5 baseline formulas, restated for comparison purposes in each entry:

- **A**: PCA principal-axis alignment of the binary lesion mask, then
  `1 - mean(IoU_horizontal_flip, IoU_vertical_flip)` after folding the aligned mask
  along its own horizontal/vertical axes.
- **B**: `B_circularity = 1 - 4π·Area/Perimeter²`; `solidity = contour_area / convex_hull_area`;
  `turning_angle_std` = std of discrete turning angle along a 100-point equal-arc-length
  resampled contour.
- **C** (LAB space): `C_value` = coefficient of variation of per-pixel LAB distance
  from sampled peri-lesional skin; `color_entropy` = Shannon entropy of a 16×16 2D
  histogram over (a,b); `lab_a_std`/`lab_b_std`; `red_fraction`/`bluegray_fraction`/`dark_fraction`
  (hand-tuned rules relative to sampled skin color); `skin_contrast` = LAB distance
  between mean lesion color and a thin outer ring.
- **D**: `D_px` = diameter of the minimum enclosing circle (raw pixels, never
  calibrated to mm); `D_px_normalized` = `D_px / sqrt(H·W)`.

---

## A — Asymmetry

### A1. Principal-Axes-Based Asymmetry Assessment (inertia-axis reflection, extended feature set)
- **Method name**: Principal Axes-Based Asymmetry Assessment Methodology
- **Category**: A
- **Citation**: Vasconcelos, Rosado, Ferreira (attribution found via search but not
  independently cross-confirmed — treat author order/spelling as tentative), "Principal
  Axes-Based Asymmetry Assessment Methodology for Skin Lesion Image Analysis," book
  chapter, Springer, 2014/2015 (chapter appears in a 2015 Springer volume, DOI prefix
  10.1007/978-3-319-14364-4_3). Source: https://link.springer.com/chapter/10.1007/978-3-319-14364-4_3
- **Formula/algorithm**: Symmetry axes are the principal axes of inertia (moments) of
  the binary lesion mask — the same starting point as V5's PCA alignment. The lesion
  is reflected across each principal axis individually, and an asymmetry index is
  computed per axis as the area of the non-overlapping region between the lesion and
  its reflection, expressed as a percentage of total lesion area. The paper extracts a
  much larger bank (310) of related asymmetry features and applies feature
  selection + ML classifiers (not a single closed-form score) to predict an asymmetry
  label, evaluated separately on dermoscopic (87% accuracy) vs. mobile-phone images
  (73.1%).
- **Required input**: Binary lesion mask (dermoscopic or clinical photo).
- **Advantages**: Directly comparable starting point to V5 (same principal-axis
  concept); area-of-non-overlap after reflection is arguably more intuitive/interpretable
  than V5's fold-and-IoU; validated across both dermoscopic and consumer-camera image
  domains, which matters if this pipeline is ever extended beyond ISIC dermoscopic
  images.
- **Limitations**: The 310-feature/ML-classifier framing is heavier than a single
  interpretable score and would need to be distilled down to be compatible with V5's
  "one number per concept" design philosophy; exact formula details for the base
  per-axis index are only summarized in secondary sources here, not read from the
  primary chapter text (Springer chapter body was not directly fetched).
- **Difference from V5**: V5 aligns via PCA then folds the *aligned* mask along
  horizontal/vertical image axes and measures 1 − mean IoU of the two folds combined
  into a single number. This method reflects across each principal axis independently
  and reports area-of-non-overlap (not IoU) per axis, which changes the scale/behavior
  for elongated lesions and would let A be decomposed into "asymmetry along major
  axis" vs. "asymmetry along minor axis" rather than a single blended value.
- **Implementation difficulty**: Low for the core per-axis reflection-overlap metric
  (same PCA infrastructure already exists in V5); high if attempting to reproduce the
  full 310-feature/ML pipeline.

### A2. Color-block CIE L*a*b* asymmetry (Seidenari, Pellacani, Grana)
- **Method name**: Computer description of asymmetry based on colour distribution
- **Category**: A
- **Citation**: Seidenari S, Pellacani G, Grana C. "Asymmetry in dermoscopic
  melanocytic lesion images: a computer description based on colour distribution."
  Acta Dermato-Venereologica. 2006;86(2):123–128. PMID 16648914.
  Source: https://pubmed.ncbi.nlm.nih.gov/16648914/ and https://www.medicaljournals.se/acta/content/html/10.1080/00015555-0043
- **Formula/algorithm**: Rather than folding a binary mask, the lesion image is
  divided into color blocks and asymmetry is scored by comparing CIE L*a*b* color
  values across corresponding blocks on either side of candidate symmetry axes
  (shape symmetry is treated as a separate, distinct measurement from color
  symmetry). Clinician-vs-computer agreement was reported: clinicians called 12.8%
  of benign nevi / 44.7% of atypical nevi / 64.2% of melanomas asymmetric, vs.
  6.3% / 33.3% / 82.2% for the computer method — i.e., the automated method was more
  conservative on benign/atypical lesions but more sensitive on melanoma.
  Exact block size / axis-search procedure was not confirmed beyond the abstract-level
  summary retrieved here.
- **Required input**: Lesion mask + LAB color image; candidate symmetry axes.
- **Advantages**: Captures asymmetry of *pigmentation*, which V5 currently does not
  measure at all (V5's `A` feature is purely geometric/shape-based; color asymmetry
  is an entirely separate signal from V5's existing color features, which measure
  overall variegation/entropy but not spatial (left-right) color asymmetry).
  Validated directly against dermatologist asymmetry judgments, which is a useful
  ground truth this pipeline's V5 A score has not been checked against.
- **Limitations**: Older method (2006); LAB block comparison requires defining block
  granularity and axis search, adding implementation complexity; published sensitivity/
  specificity for melanoma alone not extracted from this search.
- **Difference from V5**: V5's `A` is 100% shape/mask-based and ignores color
  entirely. This method is a genuinely different axis of "asymmetry" (pigment-pattern
  symmetry) that could become a *new* 18th feature rather than a replacement for
  the existing shape-based A.
- **Implementation difficulty**: Medium — needs a color-block partitioning scheme and
  an axis-matching procedure on top of existing LAB conversion code already in V5's
  color pipeline.

### A3. Pigmentation-elevation-model + Laplace–Beltrami distribution asymmetry
- **Method name**: Distribution quantification via pigmentation elevation model and
  global point signatures (GPS)
- **Category**: A (also touches C)
- **Citation**: Liu Z, Sun J, Smith L, et al. "Distribution quantification on
  dermoscopy images for computer-assisted diagnosis of cutaneous melanomas." Medical &
  Biological Engineering & Computing. 2012. PMID 22438064.
  Source: https://link.springer.com/article/10.1007/s11517-012-0895-7 and https://pubmed.ncbi.nlm.nih.gov/22438064/
  (Full author list beyond "Liu, Sun, Smith et al." was not confirmed from the
  search snippets — treat as incomplete.)
- **Formula/algorithm**: Builds a "pigmentation elevation model" from the dermoscopy
  image (an estimate of melanin/hemoglobin-related pigment intensity treated as a
  height field over the lesion), then computes Global Point Signatures derived from
  the Laplace–Beltrami operator on that surface to characterize the distribution of
  pigmentation, not just its binary footprint. Asymmetry and shape/pigmentation
  distribution are quantified jointly through this spectral shape-analysis approach.
  Tested with SVM + 10-fold CV on 351 images (88 melanoma / 263 nevi): 86.36%
  sensitivity, 82.13% specificity, 83.43% accuracy.
- **Required input**: Full-color dermoscopy image (for pigmentation estimation, not
  just a binary mask).
- **Advantages**: Moves asymmetry assessment from "shape of a binary mask" (V5's
  current approach) to "distribution of pigment intensity within the lesion,"
  which is closer to what a dermatologist actually looks at under the ABCD rule
  (structural, not just outline, asymmetry). Reasonably strong reported performance.
- **Limitations**: Substantially more complex mathematically (spectral graph/manifold
  methods) than V5's PCA+fold approach; the "pigmentation elevation model" step is
  itself a nontrivial calibration-sensitive estimation; implementation difficulty is
  high relative to the interpretability goals of this pipeline.
- **Difference from V5**: V5 measures only geometric mask asymmetry after PCA
  alignment; this method measures asymmetry of the internal pigment distribution using
  spectral shape descriptors — a fundamentally different and more information-rich
  input (full image vs. binary mask).
- **Implementation difficulty**: High.

*Note on original ABCD rule context:* the original Stolz/Nachbar dermatoscopy ABCD
rule (Nachbar F, Stolz W, et al., "The ABCD rule of dermatoscopy...," Journal of the
American Academy of Dermatology, 1994; PMID 8157780, https://pubmed.ncbi.nlm.nih.gov/8157780/)
scores asymmetry on an ordinal 0/1/2 scale (0/1/2 axes of asymmetry) as one term in a
weighted Total Dermoscopy Score (`TDS = 1.3·A + 0.1·B + 0.5·C + 0.5·D`, with a coarse
D scored differently from continuous diameter). This is included as historical context
only — it is a coarser, clinician-oriented scoring convention, not a candidate
replacement formula for V5's continuous A score.

---

## B — Border

### B1. Fractal dimension of the lesion boundary (foundational)
- **Method name**: Fractal shape analysis for melanoma border irregularity
- **Category**: B
- **Citation**: Claridge E, Hall PN, Keefe M, Allen JP. "Shape analysis for
  classification of malignant melanoma." Journal of Biomedical Engineering.
  1992;14(3):229–234. PMID 1588780.
  Source: https://pubmed.ncbi.nlm.nih.gov/1588780/ and https://iopscience.iop.org/article/10.1016/0141-5425(92)90057-R
- **Formula/algorithm**: Estimates a fractal dimension of the lesion contour (via
  boundary-length-vs-measuring-scale / box-counting-style methods) as an
  observer-independent measure of border roughness. Higher fractal dimension ≈ more
  irregular/complex boundary, on the reasoning that melanoma borders structurally
  resemble fractal curves (e.g., Koch-curve-like) more than smooth nevus borders.
  A later related paper (Piantanelli A, Maponi P, Scalise L, Serresi S, Cialabrini A,
  Basso A. "Fractal characterisation of boundary irregularity in skin pigmented
  lesions." Medical & Biological Engineering & Computing. 2005;43:436–442.
  https://link.springer.com/article/10.1007/BF02344723) extends this with a
  variogram-based fractal-dimension estimator and reports ~85% discrimination
  accuracy between nevi and melanoma using fractal dimension alone (box-counting
  method).
- **Required input**: Binary lesion mask / extracted contour.
- **Advantages**: A single scalar (fractal dimension) that is scale-invariant in
  principle and captures multi-scale roughness that a single circularity ratio
  cannot; well-established in the dermoscopy CAD literature since the early 1990s;
  computationally straightforward (box-counting or variogram estimation over an
  existing contour).
- **Limitations**: Fractal dimension estimates are sensitive to the range of scales
  used and to contour resampling/smoothing choices, which introduces its own set of
  hyperparameters and potential instability; captures "texture irregularity" (fine
  roughness) more than "structure irregularity" (large indentations/protrusions) —
  the literature explicitly distinguishes these two, and fractal dimension favors the
  former.
- **Difference from V5**: V5 currently has no multi-scale roughness measure. Its three
  border features (`B_circularity`, `solidity`, `turning_angle_std`) all operate at a
  single global scale (whole-contour compactness, whole-hull solidity) or a fixed
  100-point resampling (turning angle). A fractal-dimension feature would add
  genuinely new, scale-spanning information not captured by the current three.
- **Implementation difficulty**: Medium (box-counting on a mask is simple; getting a
  stable/well-conditioned estimate needs care in choosing the scale range).

### B2. Area-based Irregularity Index (indentation/protrusion decomposition)
- **Method name**: Irregularity Index (most-significant + overall variants)
- **Category**: B
- **Citation**: Lee TK, McLean DI, Atkins MS. "Irregularity index: A new border
  irregularity measure for cutaneous melanocytic lesions." Medical Image Analysis.
  2003;7(1):47–64. PMID 12467721.
  Source: https://pubmed.ncbi.nlm.nih.gov/12467721/ and https://www.sciencedirect.com/science/article/abs/pii/S1361841502000907
- **Formula/algorithm**: For each individual indentation and protrusion along the
  border (relative to a smoothed/reference version of the boundary, e.g. the convex
  hull or a low-frequency approximation), an area-based irregularity index is
  computed per local feature. From the set of per-feature indices, two summary
  measures are derived: the "most significant irregularity index" (largest single
  indentation/protrusion) and the "overall irregularity index" (aggregate across all
  of them).
- **Required input**: Lesion contour + a reference smooth boundary (e.g. convex hull).
- **Advantages**: Explicitly targets "structure irregularity" (global indentations
  and protrusions) as distinct from fine texture roughness — complementary to B1's
  fractal approach. Produces two interpretable numbers (worst single deviation, and
  overall deviation) rather than one blended score, which fits this pipeline's
  preference for interpretable, named features.
- **Limitations**: Requires defining/extracting individual indentation/protrusion
  regions algorithmically, which is more implementation work than a single global
  ratio; original validation used non-dermoscopic ("clinical") images in at least some
  of the group's related papers (need to confirm dermoscopy vs. clinical-photo
  validation set from the primary text, which was not fetched in full here).
- **Difference from V5**: V5's `solidity` (contour_area / convex_hull_area) captures
  *aggregate* deviation from convexity as a single ratio but cannot say whether that
  deviation comes from one large notch or many small ones. This method decomposes
  that same convex-hull-relative-deviation concept into per-feature indices and two
  summary statistics — a genuine refinement of what `solidity` already approximates,
  not an unrelated new concept.
- **Implementation difficulty**: Medium (needs boundary-vs-hull deviation
  segmentation logic beyond a single area ratio).

### B3. Centroid Distance Diagram / Fourier-based border descriptors
- **Method name**: Centroid Distance Diagram (CDD) and derived descriptors
- **Category**: B
- **Citation**: Zhou Y, et al. "A new method describing border irregularity of
  pigmented lesions." Skin Research and Technology. 2010;16(1):66–76. PMID 20384885.
  Source: https://pubmed.ncbi.nlm.nih.gov/20384885/ and https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1600-0846.2009.00403.x
  (Full first-author given name / co-author list not independently confirmed beyond
  "Zhou Y, et al.")
- **Formula/algorithm**: Computes a centroid-distance curve — the distance from the
  lesion centroid to the border, as a function of angular orientation around the
  centroid (conceptually similar to converting the contour to polar coordinates
  centered at the centroid). From this curve the paper derives: a
  "non-centroid-convexity index," a max–min distance indicator, the standard
  deviation of the centroid-distance curve, and the maximum magnitude of non-zero
  frequency components after a discrete Fourier transform of the curve. Reported
  performance on 60 melanoma / 107 benign lesions: sensitivity 74.2%, specificity
  72.6%, AUC 0.788.
- **Required input**: Lesion contour + centroid.
- **Advantages**: The Fourier-magnitude descriptor in particular is a natural,
  well-understood way to summarize periodic/aperiodic border undulation frequency
  content — arguably a more principled way to capture "how irregular is the boundary"
  than a single scalar std-of-turning-angle, since it separates irregularity by
  spatial frequency rather than collapsing everything into one number.
- **Limitations**: Centroid-distance curves are sensitive to concave/non-star-shaped
  contours (a ray from the centroid can cross the border more than once for highly
  irregular lesions, which the simple radial-distance formulation does not handle
  cleanly); moderate reported AUC (0.788) suggests it is not dramatically stronger
  than simpler measures on its own.
- **Difference from V5**: V5's `turning_angle_std` already resamples the contour to
  100 equal-arc-length points and takes the std of the discrete turning angle — which
  is conceptually adjacent (both are "how much does local border direction vary")
  but V5 never moves to the frequency domain. A DFT-magnitude-based descriptor of
  either the turning-angle sequence or the centroid-distance sequence would capture
  *where* (at what spatial frequency) irregularity concentrates, which the current std
  scalar cannot distinguish (a single big lobe vs. many small serrations can produce
  similar std but very different melanoma-relevant morphology).
- **Implementation difficulty**: Low–medium (contour is already resampled to 100
  points in V5; adding an FFT of that sequence and extracting magnitude-based
  features is a small addition on top of existing infrastructure).

*Additional B-adjacent finding (border transition sharpness, closely related to V5's
`skin_contrast` which sits in the Color group but is really a border-transition
metric):* Kaya S, Bayraktar M, Kockara S, Mete M, Halic T, Field HE, Wong HK. "Abrupt
skin lesion border cutoff measurement for malignancy detection in dermoscopy images."
BMC Bioinformatics. 2016. DOI 10.1186/s12859-016-1221-4.
https://pmc.ncbi.nlm.nih.gov/articles/PMC5073935/ — quantifies abruptness of pigment
pattern cutoff at the lesion periphery using texture-homogeneity statistics computed
at multiple scales outward from a density-based border, rather than a single LAB
color-distance ring. This is a texture-based generalization of what V5's
`skin_contrast` (a single LAB-distance number between lesion mean and one thin outer
ring) approximates with a much simpler formula; flagged here rather than given a full
entry since it sits at the B/C boundary.

---

## C — Color

### C1. Automatic color-clustering for variegation detection (Umbaugh et al.)
- **Method name**: Automatic color segmentation for variegated-coloring detection
- **Category**: C
- **Citation**: Umbaugh SE, Moss RH, Stoecker WV. "Automatic color segmentation of
  images with application to detection of variegated coloring in skin tumors." IEEE
  Engineering in Medicine and Biology Magazine. 1989. DOI 10.1109/51.45955.
  Source: https://ieeexplore.ieee.org/document/45955/ and https://pubmed.ncbi.nlm.nih.gov/18244093/
  (A closely related follow-up by an overlapping author group, "Applying artificial
  intelligence to the identification of variegated coloring in skin tumors," IEEE
  Eng Med Biol Mag, is also relevant but was not separately detailed here.)
- **Formula/algorithm**: Uses unsupervised color-clustering (splitting the lesion's
  color histogram into a data-driven set of dominant color clusters) rather than
  fixed, hand-specified color categories, then counts/characterizes the resulting
  clusters as the variegation measure. Applied to 200 digitized skin-tumor images as
  the front end of an early medical expert system.
- **Required input**: Full-color lesion image + mask.
- **Advantages**: Data-driven cluster discovery avoids the need to hand-tune specific
  color thresholds (e.g., "what RGB/LAB range counts as red vs. blue-gray vs. dark"),
  which is exactly the kind of hand-tuning V5's `red_fraction`/`bluegray_fraction`/
  `dark_fraction` currently rely on. Cluster count is a natural, clinically-aligned
  proxy for "number of distinct colors," which is literally the ABCD-rule color
  criterion.
- **Limitations**: Very old (1989) baseline method; clustering approaches need a
  principled way to choose the number of clusters and are sensitive to
  illumination/color-calibration differences across images/devices (a known general
  weakness of unsupervised color work in dermoscopy).
- **Difference from V5**: V5's `red_fraction`/`bluegray_fraction`/`dark_fraction` are
  explicitly described as "hand-tuned rules relative to sampled skin color" — fixed
  thresholds decided a priori. This method instead discovers the dominant colors
  present in each individual lesion via clustering, which would replace fixed
  thresholds with a per-image adaptive color-cluster count/description — directly
  addressing the "hand-tuned" limitation named in this pipeline's own feature
  description.
- **Implementation difficulty**: Low–medium (k-means or similar clustering in LAB
  space over in-mask pixels, using existing LAB conversion already in V5).

### C2. Shannon entropy for color variegation (direct comparator to V5's `color_entropy`)
- **Method name**: Global Shannon entropy of color histogram for variegation
- **Category**: C
- **Citation**: Martínez-Ortega JI, Ramirez Ibarra FG, Fernández-Reyna I, Macias
  Quiroga AN. "Quantifying Color Variegation in Melanoma Lesions Using Shannon
  Entropy." Cureus. 2025;17(10):e94779. DOI 10.7759/cureus.94779.
  Source: https://pubmed.ncbi.nlm.nih.gov/41250739/ and https://www.cureus.com/articles/423835-quantifying-color-variegation-in-melanoma-lesions-using-shannon-entropy
- **Formula/algorithm**: Computes Shannon entropy over a color histogram of the
  lesion to give a reproducible, quantitative measure of color variegation, framed
  explicitly as a proof-of-concept for teledermatology / low-resource settings.
  This is essentially the same core mathematical construct as V5's `color_entropy`
  (Shannon entropy of a color histogram) — this is close to independent confirmation
  that entropy-of-color-histogram is a reasonable, published approach for this
  purpose, not a critique of it.
- **Required input**: Color lesion image + mask; a color histogram (channel/binning
  scheme not confirmed in detail from the abstract-level summary retrieved).
- **Advantages**: Validates the general approach V5 already uses; simple, fast,
  interpretable, low implementation cost since it's already implemented.
- **Limitations**: The paper itself is described as a "proof-of-concept," and per C3
  below, a closely related, more recent paper from an overlapping author explicitly
  argues that *global* Shannon entropy is insufficient in some conditions (see C3).
- **Difference from V5**: Very close in concept to V5's existing `color_entropy`
  (16×16 2D histogram over LAB a,b plane); main open question is exact color space
  and binning used in this paper, which was not confirmed here — worth a closer read
  before drawing implementation conclusions, but no immediate contradiction with V5's
  approach.
- **Implementation difficulty**: Low (already implemented equivalent in V5; any
  change would be tuning histogram parameters).

### C3. Channel-specific entropy vs. global Shannon entropy (direct critique relevant to V5)
- **Method name**: Channel-specific entropy / inter-channel metrics for polychromia
- **Category**: C
- **Citation**: Martínez-Ortega JI, Martinez-Jaramillo B. "Beyond Global Shannon
  Entropy: A Channel-Specific Approach to Quantify Polychromia in Melanoma." Cureus.
  2026 (published online, DOI 10.7759/cureus.102257, PubMed record 41742992).
  Source: https://pubmed.ncbi.nlm.nih.gov/41742992/ and https://pmc.ncbi.nlm.nih.gov/articles/PMC12931566/
- **Formula/algorithm**: Directly evaluated whether a single, global Shannon entropy
  value (entropy computed over one combined color histogram) adequately captures
  polychromia (multi-color variegation), versus computing entropy per individual
  color channel and then combining/comparing channel-specific entropies plus
  inter-channel metrics. The stated finding: global Shannon entropy does *not*
  meaningfully quantify polychromia under real-world (e.g., smartphone) imaging
  conditions, whereas channel-specific entropy and inter-channel metrics reliably
  discriminate chromatically heterogeneous lesions from uniform ones.
- **Required input**: Multi-channel color lesion image + mask.
- **Advantages**: This is the most directly actionable finding in the whole review
  for the Color category — it is a peer-reviewed (Cureus) paper making a specific,
  falsifiable claim that a single combined-histogram entropy value (structurally what
  V5's `color_entropy` is — one entropy number from one 2D a,b histogram) may be
  under-sensitive, and proposing a concrete, low-complexity fix (compute entropy
  per-channel instead of, or in addition to, jointly).
- **Limitations**: Very recently published (2026); single small-scale/proof-of-concept
  study per the framing found in search results; "real-world smartphone imaging"
  framing suggests the failure mode may be more pronounced for non-dermoscopic
  images than for the calibrated dermoscopic images V5 is trained/frozen on, so the
  critique may be less applicable to this pipeline's ISIC-dermoscopic-only domain —
  worth validating on ISIC data specifically before assuming it applies.
- **Difference from V5**: V5 computes exactly one global entropy value from a single
  joint 2D (a,b) histogram. This paper's core suggestion — computing entropy
  separately per channel (e.g., L, a, b individually, or combined with inter-channel
  divergence/correlation metrics) rather than one joint histogram — is a small,
  concrete, well-motivated variant to test against the existing `color_entropy`,
  `lab_a_std`, and `lab_b_std` features already present in V5 (which already separate
  a/b std but not a/b entropy).
- **Implementation difficulty**: Low (add per-channel 1D histograms/entropy
  alongside the existing joint 2D histogram; reuses existing LAB pixel arrays).

### C4. Melanin/hemoglobin pigmentation elevation model
- See entry **A3** above (Liu et al., 2012) — the same paper's pigmentation-elevation
  model is also directly relevant to Color, since it separates estimated
  melanin/hemoglobin contribution rather than working in raw LAB space as V5 does.
  Not repeated in full here to avoid duplication; cross-referenced for completeness.

---

## D — Diameter

### D1. Feret's diameter as an alternative to minimum-enclosing-circle diameter
- **Method name**: Feret's diameter for lesion size
- **Category**: D
- **Citation**: Ali A-R, Li J, O'Shea SJ. "Towards the automatic detection of skin
  lesion shape asymmetry, color variegation and diameter in dermoscopic images."
  PLOS ONE. 2020. Source: https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0234352
  and https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7297317/ (University of Stirling).
- **Formula/algorithm**: Uses Feret's diameter (the maximum caliper distance between
  any two points on the lesion contour, i.e. the longest chord) as the diameter
  measure, rather than fitting a minimum enclosing circle. The same paper also
  proposes a decision-tree-based asymmetry measure (80% accuracy) and a
  derived-suspicious-color-count approach for variegation, making it a useful
  cross-cutting reference for A, C, and D simultaneously.
- **Required input**: Lesion contour.
- **Advantages**: Feret's diameter is simpler to compute than a minimum enclosing
  circle and is a standard, well-understood morphometric measure; for elongated
  (non-circular) lesions it directly reports the true longest axis, whereas a minimum
  enclosing circle's diameter is always ≥ the longest chord and can overstate size
  for very elongated shapes.
- **Limitations**: Like V5's `D_px`, this paper's diameter appears to be reported in
  raw/pixel terms without confirmation of physical mm calibration in the summary
  retrieved here — i.e., it does not, by itself, solve V5's stated "never calibrated"
  limitation; it only changes which geometric quantity is measured, not whether it's
  in physical units.
- **Difference from V5**: V5 uses the diameter of the minimum enclosing circle around
  the contour. Feret's diameter (max caliper / longest chord) is a related but
  distinct geometric quantity — for a convex, roughly circular lesion the two are
  close, but for elongated or highly irregular lesions they diverge, with Feret's
  diameter generally being smaller than the minimum-enclosing-circle diameter and
  arguably closer to what a clinician means by "the long axis measured with a ruler."
- **Implementation difficulty**: Low (`cv2`/`skimage` support max caliper / Feret
  diameter computation directly from a contour; roughly a drop-in swap for the
  existing min-enclosing-circle call).

### D2. Ruler-detection-based physical-scale calibration (directly targets V5's stated D limitation)
- **Method name**: GraphDerm scale regression (ruler segmentation + pixels-per-mm CNN regression)
- **Category**: D
- **Citation**: Yousefzadeh M, et al. "GraphDerm: Fusing Imaging, Physical Scale, and
  Metadata in a Population-Graph Classifier for Dermoscopic Lesions." arXiv preprint,
  arXiv:2509.12277, September 2025 (preprint — not yet confirmed as peer-reviewed
  publication). Source: https://arxiv.org/abs/2509.12277 and
  https://arxiv.org/pdf/2509.12277 (full author list beyond "Yousefzadeh M, et al."
  not independently confirmed from search results).
- **Formula/algorithm**: Trains a U-Net (SE-ResNet-18 backbone) to segment any
  physical ruler markings present in dermoscopy images, plus the lesion itself
  (reported Dice ≈0.904–0.908 for ruler/lesion segmentation respectively), then
  regresses pixels-per-millimeter from the ruler-mask's two-point correlation
  function using a lightweight 1D CNN (reported MAE ≈1.5 px, RMSE ≈6.6). This
  recovered mm/px scale is then fused with imaging + metadata features in a
  population-graph classifier. Directly frames the underlying problem this pipeline
  faces: "physical scale cannot be inferred from raw pixels alone and is often
  absent due to heterogeneous imaging protocols and lack of calibration markers" —
  i.e., an explicit peer/preprint acknowledgment of the exact limitation this
  pipeline's `D_px` docstring already states.
- **Required input**: Dermoscopy image that contains (or can be synthetically
  augmented with) a visible ruler/scale marking; requires training data with
  ruler annotations, which the paper synthesizes onto ISIC images since ISIC does
  not natively provide this.
- **Advantages**: This is the most directly relevant finding for D in the whole
  review — it is a concrete, recent (2025), technically detailed method for solving
  exactly the calibration gap V5 explicitly flags as unsolved ("D_px... NEVER
  calibrated to physical mm — no reliable calibration source was found for this
  dataset"). If ISIC images occasionally contain rulers (some ISIC source
  collections do include ruler/scale markers in a subset of images), this offers a
  concrete path to a calibrated diameter feature without new data collection.
- **Limitations**: This is an arXiv preprint (September 2025), not yet confirmed
  peer-reviewed — should be flagged as lower-confidence evidence than the
  journal-published items in this review. Requires ISIC images to actually contain
  ruler markings to be useful (many ISIC dermoscopic images do not include a visible
  ruler), which would limit applicability to only a subset of this pipeline's
  training/inference images, creating a missing-calibration-marker problem that the
  method itself doesn't fully solve for ruler-absent images (the paper's own
  approach relies partly on synthetic ruler augmentation for training, implying
  real rulers aren't universally present in ISIC either).
- **Difference from V5**: V5 makes no attempt at physical-mm calibration and
  explicitly documents this as a known, accepted limitation, using
  `D_px_normalized = D_px / sqrt(H·W)` as a resolution-invariant proxy instead. This
  method is a direct, purpose-built attempt to solve the calibration problem itself
  rather than work around it with a normalized-pixel proxy.
- **Implementation difficulty**: High (requires training a segmentation model for
  ruler detection, synthetic ruler-augmented training data, and a regression model
  for scale estimation — a substantial new sub-pipeline, not a formula change).

### D3. DICOM dermoscopy metadata standard for physical pixel spacing (calibration, standards-based alternative)
- **Method name**: DICOM Supplement 221 — Dermoscopy (pixel spacing metadata standard)
- **Category**: D
- **Citation**: DICOM Standards Committee, "Supplement 221: Dermoscopy," DICOM
  standard supplement (not a peer-reviewed research paper — included as relevant
  standards context, flagged accordingly). Source:
  https://www.dicomstandard.org/News/current/docs/sups/sup221.pdf
- **Formula/algorithm**: Not a measurement formula but a data standard: defines how
  dermoscopy image files should carry calibrated physical pixel-spacing metadata
  (and a diameter/scale reference) so that downstream diameter measurements can be
  converted from pixels to mm reliably and consistently across devices, rather than
  inferred post hoc from image content (e.g., a ruler in-frame).
- **Required input**: Acquisition-time device support for writing DICOM dermoscopy
  metadata; not applicable retroactively to images that were never captured/exported
  with this metadata (which almost certainly includes ISIC's existing archive).
  Reported here for completeness/context only, since the pipeline description
  states no reliable calibration source was found for this dataset.
- **Advantages**: The "correct" long-term fix for the calibration problem generally
  (standardized metadata beats post hoc pixel inference), and worth knowing about if
  any future, non-ISIC data source used by this project supports DICOM export.
- **Limitations**: Not applicable to the existing frozen ISIC-based training set;
  not a research finding but a standards document, included transparently as such
  rather than being disguised as a peer-reviewed method.
- **Difference from V5**: Orthogonal to V5 entirely — a data-acquisition-time
  solution rather than a post hoc image-analysis method; listed because it directly
  answers "is there a standard, established calibration source" that V5's own
  documentation says it could not find, at least for any *future* non-ISIC dermoscopy
  source this project might use.
- **Implementation difficulty**: N/A for the current frozen ISIC-based pipeline
  (not implementable retroactively); low if a future data source is DICOM-dermoscopy
  compliant.

---

## Honest gap assessment

- **Asymmetry (A)**: 3 solid findings, all genuinely comparable/citable. Good
  coverage, though note that A1's exact per-axis formula was only confirmed at
  summary level (chapter body not directly read) and A3's author list is incomplete —
  both flagged inline above.
- **Border (B)**: 3 full findings plus 1 closely related B/C-boundary finding
  (abrupt border-cutoff paper). Good, well-established literature base going back to
  1992; this is the category with the deepest, most mature body of citable work.
- **Color (C)**: 3 solid findings (C1, C2, C3) plus a cross-reference to A3. C3 in
  particular is a strong, directly-actionable, recent finding specifically about the
  weakness of global Shannon entropy — the closest thing in this whole review to a
  direct critique of one of V5's existing formulas (`color_entropy`).
- **Diameter (D)**: This is the category with the thinnest *peer-reviewed* coverage.
  D1 (Feret's diameter) is solid and peer-reviewed but doesn't solve calibration.
  D2 (GraphDerm) is the most relevant finding for D's core stated problem
  (uncalibrated pixels) but is an arXiv preprint, not yet confirmed peer-reviewed —
  flagged accordingly rather than presented as established literature. D3 is a
  standards document, not a research finding. **Be honest that classical,
  peer-reviewed literature specifically on solving diameter calibration from
  dermoscopic images alone (without a physical ruler or DICOM metadata) is genuinely
  sparse** — most published work either assumes a known reference object/ruler is in
  frame, or (like this pipeline) works entirely in normalized/relative pixel units
  and does not attempt physical-mm conversion at all. This is a real, reportable gap
  in the literature, not a search failure.
