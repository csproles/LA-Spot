# Melanoma risk model integration -- NOTES

Branch: `model-integration` (copy of `user`). Not pushed -- final checks pending.

## What this branch does

Wires the Kaggle-trained CNN+CatBoost melanoma risk model into the Flask API as a
new, additive `POST /predict` endpoint, alongside (not replacing) the existing
V5Detector `/api/image/process` pipeline. Both pipelines run independently and
share nothing except the Blazor upload flow and the `/api/image/explain/{id}`
LLM-explanation endpoint (reused, not duplicated).

## Setup performed

- Unzipped `app_models.zip`; its `slim/` contents moved to `models/` at the repo
  root. Added `/models/` to `.gitignore` and to `.dockerignore` (never committed,
  excluded from the Docker build context, delivered to the container as a
  read-only bind mount instead -- see docker-compose.yml).
- Unzipped `melanoma_pipeline-...zip` into `melanoma_pipeline/` at the repo root.
  **`train_catboost_meta.py` was not present in that zip** -- it was only given
  to me as pasted text in the task description. I did not add it to the repo,
  since nothing imports it at runtime (see "How the APP_MODE feature set was
  confirmed" below) and I didn't want to commit an unverified transcription as
  if it were the file that actually produced these weights. If you have the
  real file, drop it into `melanoma_pipeline/` for reference -- it isn't a
  runtime dependency either way.
- Created `model-integration` branch from `user`.

## How the APP_MODE feature set was confirmed (important)

The task said to read `train_catboost_meta.py` (run with `APP_MODE=1` on
Kaggle) to understand the feature set, but that script (as pasted into the
task) builds features from the **full ISIC 2024 tabular metadata**
(`tbp_lv_*` columns, patient-normalized "ugly duckling" features, etc.) --
columns this app has no way to produce for a phone photo. There's no
`APP_MODE` flag in that script as given, so I couldn't trace the exact code
path that trimmed it down for the app models.

Instead of guessing, I loaded the actual
`models/catboost_app_fold_{0..4}_feature_names.npy` files directly -- these
are the literal, authoritative column list/order each `.cbm` was fit on. All
5 folds have **identical** 148-column layouts:

```
cnn_pca_0 .. cnn_pca_127        (128 cols -- PCA of the 1024-dim ConvNeXt embedding)
A_value, B_circularity, C_value, D_px, confidence, lesion_fraction,
color_entropy, lab_a_std, lab_b_std, red_fraction, bluegray_fraction,
dark_fraction, skin_contrast, solidity, turning_angle_std, eccentricity,
D_px_normalized                  (17 cols -- exactly melanoma_pipeline/abcd/feature_config.py's V5_ALL_FEATURES, same order)
age_approx, sex, anatom_site_general   (3 cols)
```

This confirms APP_MODE trimmed the metadata down to exactly the three fields
the app collects (age, sex, body site) and nothing else -- no `tbp_lv_*`
columns, no patient-normalization, no `patient_lesion_count`. `risk_model.py`
builds its DataFrame from this `.npy` file directly (never hardcoded), so it
can't silently drift from what a given `.cbm` was actually trained on.

## Implementation

**`risk_model.py`** (new) -- loads all 5 folds (CNN checkpoint, PCA, CatBoost,
feature-name list) once at import, held by one `RiskModel` instance created at
Flask startup in `main.py` (`risk_model = RiskModel()`, next to the existing
`detector = V5Detector()`). Per request:

1. `melanoma_pipeline.features.extract_abcd_features(image_path, yolo_model)` --
   the training pipeline's own YOLO+ABCD code, reusing the **new**
   `models/yolo_melanoma_seg.pt` (not the older, unrelated YOLO checkpoint
   under `MelanomaDetection.Python/models/` that `v5_detector.py` uses -- two
   separate YOLO models, easy to confuse, kept namespaced by directory).
   Returns NaN for all 17 features if no lesion is found; never rejects the
   image, per the task's own instruction.
2. CNN embedding: `melanoma_pipeline.dataset.build_val_transforms()` (224px
   resize+center-crop+ImageNet normalize) into
   `MelanomaCNN(pretrained=False).forward_features()` (pre-head, 1024-dim),
   one forward pass per fold's own checkpoint (folds don't share weights).
3. Each fold's own PCA transforms that fold's own embedding -> 128 components.
4. A single-row DataFrame is built in the exact column order from that fold's
   `_feature_names.npy`; `predict_proba` gives that fold's score.
5. Final `risk_score` = mean of the 5 fold scores (0-1, **not** a calibrated
   probability -- see risk_model.RISK_THRESHOLDS' own TODO for the
   placeholder low/medium/high cut points at 0.33/0.66).

**`main.py`** -- new `POST /predict` (multipart `image`, `age`, `sex`,
`body_site`), rate-limited the same as `/api/image/process` (10/min/user).
Response: `{ risk_score, risk_level, abcd_features, yolo_found_lesion,
fold_scores, processingId }`. `processingId` (prefixed `pred_`, distinct from
V5's `proc_`) is a deliberate bit of reuse: a minimal, compatible result
(`abcde_scores`, `risk_score` rescaled to 0-100, `overall_visual_concern:
null`, `symptoms: []`) is stashed in the existing in-memory `_results_store`
under it, so the **existing, unmodified** `POST /api/image/explain/{id}`
endpoint (`llm_explainer.explain_findings`) works for this pipeline's results
too, with zero new LLM code. `_abcd_features_to_abcde_scores()` adapts the raw
17-value ABCD dict into the `{"asymmetry": {"score", "details"}, ...}` shape
that function expects, reusing `v5_detector._scaled_0_10` and V5's own
display thresholds (0.20/0.50/0.35) since these are the same ABCD features V5
computes, just fed to a different decision model.

- ponytail: the per-letter `"concern"` flag in that adapter is approximated as
  `raw_value > threshold`, not the fuller logic in
  `melanoma_pipeline/abcd/pipeline.py`'s scoring functions (which factor in
  more than one signal per letter). `extract_abcd_features` doesn't expose
  those richer concern booleans at all -- only the raw values. Good enough for
  the explanation prompt's flagged/unflagged split; call
  `abcd.pipeline.score_instance` directly instead if per-letter concern
  accuracy ever matters for this model's own explanations.

**`validation.py`** -- added `clean_age` (0-120, `None` when blank),
`clean_sex`, `clean_body_site` (both fall back to the literal string
`"missing"` when blank/unrecognized, matching what the CatBoost models were
trained to treat as missing metadata).

**A real bug found and fixed during testing**: `extract_abcd_features` returns
a numpy array, so every ABCD value was a `numpy.float64` all the way through.
`llm_explainer.py` calls `json.dumps()` on its payload **directly** (not
Flask's numpy-aware `NumpyJSONProvider`), and arithmetic on a numpy scalar
(`value > threshold` in `_map_to_llm_schema`'s `"spread_high"` field)
produces a `numpy.bool_`, which crashes that `json.dumps()` with `TypeError:
Object of type bool is not JSON serializable`. Fixed at the source in
`risk_model.py` (`abcd_dict = {name: float(value) for ...}`) rather than
patching every consumer, since native Python floats were what `llm_explainer`
was always written to expect from `V5Detector`. Verified fixed against a live
server (see "Test run" below): the explain call now fails only on the OpenAI
key, not on JSON serialization.

**Dockerfile** -- added `COPY melanoma_pipeline melanoma_pipeline` (mirrors the
existing `Code`/`revised_abcd`/`pipeline_v5` pattern). `models/` is *not*
COPY'd (see next point).

**docker-compose.yml** -- added `./models:/app/models:ro` to `python-api`'s
volumes. `risk_model.py` resolves `MODELS_DIR` as
`Path(__file__).resolve().parents[2] / "models"`, i.e. repo-root `models/` --
this is `/app/models` inside the container (matching the mount target) since
the Dockerfile's `WORKDIR /app` mirrors the repo root, exactly like
`v5_detector.py`'s own `REPO_ROOT` computation already does for
`pipeline_v5`/`revised_abcd`.

**requirements.txt** -- added `timm`, `catboost`, `albumentations`, `pandas`
(scikit-learn, ultralytics, torch, opencv, numpy were already pinned). Pinned
to whatever actually resolved and was verified working in the dev venv:
`timm==1.0.30`, `catboost==1.2.10`, `albumentations==2.0.8`, `pandas==2.2.3`
(pandas 3.0.6 also resolved, but a local Windows Application Control policy
blocked loading its native DLL -- 2.2.3 avoided that and is also the safer
choice generally, since pandas 3.x has breaking changes). These are **not
verified to match the original Kaggle training environment** (unknown) --
only the val-time transforms (`Resize`/`CenterCrop`/`Normalize`/`ToTensorV2`)
are used at inference, which are stable across albumentations 1.x/2.x, so this
is a reasonable bet, not a certainty. TODO if scores look off in practice:
pin exact versions from the Kaggle notebook's environment.

Also noted, not fixed: `albumentations` pulls in `opencv-python-headless` as a
transitive dependency, alongside the already-pinned `opencv-python`. Both
provide the `cv2` module; this worked fine in testing but is a latent
dependency smell worth cleaning up later (e.g. via `--no-deps` or an explicit
exclude) if it ever causes an import conflict in a fresh environment.

Also noted: loading `pca_app_fold_*.pkl` prints an sklearn
`InconsistentVersionWarning` (pickled with scikit-learn 1.6.1, this venv has
1.9.1). It loaded and produced sane-looking output in testing, but pickle
compatibility across sklearn versions isn't guaranteed -- if PCA output ever
looks wrong, pin `scikit-learn==1.6.1` in requirements.txt to match.

## Frontend (Blazor)

- `CheckDetailsStep.razor`: added Age (number input), Sex (select), Body site
  (select) fields -- all optional ("Prefer not to say"), in a new card
  alongside the existing symptoms/notes card.
- `Upload.razor`: holds `Age`/`Sex`/`BodySite`/`RiskModelResult` state. Moving
  from the Details step to Results now also calls the new
  `ImageProcessingService.PredictRiskAsync(...)` (in `ContinueToResults()`).
  If that call fails, it's swallowed and `RiskModelResult` stays null -- V5's
  own results (the page's main flow) are never blocked by this second,
  experimental model.
- `RiskModelPanel.razor` (new, in `Components/Analysis/`): shows the risk
  model's score/level in a card styled like the existing `RiskBandPanel`
  (reuses its `risk-band`/`risk-band-{low,moderate,high}` CSS classes), framed
  explicitly as "an experimental screening score ... not a diagnosis ... not a
  substitute for the visual-concern result above". Shows "Lesion not clearly
  detected in this photo -- this result is less reliable" when
  `yolo_found_lesion` is false. Renders nothing when `Result` is null.
- `ImageProcessingService.cs` / `ImageProcessingModels.cs`: added
  `PredictRiskAsync` and `PredictResponse`, following the exact conventions of
  the existing `ProcessImageAsync`/`ProcessImageResponse` pair.

**Skipped, not built** (time-boxed; flag if you want these):
- No loading spinner/disabled-state on the Details step's "See results" button
  while `PredictRiskAsync` is in flight (it's a single click, ~5-8s on CPU;
  a double-click could fire two `/predict` calls). `AnalyzeImage` already has
  this pattern (`IsAnalyzing`) -- copy it if this becomes annoying in practice.
- No dedicated "AI explanation" button wired to the new `processingId` on the
  results page -- `RiskModelPanel` only shows the score. The plumbing for an
  explanation (`POST /api/image/explain/{processingId}`) works end-to-end
  (verified below), it's just not surfaced with its own UI button yet the way
  V5's results are (`BreakdownDisclosure.razor`).

## Risk levels

Placeholder-only, per the task: `risk_model.RISK_THRESHOLDS = {"low": 0.33,
"medium": 0.66}` over the mean-of-5-folds score. **TODO** in the code itself
(`risk_model.py`): replace with real cut points once out-of-fold predictions
are available.

## Test run

Live server (`python main.py`, `SKINCHECK_ALLOW_NO_KEY=1`), sample image
`Images/Benign/ISIC_0000005.jpg`, `scripts/test_predict.py`:

```
POST http://localhost:5002/predict  image=...\Images\Benign\ISIC_0000005.jpg
HTTP 200 in 7.94s

{
  "abcd_features": {
    "A_value": 0.204, "B_circularity": 0.409, "C_value": 0.384, "D_px": 526.8,
    "D_px_normalized": 0.595, "bluegray_fraction": 0.0,
    "color_entropy": 1.547, "confidence": 0.8816, "dark_fraction": 0.12,
    "eccentricity": 0.598, "lab_a_std": 4.477, "lab_b_std": 5.334,
    "lesion_fraction": 0.1878, "red_fraction": 0.0642,
    "skin_contrast": 44.03, "solidity": 0.8724, "turning_angle_std": 0.4249
  },
  "fold_scores": [0.8044, 0.1070, 0.8360, 0.0815, 0.7569],
  "processingId": "pred_d83bbc9644e7",
  "risk_level": "medium",
  "risk_score": 0.5172,
  "yolo_found_lesion": true
}
```

Latency: ~8s end-to-end on CPU for a cold-ish first request (model load
itself, done once at startup, took ~7.7s separately in an isolated run;
steady-state per-request time is dominated by 5 sequential ConvNeXt-Base
forward passes + one YOLO pass, all CPU). Logged per-request via
`risk_model`'s own `logger.info(...)` call (elapsed ms, risk_score,
yolo_found_lesion).

**Observation, not necessarily a bug**: fold scores alternate sharply (folds
0/2/4 ~0.75-0.84, folds 1/3 ~0.08-0.11) on this one sample. Indexing was
double-checked (`cnn_models[fold]`, `pcas[fold]`, `catboosts[fold]`,
`feature_names[fold]` all keyed consistently by the same loop variable) --
this isn't a fold-mismatch bug on the integration side. It may reflect genuine
disagreement between folds trained on very different `GroupKFold` splits of a
small/imbalanced dataset (ISIC 2024 has ~400 malignant cases out of ~400k).
Worth a second look with more sample images before trusting the averaged
score in production.

Also verified: `POST /api/image/explain/{processingId}` on a `pred_...` id
reaches the OpenAI call (fails only on the test API key used here) --
confirms the reuse of the existing LLM-explanation flow works structurally
end to end.

**Regression checks**: full existing suites pass unmodified --
`pytest` (`MelanomaDetection.Python`): 221 passed. `dotnet test`
(`MelanomaDetection.Web.Tests`): 41 passed. `dotnet build`
(`MelanomaDetection.Web`): 0 warnings, 0 errors.

**Pre-existing, unrelated issue found**: `pytest` fails to collect
`tests/test_api_hardening.py` (`ModuleNotFoundError: No module named
'tests.test_validation'`) when run directly against this venv -- reproduced
on the unmodified `user` branch too (via `git stash`), so it predates this
work and isn't something I introduced. Worked around by running with
`--ignore=tests/test_api_hardening.py` for the regression check above; left
otherwise untouched since it's out of scope.

## Not done / explicitly skipped

- Did not attempt to verify or reproduce the original Kaggle training run
  (no training data, no GPU requirement implied by the task -- inference only).
- Did not calibrate risk thresholds (explicitly deferred by the task).
- Did not add a loading state for the Details-step "See results" button (see
  Frontend section).
- Did not wire an explicit "get AI explanation" button for the new pipeline's
  results in the UI (endpoint works, just not surfaced with its own button).
- Did not commit `train_catboost_meta.py` itself (see "Setup performed").
