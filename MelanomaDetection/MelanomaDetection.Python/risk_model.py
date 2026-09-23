"""Loads the Kaggle-trained melanoma risk model and runs inference for the
/predict endpoint.

Model: 5-fold ConvNeXt-Base (timm) CNN -> per-fold PCA on the 1024-dim
embedding -> per-fold CatBoost on [128 PCA components + 17 ABCD features +
age_approx + sex + anatom_site_general]. Mirrors melanoma_pipeline/
train_catboost_meta.py as trained on Kaggle with APP_MODE=1: the full ISIC
tabular metadata was trimmed down to the three fields this app actually
collects at inference time (age, sex, body site) -- see NOTES.md for the
full trace of how that was confirmed. The authoritative source for each
fold's exact column order is catboost_app_fold_{k}_feature_names.npy
(loaded directly, never hardcoded), since that is what each .cbm was
literally fit on.

Preprocessing must match training exactly:
  - ABCD features: melanoma_pipeline.features.extract_abcd_features, the
    same YOLO-segmentation + frozen V5 formula code the training pipeline
    used (NOT this app's other, older YOLO checkpoint under
    MelanomaDetection.Python/models/ -- that one belongs to v5_detector.py's
    unrelated classical pipeline).
  - CNN embedding: melanoma_pipeline.dataset.build_val_transforms (224px
    resize+center-crop+ImageNet normalize) into MelanomaCNN.forward_features
    (pre-head, 1024-dim pooled output), exactly as extract_embeddings.py did.

All five folds' models are loaded once (RiskModel() at Flask startup in
main.py), never per-request.
"""
from __future__ import annotations

import logging
import os
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = Path(os.environ.get("MELANOMA_MODELS_DIR", str(REPO_ROOT / "models")))
PIPELINE_DIR = REPO_ROOT / "melanoma_pipeline"
if str(PIPELINE_DIR) not in sys.path:
    # melanoma_pipeline's own modules use bare imports (`import config`,
    # `from abcd.feature_config import ...`), i.e. they assume the package
    # directory itself is on sys.path rather than being imported as
    # `melanoma_pipeline.xxx` -- same convention v5_detector.py follows for
    # pipeline_v5/revised_abcd.
    sys.path.insert(0, str(PIPELINE_DIR))

from features import extract_abcd_features, ABCD_FEATURE_NAMES  # noqa: E402
from dataset import build_val_transforms  # noqa: E402
from train_cnn import MelanomaCNN  # noqa: E402
from utils import load_checkpoint  # noqa: E402

logger = logging.getLogger(__name__)

N_FOLDS = 5
VALID_SEX = {"male", "female"}
VALID_BODY_SITES = {"head/neck", "upper extremity", "lower extremity", "anterior torso", "posterior torso"}

# Placeholder low/medium/high bands over the mean fold score (0-1 -- CatBoost's
# raw predict_proba output, NOT a calibrated probability of malignancy).
# TODO(real thresholds): replace with cut points derived from out-of-fold
# predictions once available; these three numbers are only a starting guess.
RISK_THRESHOLDS = {"low": 0.33, "medium": 0.66}


def risk_level(score: float) -> str:
    if score < RISK_THRESHOLDS["low"]:
        return "low"
    if score < RISK_THRESHOLDS["medium"]:
        return "medium"
    return "high"


def _nan_to_none(value: float):
    return None if value != value else value  # value != value is the NaN check


class RiskModel:
    """Owns all 15 loaded model artifacts (5 folds x {CNN, PCA, CatBoost})
    plus the shared YOLO segmentation model. One instance lives for the life
    of the process (see main.py)."""

    def __init__(self, models_dir: Path = MODELS_DIR, device: str = "cpu"):
        self.device = torch.device(device)
        self.val_transforms = build_val_transforms()

        self.yolo = YOLO(str(models_dir / "yolo_melanoma_seg.pt"))

        self.cnn_models = []
        self.pcas = []
        self.catboosts = []
        self.feature_names = []

        for fold in range(N_FOLDS):
            model = MelanomaCNN(pretrained=False).to(self.device)
            load_checkpoint(models_dir / f"fold_{fold}_best.pt", model, optimizer=None, map_location=self.device)
            model.eval()
            self.cnn_models.append(model)

            with open(models_dir / f"pca_app_fold_{fold}.pkl", "rb") as f:
                self.pcas.append(pickle.load(f))

            catboost_model = CatBoostClassifier()
            catboost_model.load_model(str(models_dir / f"catboost_app_fold_{fold}.cbm"))
            self.catboosts.append(catboost_model)

            names = np.load(models_dir / f"catboost_app_fold_{fold}_feature_names.npy", allow_pickle=True)
            self.feature_names.append([str(n) for n in names])

        logger.info("RiskModel loaded: %d folds from %s", N_FOLDS, models_dir)

    @torch.no_grad()
    def _cnn_embeddings(self, image_path: str) -> list[np.ndarray]:
        """One (1, 1024) embedding per fold's own CNN -- folds never share
        weights, so each fold's PCA must see that same fold's embedding."""
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.val_transforms(image=image_rgb)["image"].unsqueeze(0).to(self.device)
        return [model.forward_features(tensor).float().cpu().numpy() for model in self.cnn_models]

    def predict(self, image_path: str, age: float | None, sex: str, body_site: str) -> dict:
        t0 = time.monotonic()

        abcd_values = extract_abcd_features(image_path, self.yolo)
        yolo_found_lesion = not bool(np.isnan(abcd_values).any())
        # Native Python floats, not numpy.float64 -- these values flow into
        # llm_explainer.py, which json.dumps()s them directly (no numpy-aware
        # encoder like Flask's), and arithmetic on a numpy scalar anywhere
        # downstream (e.g. `value > threshold`) produces a numpy.bool_ that
        # the same plain json.dumps() call also can't serialize.
        abcd_dict = {name: float(value) for name, value in zip(ABCD_FEATURE_NAMES, abcd_values)}

        sex_clean = sex if sex in VALID_SEX else "missing"
        site_clean = body_site if body_site in VALID_BODY_SITES else "missing"
        age_value = float(age) if age is not None else np.nan

        fold_embeddings = self._cnn_embeddings(image_path)

        fold_scores = []
        for fold in range(N_FOLDS):
            pca_row = self.pcas[fold].transform(fold_embeddings[fold])[0]
            values = {f"cnn_pca_{i}": v for i, v in enumerate(pca_row)}
            values.update(abcd_dict)
            values["age_approx"] = age_value
            values["sex"] = sex_clean
            values["anatom_site_general"] = site_clean

            columns = self.feature_names[fold]
            row = pd.DataFrame([[values[c] for c in columns]], columns=columns)
            score = self.catboosts[fold].predict_proba(row)[0, 1]
            fold_scores.append(float(score))

        score = float(np.mean(fold_scores))
        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info(
            "RiskModel.predict: %.1fms risk_score=%.4f yolo_found_lesion=%s",
            elapsed_ms, score, yolo_found_lesion,
        )

        return {
            "risk_score": score,
            "risk_level": risk_level(score),
            "abcd_features": {name: _nan_to_none(value) for name, value in abcd_dict.items()},
            "yolo_found_lesion": yolo_found_lesion,
            "fold_scores": fold_scores,
        }
