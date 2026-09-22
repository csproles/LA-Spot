"""Shared, dependency-light utilities: the pAUC metric, focal loss,
reproducibility seeding, checkpoint I/O, and a running-average tracker.
Importable without side effects; nothing here touches the filesystem except
the explicit save_checkpoint/load_checkpoint calls.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import auc, roc_curve
from torch import nn


# ------------------------------------------------------------------ pAUC --
def compute_paauc(y_true: np.ndarray, y_score: np.ndarray, min_tpr: float = 0.80,
                   normalize: bool = False) -> float:
    """ISIC-2024-reference-style partial AUC above TPR=min_tpr (the
    label/score-flip construction), NOT an average TPR over some FPR
    interval -- see the correction history at the bottom of this docstring
    for what earlier versions of this function got wrong.

    Method (label/score flip -> partial AUC on the flipped problem):
      1. Flip both labels and scores: y_flipped = 1 - y_true,
         scores_flipped = 1 - y_score. This is a lossless re-expression, not
         an approximation: AUC(1 - y, 1 - s) == AUC(y, s) always (flipping
         both together just swaps which class roc_curve treats as
         "positive"), but it turns "the region where TPR >= min_tpr" on the
         original curve into "the region where FPR <= max_fpr" on the
         flipped curve -- and FPR-bounded truncation is what roc_curve's own
         point ordering (sorted by ascending threshold / ascending FPR)
         supports directly, which is why this flip is done at all rather
         than truncating the original curve on its TPR axis.
      2. fpr, tpr = roc_curve(y_flipped, scores_flipped).
      3. max_fpr = 1 - min_tpr.
      4. Truncate to the points with fpr <= max_fpr, inserting one
         linearly-interpolated point exactly at fpr == max_fpr first (via
         np.interp against the already-sorted fpr axis) -- without this, if
         no ROC vertex lands exactly on the max_fpr boundary, the truncated
         region's right edge is undefined and the integral can over- or
         under-shoot depending on where the nearest real vertex happens to
         fall.
      5. raw_area = sklearn.metrics.auc(fpr_truncated, tpr_truncated) -- the
         area under the flipped curve's [0, max_fpr] region, which by the
         flip-invariance in step 1 equals the area under the ORIGINAL
         curve's [min_tpr, 1] TPR region. This is a real partial-AUC
         integral (area under a curve), not an average of tpr values over
         an interval.
      6. Returns raw_area by default -- this is the number the ISIC-2024
         Kaggle leaderboard itself reports (typically ~0.15-0.20 for a
         competitive model at min_tpr=0.80, bounded to [0, max_fpr], NOT
         rescaled to [0, 1]). Pass `normalize=True` to instead return
         `raw_area / max_fpr`, bounded to [0, 1], if a 0-1-scaled number is
         more convenient for a particular comparison.

    Verified by direct computation (not just derived) against three known
    reference points, including tied scores and an exact-vertex boundary
    case (see this module's own verification snippets/tests):
      - perfect classifier:   raw=0.1999... (~0.20), normalized=1.00  (min_tpr=0.80)
      - random/diagonal:      raw~=0.02, normalized~=0.10             (min_tpr=0.80, converges as n grows)
      - completely reversed:  raw=0.00, normalized=0.00               (min_tpr=0.80)
      - min_tpr=0 (no truncation): raw == normalized == the full,
        un-truncated `sklearn.metrics.roc_auc_score(y_true, y_score)`,
        confirming the flip-invariance property holds end to end.

    *** CORRECTION HISTORY *** two earlier versions of this function were
    both wrong, for different reasons:
      - v1 computed "AUC where FPR <= threshold" -- the wrong ROC region
        entirely (a low-FPR region, not the high-TPR/high-sensitivity
        region this metric is about).
      - v2 computed "AUC where TPR >= min_tpr" directly on the ORIGINAL
        (non-flipped) curve, then normalized by the OBSERVED, classifier-
        dependent FPR width of the kept region. That is not a partial AUC:
        dividing a region's area by its own width recovers an *average TPR*
        over that interval, not an area comparable across classifiers with
        different curve shapes -- it happened to return exactly 1.0 for a
        perfect classifier (a coincidence of that specific edge case, where
        the kept region's width and the "correct" width match), but gives
        the wrong value in the general case, including the random-diagonal
        and reversed-classifier reference points above. This version (v3)
        is the reference label/score-flip construction and matches all
        three worked examples exactly.
    """
    if len(np.unique(y_true)) < 2:
        return float("nan")  # AUC (partial or full) is undefined with a single class present

    max_fpr = 1.0 - min_tpr
    if max_fpr <= 0:
        return float("nan")

    y_flipped = 1 - np.asarray(y_true)
    scores_flipped = 1 - np.asarray(y_score)
    fpr, tpr, _ = roc_curve(y_flipped, scores_flipped)

    if max_fpr < fpr.max():
        # fpr is sorted ascending (roc_curve's own guarantee); keep every
        # point with fpr <= max_fpr, then append one interpolated point
        # exactly at the max_fpr boundary so the truncated curve's right
        # edge is well-defined.
        stop = int(np.searchsorted(fpr, max_fpr, side="right"))
        tpr_at_boundary = float(np.interp(max_fpr, fpr, tpr))
        fpr_trunc = np.concatenate([fpr[:stop], [max_fpr]])
        tpr_trunc = np.concatenate([tpr[:stop], [tpr_at_boundary]])
    else:
        fpr_trunc, tpr_trunc = fpr, tpr

    raw_area = float(auc(fpr_trunc, tpr_trunc))
    return raw_area / max_fpr if normalize else raw_area


# ------------------------------------------------------------- focal loss --
class FocalLoss(nn.Module):
    """Binary focal loss (Lin et al., 2017), consuming PROBABILITIES (i.e.
    the model already applies Sigmoid internally, per this pipeline's
    Linear(1024, 1) + Sigmoid head spec) rather than raw logits.

    NOTE on numerical stability: the more common/robust pattern is to keep
    the model's head as raw logits (no Sigmoid layer) and implement focal
    loss on top of `F.binary_cross_entropy_with_logits`, which is numerically
    stable via the log-sum-exp trick even for very confident (near-0/near-1)
    predictions. Because this pipeline's model head applies Sigmoid BEFORE
    the loss, probabilities are clamped away from exactly 0/1 below to avoid
    log(0); this is safe but marginally less numerically stable at extreme
    confidence than the logits-based formulation. Swap to a logits-based
    version (drop the model's Sigmoid, use `F.binary_cross_entropy_with_logits`
    here) if training shows instability at high confidence.

    Focal loss math: standard BCE is -[y*log(p) + (1-y)*log(1-p)]. Focal
    loss adds two things on top of that per-sample BCE term:
      1. An alpha-balancing term (`alpha` for the positive class, `1-alpha`
         for the negative class) -- a plain class-frequency reweighting,
         the same role `scale_pos_weight` plays for CatBoost later in this
         pipeline, just expressed as a loss-side multiplier instead of a
         sample weight.
      2. A `(1 - p_t)^gamma` modulating factor, where `p_t` is the model's
         predicted probability for the TRUE class (p if y=1, 1-p if y=0).
         When the model is already confident and correct (p_t close to 1),
         (1-p_t)^gamma is close to 0, which DOWN-WEIGHTS the loss
         contribution from already-easy, correctly-classified examples --
         the mechanism that lets focal loss concentrate gradient signal on
         hard/misclassified examples instead of being dominated by the
         (here: overwhelming) volume of easy benign negatives.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, eps: float = 1e-7) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps

    def forward(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = probs.clamp(self.eps, 1.0 - self.eps)  # avoid log(0) at the extremes
        targets = targets.float()

        p_t = torch.where(targets == 1, probs, 1.0 - probs)
        alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)

        bce = -torch.log(p_t)
        focal_weight = (1.0 - p_t).pow(self.gamma)
        loss = alpha_t * focal_weight * bce
        return loss.mean()


# ------------------------------------------------------ reproducibility ---
def seed_everything(seed: int) -> None:
    """Seed python/numpy/torch (CPU + all CUDA devices) for reproducibility.
    Does NOT force `torch.backends.cudnn.deterministic = True` by default,
    since that trades meaningful training speed for exact bit-reproducibility
    this pipeline doesn't strictly require -- uncomment below if exact
    reproducibility matters more than speed for a given run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


# ----------------------------------------------------------- checkpoints --
def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    fold: int,
    path: Path,
    extra: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "fold": fold,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


# ------------------------------------------------------------ tracking ----
class AverageMeter:
    """Tracks a running mean of a scalar (loss, metric, ...) across an
    arbitrary number of `update()` calls, e.g. once per batch within an
    epoch."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum: float = 0.0
        self.count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else float("nan")
