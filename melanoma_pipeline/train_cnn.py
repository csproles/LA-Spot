"""Fine-tunes ConvNeXt-Base (full fine-tuning, no frozen layers) under
GroupKFold(5) cross-validation, with focal loss, mixed precision, and
gradient accumulation. Ready to run once `config.IMAGE_DIR`/
`config.METADATA_CSV` point at real data.

Nothing executes at import time -- `main()` must be called explicitly
(directly, or via `python train_cnn.py`).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

import config
from dataset import get_dataloader
from utils import AverageMeter, FocalLoss, compute_paauc, save_checkpoint, seed_everything


class MelanomaCNN(nn.Module):
    """ConvNeXt-Base backbone (via timm) + a Linear(EMBEDDING_DIM, 1) +
    Sigmoid head. All backbone parameters remain trainable (this pipeline
    does full fine-tuning, no layer freezing) -- if partial freezing is
    ever wanted, freeze via `for p in self.backbone.stem.parameters():
    p.requires_grad = False`-style calls on specific timm submodules, not
    by editing this class's structure.
    """

    def __init__(self, model_name: str = config.MODEL_NAME, pretrained: bool = True) -> None:
        super().__init__()
        # num_classes=0 tells timm to drop its own classifier and return the
        # pooled (B, EMBEDDING_DIM) feature vector directly from forward() --
        # this is the exact vector extract_embeddings.py later saves.
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.head = nn.Sequential(nn.Linear(config.EMBEDDING_DIM, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features).squeeze(1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the pre-head, EMBEDDING_DIM-dim pooled feature vector.
        extract_embeddings.py calls this instead of `forward()` to get
        embeddings rather than predictions -- see that module's docstring
        for why this explicit-method approach is used instead of a forward
        hook."""
        return self.backbone(x)


def build_optimizer_and_scheduler(model: nn.Module, epochs: int = config.EPOCHS):
    optimizer = AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    return optimizer, scheduler


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    grad_accum_steps: int = config.GRAD_ACCUM_STEPS,
) -> float:
    model.train()
    loss_meter = AverageMeter()
    optimizer.zero_grad()

    n_batches = len(loader)
    for step, batch in enumerate(tqdm(loader, desc="train", leave=False)):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            preds = model(images)
            # Divide by grad_accum_steps so GRAD_ACCUM_STEPS consecutive
            # backward() calls sum to the same gradient magnitude a single
            # step over the full effective batch (BATCH_SIZE *
            # GRAD_ACCUM_STEPS) would have produced -- without this, the
            # accumulated gradient would be `grad_accum_steps` times too
            # large relative to the LR that was tuned for the effective
            # batch size.
            loss = criterion(preds, targets) / grad_accum_steps

        scaler.scale(loss).backward()

        is_accum_boundary = (step + 1) % grad_accum_steps == 0
        is_last_batch = (step + 1) == n_batches
        if is_accum_boundary or is_last_batch:
            # Reset logic: only step the optimizer (and only THEN zero the
            # gradients) once `grad_accum_steps` batches' worth of gradients
            # have been accumulated into `.grad`, OR at the final batch of
            # the epoch so any leftover partial accumulation (when
            # len(loader) isn't an exact multiple of grad_accum_steps) still
            # gets applied instead of silently discarded. Zeroing BEFORE
            # this point would erase gradients from batches already
            # accumulated but not yet applied.
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        loss_meter.update(loss.item() * grad_accum_steps, images.size(0))

    return loss_meter.avg


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    loss_meter = AverageMeter()
    all_targets: list[float] = []
    all_preds: list[float] = []

    for batch in tqdm(loader, desc="val", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            preds = model(images)
            loss = criterion(preds, targets)

        loss_meter.update(loss.item(), images.size(0))
        all_targets.extend(targets.detach().cpu().numpy().tolist())
        all_preds.extend(preds.detach().float().cpu().numpy().tolist())

    paauc = compute_paauc(np.array(all_targets), np.array(all_preds), min_tpr=config.MIN_TPR)
    return loss_meter.avg, paauc


def run_fold(
    fold: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    df: pd.DataFrame,
    device: torch.device,
) -> float:
    """Trains one fold end to end; returns that fold's best validation
    pAUC. Instantiates a fresh model/optimizer/scheduler for this fold --
    folds never share weights, avoiding any cross-fold leakage."""
    train_loader = get_dataloader(df, train_idx, mode="train")
    val_loader = get_dataloader(df, val_idx, mode="val")

    model = MelanomaCNN().to(device)
    criterion = FocalLoss(gamma=config.FOCAL_GAMMA, alpha=config.FOCAL_ALPHA)
    optimizer, scheduler = build_optimizer_and_scheduler(model)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    best_paauc = -float("inf")
    checkpoint_path = config.CHECKPOINT_DIR / f"fold_{fold}_best.pt"

    for epoch in range(config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_paauc = validate_one_epoch(model, val_loader, criterion, device)
        scheduler.step()

        print(f"[fold {fold}] epoch {epoch + 1}/{config.EPOCHS} "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_pAUC={val_paauc:.4f}")

        if val_paauc > best_paauc:
            best_paauc = val_paauc
            save_checkpoint(model, optimizer, epoch, fold, checkpoint_path, extra={"val_paauc": val_paauc})

    return best_paauc


def main() -> None:
    config.ensure_dirs()
    seed_everything(config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_csv(config.METADATA_CSV)
    gkf = GroupKFold(n_splits=config.N_FOLDS)
    groups = df[config.GROUP_COL].to_numpy()

    fold_paaucs: list[float] = []
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, df[config.TARGET_COL], groups=groups)):
        best_paauc = run_fold(fold, train_idx, val_idx, df, device)
        fold_paaucs.append(best_paauc)

    print("\nPer-fold best pAUC:")
    for fold, p in enumerate(fold_paaucs):
        print(f"  fold {fold}: {p:.4f}")
    print(f"Mean pAUC across {config.N_FOLDS} folds: {np.mean(fold_paaucs):.4f} "
          f"(std {np.std(fold_paaucs, ddof=1):.4f})")


if __name__ == "__main__":
    main()
