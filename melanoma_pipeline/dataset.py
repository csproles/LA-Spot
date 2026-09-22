"""torch Dataset + albumentations augmentation pipelines + a DataLoader
factory for the CNN stage. Importable without side effects -- no dataset is
instantiated and no image is read at import time.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

import config

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

Mode = Literal["train", "val"]


# ------------------------------------------------------------ transforms --
def build_train_transforms(image_size: int = config.IMAGE_SIZE) -> A.Compose:
    return A.Compose([
        A.RandomResizedCrop(size=(image_size, image_size), scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=30, p=0.5),
        A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(0.05, 0.15),
                         hole_width_range=(0.05, 0.15), p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def build_val_transforms(image_size: int = config.IMAGE_SIZE) -> A.Compose:
    return A.Compose([
        A.Resize(height=int(image_size * 1.14), width=int(image_size * 1.14)),  # resize-then-center-crop margin
        A.CenterCrop(height=image_size, width=image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


train_transforms: A.Compose = build_train_transforms()
val_transforms: A.Compose = build_val_transforms()


# ---------------------------------------------------------------- dataset --
class ISICDataset(Dataset):
    """Loads (image, label) pairs from a dataframe slice. Expects `df` to
    have at least an image-id column (used to locate the file under
    `image_dir`) and, unless `has_labels=False`, `config.TARGET_COL`.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: Path,
        transforms: A.Compose,
        image_id_col: str = "isic_id",
        has_labels: bool = True,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transforms = transforms
        self.image_id_col = image_id_col
        self.has_labels = has_labels

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_image_path(self, image_id: str) -> Path:
        # Accept either a bare id ("ISIC_0000000") or an id that already has
        # an extension in the dataframe -- try the common ISIC extensions if
        # a bare id is given, without hardcoding a single extension.
        candidate = self.image_dir / image_id
        if candidate.suffix:
            return candidate
        for ext in (".jpg", ".jpeg", ".png"):
            p = self.image_dir / f"{image_id}{ext}"
            if p.exists():
                return p
        return self.image_dir / f"{image_id}.jpg"  # fallback; will raise a clear error on read

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        image_id = str(row[self.image_id_col])
        image_path = self._resolve_image_path(image_id)

        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"could not read image for id={image_id!r} at {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        augmented = self.transforms(image=image_rgb)
        image_tensor = augmented["image"]

        item = {"image": image_tensor, "isic_id": image_id}
        if self.has_labels:
            item["target"] = torch.tensor(float(row[config.TARGET_COL]), dtype=torch.float32)
        return item


# ---------------------------------------------------------- dataloader ----
def get_dataloader(
    df: pd.DataFrame,
    fold_indices: np.ndarray,
    mode: Mode,
    cfg=config,
    image_dir: Path | None = None,
    image_id_col: str = "isic_id",
    num_workers: int = 4,
) -> DataLoader:
    """Builds a DataLoader over `df.iloc[fold_indices]` using the transform
    pipeline appropriate for `mode` ("train" -> augmented + shuffled,
    "val" -> deterministic + not shuffled)."""
    subset = df.iloc[fold_indices]
    transforms = train_transforms if mode == "train" else val_transforms
    dataset = ISICDataset(
        subset,
        image_dir=image_dir or cfg.IMAGE_DIR,
        transforms=transforms,
        image_id_col=image_id_col,
        has_labels=True,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=(mode == "train"),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=(mode == "train"),
    )
