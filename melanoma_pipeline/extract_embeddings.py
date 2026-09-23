"""For each fold's best CNN checkpoint, extracts EMBEDDING_DIM-dim pooled
feature vectors for that fold's train and validation images, and saves them
alongside their labels/isic_ids for train_catboost.py to consume.

Nothing executes at import time -- `main()` must be called explicitly
(directly, or via `python extract_embeddings.py`), same convention as
train_cnn.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

import config
from dataset import get_dataloader
from train_cnn import MelanomaCNN
from utils import load_checkpoint, seed_everything

# Why `model.forward_features(x)` (attribute/method-based head removal)
# instead of a forward hook:
#   A forward hook (`module.register_forward_hook(...)`) captures whatever
#   tensor flows through a NAMED submodule at call time -- it works, but it
#   ties this extraction code to knowing the exact internal module name/
#   position to hook (e.g. "backbone", or a specific timm block), which is
#   fragile if the model definition in train_cnn.py ever changes (a
#   different head architecture, a renamed submodule, an extra wrapper
#   layer). It also requires remembering to remove the hook afterward to
#   avoid leaking memory/side effects if the same model object is reused
#   elsewhere.
#   Head removal via an explicit `forward_features()` method (already
#   defined on MelanomaCNN in train_cnn.py) has neither problem: it's a
#   stable public contract on the model class itself, is trivially testable
#   in isolation, and can't accidentally capture an activation from the
#   wrong layer. This is the simpler, more robust choice for a pipeline
#   where the embedding boundary (pre-head, EMBEDDING_DIM-dim) is a fixed,
#   important contract that train_catboost.py depends on downstream.


@torch.no_grad()
def extract_embeddings_for_loader(
    model: torch.nn.Module,
    loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Runs every batch in `loader` through `model.forward_features` and
    returns (embeddings [N, EMBEDDING_DIM], labels [N], isic_ids [N])."""
    model.eval()
    all_embeddings: list[np.ndarray] = []
    all_labels: list[float] = []
    all_ids: list[str] = []

    for batch in tqdm(loader, desc="extract", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            embeddings = model.forward_features(images)
        all_embeddings.append(embeddings.float().cpu().numpy())
        all_labels.extend(batch["target"].cpu().numpy().tolist())
        all_ids.extend(batch["isic_id"])

    return np.vstack(all_embeddings), np.array(all_labels), np.array(all_ids)


def main() -> None:
    config.ensure_dirs()
    seed_everything(config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_csv(config.METADATA_CSV)
    gkf = GroupKFold(n_splits=config.N_FOLDS)
    groups = df[config.GROUP_COL].to_numpy()

    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, df[config.TARGET_COL], groups=groups)):
        checkpoint_path = config.CHECKPOINT_DIR / f"fold_{fold}_best.pt"
        if not checkpoint_path.exists():
            print(f"[fold {fold}] checkpoint not found at {checkpoint_path}, skipping "
                  f"(run train_cnn.py first)")
            continue

        model = MelanomaCNN(pretrained=False).to(device)  # weights come from the checkpoint, not ImageNet init
        load_checkpoint(checkpoint_path, model, optimizer=None, map_location=device)

        train_loader = get_dataloader(df, train_idx, mode="val")  # mode="val": deterministic transforms for embedding extraction, not train-time augmentation
        val_loader = get_dataloader(df, val_idx, mode="val")

        train_emb, train_labels, train_ids = extract_embeddings_for_loader(model, train_loader, device)
        val_emb, val_labels, val_ids = extract_embeddings_for_loader(model, val_loader, device)

        np.save(config.EMBEDDING_DIR / f"fold_{fold}_train_embeddings.npy", train_emb)
        np.save(config.EMBEDDING_DIR / f"fold_{fold}_train_labels.npy", train_labels)
        np.save(config.EMBEDDING_DIR / f"fold_{fold}_train_isic_ids.npy", train_ids)

        np.save(config.EMBEDDING_DIR / f"fold_{fold}_val_embeddings.npy", val_emb)
        np.save(config.EMBEDDING_DIR / f"fold_{fold}_val_labels.npy", val_labels)
        np.save(config.EMBEDDING_DIR / f"fold_{fold}_val_isic_ids.npy", val_ids)

        print(f"[fold {fold}] saved train embeddings {train_emb.shape}, val embeddings {val_emb.shape} "
              f"-> {config.EMBEDDING_DIR}")


if __name__ == "__main__":
    main()
