"""Trains the downstream CatBoost classifier on (PCA-reduced CNN embedding
++ validated ABCD feature) concatenated vectors, per fold, reusing the
exact fold train/val embeddings extract_embeddings.py already saved (so
there is no re-derivation of fold membership here -- this module only
reads what extract_embeddings.py wrote).

PCA on the CNN embedding is fit ONCE PER FOLD, on that fold's training rows
only, then applied to both that fold's train and val rows -- never fit on
pooled/all-fold data and never fit including validation rows. This is the
same per-fold-fit discipline every other learned step in this pipeline
(the CNN itself, CatBoost) already follows, and it's what keeps the
~1024-dim CNN embedding from numerically swamping the much smaller
(N_ABCD_FEATURES-dim, currently 17) validated ABCD feature vector -- see
config.PCA_N_COMPONENTS's docstring and docs/design/ for the reasoning.

Nothing executes at import time -- `main()` must be called explicitly
(directly, or via `python train_catboost.py`).
"""
from __future__ import annotations

import numpy as np
from catboost import CatBoostClassifier
from sklearn.decomposition import PCA

import config
from features import ABCD_FEATURE_NAMES, N_ABCD_FEATURES
from utils import compute_paauc, seed_everything


def load_abcd_lookup() -> tuple[np.ndarray, dict[str, int]]:
    """Loads features.py's __main__-produced (N, N_ABCD_FEATURES) array and
    its matching isic_id index, and builds an id -> row-index lookup so
    ABCD features can be aligned to whatever isic_id order a given fold's
    CNN embeddings are in (the two arrays are not guaranteed to share row
    order)."""
    abcd_features = np.load(config.OUTPUT_DIR / "abcd_features.npy")
    abcd_index = np.load(config.OUTPUT_DIR / "abcd_index.npy", allow_pickle=True)
    assert abcd_features.shape[1] == N_ABCD_FEATURES, (
        f"abcd_features.npy has {abcd_features.shape[1]} columns but the live "
        f"ABCD_FEATURE_NAMES config has {N_ABCD_FEATURES} -- re-run features.py, "
        f"the cached array is stale relative to the current feature definition."
    )
    lookup = {str(image_id): i for i, image_id in enumerate(abcd_index)}
    return abcd_features, lookup


def align_abcd_features(
    isic_ids: np.ndarray,
    abcd_features: np.ndarray,
    abcd_lookup: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (aligned_abcd_features [N, N_ABCD_FEATURES], valid_mask [N])
    for the given isic_ids, in that same order. Rows whose isic_id has no
    matching ABCD features (e.g. that image had no lesion detected) get
    `valid_mask=False` rather than a zero-filled/guessed feature vector --
    silently zero-filling would misrepresent "no data" as "ABCD features
    indicating a specific value", which is worse than excluding the row.
    """
    aligned = np.full((len(isic_ids), N_ABCD_FEATURES), np.nan, dtype=np.float64)
    for i, image_id in enumerate(isic_ids):
        abcd_row_idx = abcd_lookup.get(str(image_id))
        if abcd_row_idx is not None:
            aligned[i] = abcd_features[abcd_row_idx]
    valid_mask = ~np.isnan(aligned).any(axis=1)
    return aligned, valid_mask


def fit_transform_pca(train_embeddings: np.ndarray, val_embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray, PCA]:
    """Fits PCA on TRAIN embeddings only, transforms both train and val with
    that fitted object. `n_components` is clipped to
    min(config.PCA_N_COMPONENTS, n_train_samples, n_features) so a small
    fold (fewer training rows than the configured component count) doesn't
    raise -- PCA's own component count can never exceed min(n_samples,
    n_features)."""
    n_components = min(config.PCA_N_COMPONENTS, train_embeddings.shape[0], train_embeddings.shape[1])
    pca = PCA(n_components=n_components, random_state=config.SEED)
    train_reduced = pca.fit_transform(train_embeddings)
    val_reduced = pca.transform(val_embeddings)
    return train_reduced, val_reduced, pca


def main() -> None:
    config.ensure_dirs()
    seed_everything(config.SEED)

    abcd_features, abcd_lookup = load_abcd_lookup()

    fold_paaucs: list[float] = []
    oof_targets: list[float] = []
    oof_preds: list[float] = []

    for fold in range(config.N_FOLDS):
        train_emb_path = config.EMBEDDING_DIR / f"fold_{fold}_train_embeddings.npy"
        if not train_emb_path.exists():
            print(f"[fold {fold}] embeddings not found at {train_emb_path}, skipping "
                  f"(run train_cnn.py then extract_embeddings.py first)")
            continue

        train_emb = np.load(train_emb_path)
        train_labels = np.load(config.EMBEDDING_DIR / f"fold_{fold}_train_labels.npy")
        train_ids = np.load(config.EMBEDDING_DIR / f"fold_{fold}_train_isic_ids.npy", allow_pickle=True)

        val_emb = np.load(config.EMBEDDING_DIR / f"fold_{fold}_val_embeddings.npy")
        val_labels = np.load(config.EMBEDDING_DIR / f"fold_{fold}_val_labels.npy")
        val_ids = np.load(config.EMBEDDING_DIR / f"fold_{fold}_val_isic_ids.npy", allow_pickle=True)

        # Align ABCD features first and drop rows with no valid ABCD vector
        # from BOTH the embedding and label arrays, before PCA ever sees them
        # -- PCA must never be fit on a row that will later be discarded, and
        # more importantly must never see validation-fold data during fit.
        train_abcd, train_abcd_valid = align_abcd_features(train_ids, abcd_features, abcd_lookup)
        val_abcd, val_abcd_valid = align_abcd_features(val_ids, abcd_features, abcd_lookup)

        train_emb, train_abcd, y_train = train_emb[train_abcd_valid], train_abcd[train_abcd_valid], train_labels[train_abcd_valid]
        val_emb, val_abcd, y_val = val_emb[val_abcd_valid], val_abcd[val_abcd_valid], val_labels[val_abcd_valid]

        # PCA: fit on this fold's TRAINING embeddings only, transform both
        # train and val with that same fitted object.
        train_emb_reduced, val_emb_reduced, pca = fit_transform_pca(train_emb, val_emb)

        X_train = np.concatenate([train_emb_reduced, train_abcd], axis=1)
        X_val = np.concatenate([val_emb_reduced, val_abcd], axis=1)

        n_pos, n_neg = int(y_train.sum()), int((y_train == 0).sum())
        scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

        model = CatBoostClassifier(
            iterations=config.CATBOOST_ITERATIONS,
            learning_rate=config.CATBOOST_LR,
            depth=config.CATBOOST_DEPTH,
            loss_function="Logloss",
            eval_metric="AUC",
            scale_pos_weight=scale_pos_weight,
            random_seed=config.SEED,
            verbose=False,
        )
        model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)

        val_scores = model.predict_proba(X_val)[:, 1]
        val_paauc = compute_paauc(y_val, val_scores, min_tpr=config.MIN_TPR)
        fold_paaucs.append(val_paauc)
        oof_targets.extend(y_val.tolist())
        oof_preds.extend(val_scores.tolist())

        model_path = config.OUTPUT_DIR / f"catboost_fold_{fold}.cbm"
        model.save_model(str(model_path))

        # Feature names for this fold's model, in the exact column order
        # X_train/X_val were built in -- SHAP (explain.py) and any other
        # downstream introspection must use THIS list, not a hardcoded one,
        # since the PCA component count can vary fold-to-fold on small data.
        pca_feature_names = [f"cnn_pca_{i}" for i in range(pca.n_components_)]
        fold_feature_names = pca_feature_names + list(ABCD_FEATURE_NAMES)
        np.save(config.OUTPUT_DIR / f"catboost_fold_{fold}_feature_names.npy", np.array(fold_feature_names))

        print(f"[fold {fold}] n_train={len(X_train)} n_val={len(X_val)} "
              f"pca_components={pca.n_components_} (explained_variance={pca.explained_variance_ratio_.sum():.3f}) "
              f"feature_dim={X_train.shape[1]} scale_pos_weight={scale_pos_weight:.2f} "
              f"val_pAUC={val_paauc:.4f} -> {model_path}")

    print("\nPer-fold pAUC:")
    for fold, p in enumerate(fold_paaucs):
        print(f"  fold {fold}: {p:.4f}")
    if fold_paaucs:
        print(f"Mean pAUC across folds: {np.mean(fold_paaucs):.4f} (std {np.std(fold_paaucs, ddof=1):.4f})")

    oof_paauc = compute_paauc(np.array(oof_targets), np.array(oof_preds), min_tpr=config.MIN_TPR)
    print(f"OOF (pooled across all folds) pAUC: {oof_paauc:.4f}")


if __name__ == "__main__":
    main()
