"""SHAP-based explainability for the fitted CatBoost models.

Two things this module produces, per fold:
  1. Per-feature SHAP attributions for every validated ABCD/structured
     feature (named individually -- e.g. "solidity contributed -0.08 to
     this case's score") -- a direct, per-patient upgrade over a plain
     logistic-regression's global coefficients, suitable for patient-facing
     explanation text (subject to this project's existing policy against
     presenting model internals as a diagnosis).
  2. A modality-level split: summed |SHAP| across the CNN-embedding PCA
     components vs. summed |SHAP| across the named ABCD features, reported
     as a percentage (e.g. "visual/deep-learning signal contributed ~40%
     of this case's score, structured ABCD features contributed ~60%").
     Individual PCA-component SHAP values are NOT themselves human-
     meaningful (a component is a rotated/scaled linear combination of 1024
     CNN activations, not a named concept) -- only this aggregate split is
     interpretable on its own. If you need to know WHERE in an image the
     CNN was drawing signal from, that requires Grad-CAM on the CNN
     backbone directly (see docs/design/multimodal_pipeline_design.md's
     Section 7); this module does not implement that, only the
     SHAP-based structured/CNN attribution split.

Uses TreeExplainer (exact, fast for CatBoost's tree ensembles) -- not the
slower, approximate KernelExplainer.

Nothing executes at import time -- `main()` must be called explicitly
(directly, or via `python explain.py`), same convention as every other
entry-point module in this package.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier

import config
from features import ABCD_FEATURE_NAMES
from train_catboost import align_abcd_features, load_abcd_lookup


def load_fold_artifacts(fold: int) -> tuple[CatBoostClassifier, list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Loads one fold's saved CatBoost model, the exact feature-name order
    train_catboost.py used for that fold (PCA component count can vary
    fold-to-fold on small data, so this must be read per-fold, never
    assumed fixed), and rebuilds that fold's validation feature matrix the
    same way train_catboost.py did (PCA fit on that fold's TRAIN embeddings
    only -- refit here identically rather than re-saving a pickled PCA
    object, so this module has no new artifact-versioning surface to keep
    in sync with train_catboost.py's own).
    """
    model_path = config.OUTPUT_DIR / f"catboost_fold_{fold}.cbm"
    names_path = config.OUTPUT_DIR / f"catboost_fold_{fold}_feature_names.npy"
    if not model_path.exists() or not names_path.exists():
        raise FileNotFoundError(
            f"fold {fold} artifacts not found ({model_path}, {names_path}) -- "
            f"run train_catboost.py first."
        )

    model = CatBoostClassifier()
    model.load_model(str(model_path))
    feature_names = np.load(names_path, allow_pickle=True).tolist()

    from train_catboost import fit_transform_pca  # local import to avoid a module-level circular-import risk

    train_emb = np.load(config.EMBEDDING_DIR / f"fold_{fold}_train_embeddings.npy")
    train_ids = np.load(config.EMBEDDING_DIR / f"fold_{fold}_train_isic_ids.npy", allow_pickle=True)
    val_emb = np.load(config.EMBEDDING_DIR / f"fold_{fold}_val_embeddings.npy")
    val_ids = np.load(config.EMBEDDING_DIR / f"fold_{fold}_val_isic_ids.npy", allow_pickle=True)

    abcd_features, abcd_lookup = load_abcd_lookup()
    train_abcd, train_valid = align_abcd_features(train_ids, abcd_features, abcd_lookup)
    val_abcd, val_valid = align_abcd_features(val_ids, abcd_features, abcd_lookup)

    train_emb, train_abcd = train_emb[train_valid], train_abcd[train_valid]
    val_emb, val_abcd, val_ids = val_emb[val_valid], val_abcd[val_valid], val_ids[val_valid]

    train_emb_reduced, val_emb_reduced, _pca = fit_transform_pca(train_emb, val_emb)
    X_val = np.concatenate([val_emb_reduced, val_abcd], axis=1)

    assert X_val.shape[1] == len(feature_names), (
        f"rebuilt val feature matrix has {X_val.shape[1]} columns but the saved "
        f"feature-name list for fold {fold} has {len(feature_names)} -- PCA component "
        f"count may have changed since training; re-run train_catboost.py."
    )
    return model, feature_names, X_val, val_ids, val_abcd


def compute_shap_values(model: CatBoostClassifier, X: np.ndarray) -> np.ndarray:
    """TreeExplainer SHAP values for the positive (malignant) class. Exact
    (not sampled/approximated) for tree ensembles -- no background dataset
    or sampling parameter needed, unlike KernelExplainer."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    # CatBoost binary classification: TreeExplainer returns a single
    # (n_samples, n_features) array of SHAP values for the positive class
    # (log-odds space) -- no [:, :, 1] class-indexing needed, unlike some
    # sklearn multi-class classifiers' SHAP output shape.
    return shap_values


def modality_split(shap_values: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """Per-sample modality-level attribution: summed |SHAP| across CNN PCA
    components vs. summed |SHAP| across named ABCD features, as a
    percentage of that sample's total |SHAP| mass."""
    cnn_cols = [i for i, name in enumerate(feature_names) if name.startswith("cnn_pca_")]
    abcd_cols = [i for i, name in enumerate(feature_names) if name in ABCD_FEATURE_NAMES]

    abs_shap = np.abs(shap_values)
    cnn_mass = abs_shap[:, cnn_cols].sum(axis=1)
    abcd_mass = abs_shap[:, abcd_cols].sum(axis=1)
    total_mass = cnn_mass + abcd_mass
    total_mass_safe = np.where(total_mass > 0, total_mass, 1.0)  # avoid divide-by-zero for an all-zero-SHAP row

    return pd.DataFrame({
        "cnn_pct": 100.0 * cnn_mass / total_mass_safe,
        "abcd_pct": 100.0 * abcd_mass / total_mass_safe,
    })


def per_feature_summary(shap_values: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """Population-level mean |SHAP| per named feature, sorted descending --
    an aggregate importance ranking (not a per-patient explanation; use
    `shap_values` directly, row-indexed by image_id, for a specific case)."""
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    return (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def explain_fold(fold: int) -> None:
    model, feature_names, X_val, val_ids, _val_abcd = load_fold_artifacts(fold)
    shap_values = compute_shap_values(model, X_val)

    # 1. Per-image, per-feature attributions (full matrix -- every named
    # ABCD feature and every CNN PCA component, one row per validation image).
    per_image = pd.DataFrame(shap_values, columns=feature_names)
    per_image.insert(0, "isic_id", val_ids)
    per_image_path = config.OUTPUT_DIR / f"shap_fold_{fold}_per_image.csv"
    per_image.to_csv(per_image_path, index=False)

    # 2. Modality-level split, per image.
    modality = modality_split(shap_values, feature_names)
    modality.insert(0, "isic_id", val_ids)
    modality_path = config.OUTPUT_DIR / f"shap_fold_{fold}_modality_split.csv"
    modality.to_csv(modality_path, index=False)

    # 3. Population-level feature-importance ranking.
    summary = per_feature_summary(shap_values, feature_names)
    summary_path = config.OUTPUT_DIR / f"shap_fold_{fold}_feature_importance.csv"
    summary.to_csv(summary_path, index=False)

    abcd_only = summary[summary["feature"].isin(ABCD_FEATURE_NAMES)]
    print(f"[fold {fold}] wrote {per_image_path}, {modality_path}, {summary_path}")
    print(f"[fold {fold}] mean modality split: CNN={modality['cnn_pct'].mean():.1f}% "
          f"ABCD={modality['abcd_pct'].mean():.1f}%")
    print(f"[fold {fold}] top 5 ABCD features by mean |SHAP|:")
    for _, row in abcd_only.head(5).iterrows():
        print(f"    {row['feature']}: {row['mean_abs_shap']:.4f}")


def main() -> None:
    config.ensure_dirs()
    for fold in range(config.N_FOLDS):
        try:
            explain_fold(fold)
        except FileNotFoundError as e:
            print(f"[fold {fold}] skipped: {e}")


if __name__ == "__main__":
    main()
