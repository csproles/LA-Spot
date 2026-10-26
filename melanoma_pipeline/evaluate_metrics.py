"""Computes accuracy / precision / recall / specificity / F1 / ROC-AUC / pAUC
for three models on the SAME patient-separated GroupKFold out-of-fold (OOF)
predictions, so their numbers are directly comparable:

  1. cnn    -- ConvNeXt alone: each fold's own `fold_{k}_best.pt` head
               (Linear(EMBEDDING_DIM, 1) + Sigmoid) applied to that fold's
               saved validation embeddings. The embeddings were produced by
               that same checkpoint's backbone with the same val transforms
               validate_one_epoch() uses, so this reproduces the CNN's own
               validation scores without re-running the backbone (sanity-
               checked below against the checkpoint's stored val_pAUC).
  2. abcd   -- CatBoost on the 17 validated ABCD features alone, trained per
               fold with train_catboost.build_catboost() (same settings as
               the fusion model), so the only difference from (3) is the
               missing CNN embedding.
  3. fusion -- CatBoost on PCA(CNN embedding) ++ ABCD, i.e. exactly
               train_catboost.py: loads `catboost_fold_{k}.cbm` if present
               (PCA is re-fit on the fold's training rows -- deterministic,
               random_state=config.SEED -- since train_catboost.py doesn't
               save it), otherwise trains it identically.

Thresholded metrics are reported twice:
  - at a fixed 0.5 cutoff, and
  - at a cross-validated cutoff: for fold k, the threshold maximizing
    balanced accuracy (Youden's J) on the OTHER folds' OOF scores, then
    applied to fold k. The cutoff is never tuned on the rows it is scored on.

Requires train_cnn.py, extract_embeddings.py and features.py to have been
run first (same inputs train_catboost.py needs). Writes
`metrics_report.json` and `oof_predictions.csv` to config.OUTPUT_DIR.

Nothing executes at import time -- `main()` must be called explicitly
(directly, or via `python evaluate_metrics.py`).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, roc_curve

import config
from features import ABCD_FEATURE_NAMES
from train_catboost import align_abcd_features, build_catboost, fit_transform_pca, load_abcd_lookup
from utils import compute_paauc, seed_everything

MODELS = ("cnn", "abcd", "fusion")
FIXED_THRESHOLD = 0.5


def cnn_head_scores(checkpoint_path, embeddings: np.ndarray) -> tuple[np.ndarray, float | None]:
    """Applies the checkpoint's Linear+Sigmoid head to pre-head embeddings.
    Returns (scores, the checkpoint's stored val_pAUC or None)."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint["model_state_dict"]
    weight = state["head.0.weight"].float().numpy()  # (1, EMBEDDING_DIM)
    bias = state["head.0.bias"].float().numpy()      # (1,)
    logits = embeddings.astype(np.float64) @ weight.T.astype(np.float64) + bias.astype(np.float64)
    scores = 1.0 / (1.0 + np.exp(-logits[:, 0]))
    stored = checkpoint.get("val_paauc")
    return scores, (float(stored) if stored is not None else None)


def fusion_scores(fold: int, train_emb, train_abcd, y_train, val_emb, val_abcd, y_val) -> tuple[np.ndarray, str]:
    train_reduced, val_reduced, pca = fit_transform_pca(train_emb, val_emb)
    X_train = np.concatenate([train_reduced, train_abcd], axis=1)
    X_val = np.concatenate([val_reduced, val_abcd], axis=1)

    model_path = config.OUTPUT_DIR / f"catboost_fold_{fold}.cbm"
    if model_path.exists():
        model = CatBoostClassifier()
        model.load_model(str(model_path))
        names_path = config.OUTPUT_DIR / f"catboost_fold_{fold}_feature_names.npy"
        expected = [f"cnn_pca_{i}" for i in range(pca.n_components_)] + list(ABCD_FEATURE_NAMES)
        if names_path.exists():
            saved = [str(n) for n in np.load(names_path, allow_pickle=True)]
            if saved != expected:
                raise ValueError(f"[fold {fold}] {names_path} does not match the re-built feature layout; "
                                 f"the saved model was trained on different inputs than the current outputs/.")
        if len(model.feature_names_) != X_val.shape[1]:
            raise ValueError(f"[fold {fold}] {model_path} expects {len(model.feature_names_)} features, "
                             f"re-built matrix has {X_val.shape[1]}.")
        source = f"loaded {model_path.name}"
    else:
        model, _ = build_catboost(y_train)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
        source = "trained (no saved .cbm found)"
    return model.predict_proba(X_val)[:, 1], source


def abcd_only_scores(train_abcd, y_train, val_abcd, y_val) -> np.ndarray:
    model, _ = build_catboost(y_train)
    model.fit(train_abcd, y_train, eval_set=(val_abcd, y_val), use_best_model=True)
    return model.predict_proba(val_abcd)[:, 1]


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def threshold_metrics(y: np.ndarray, scores: np.ndarray, thresholds: np.ndarray | float) -> dict:
    pred = scores >= thresholds
    truth = y == 1
    tp = int((pred & truth).sum())
    tn = int((~pred & ~truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    precision = _safe_div(tp, tp + fp)
    f1 = _safe_div(2 * tp, 2 * tp + fp + fn)
    return {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "accuracy": _safe_div(tp + tn, len(y)),
        "balanced_accuracy": (recall + specificity) / 2,
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": specificity,
        "f1": f1,
    }


def cross_validated_thresholds(y: np.ndarray, scores: np.ndarray, folds: np.ndarray) -> tuple[np.ndarray, dict]:
    """Per-row threshold: for each fold, the max-Youden-J cutoff picked on
    the other folds' OOF scores only."""
    per_row = np.empty(len(y), dtype=np.float64)
    per_fold = {}
    for fold in np.unique(folds):
        others = folds != fold
        fpr, tpr, thr = roc_curve(y[others], scores[others])
        finite = np.isfinite(thr)
        best = int(np.argmax(np.where(finite, tpr - fpr, -np.inf)))
        per_fold[int(fold)] = float(thr[best])
        per_row[folds == fold] = thr[best]
    return per_row, per_fold


def evaluate(y: np.ndarray, scores: np.ndarray, folds: np.ndarray) -> dict:
    per_fold_auc = [roc_auc_score(y[folds == f], scores[folds == f])
                    for f in np.unique(folds) if len(np.unique(y[folds == f])) == 2]
    cv_thr, cv_thr_by_fold = cross_validated_thresholds(y, scores, folds)
    return {
        "n": int(len(y)),
        "n_malignant": int(y.sum()),
        "roc_auc_pooled_oof": float(roc_auc_score(y, scores)),
        "roc_auc_per_fold_mean": float(np.mean(per_fold_auc)),
        "roc_auc_per_fold_std": float(np.std(per_fold_auc, ddof=1)) if len(per_fold_auc) > 1 else float("nan"),
        "pauc_tpr80_raw_pooled_oof": compute_paauc(y, scores, min_tpr=config.MIN_TPR),
        f"at_threshold_{FIXED_THRESHOLD}": threshold_metrics(y, scores, FIXED_THRESHOLD),
        "at_cross_validated_threshold": threshold_metrics(y, scores, cv_thr),
        "cross_validated_threshold_by_fold": cv_thr_by_fold,
    }


def collect_oof() -> pd.DataFrame:
    abcd_features, abcd_lookup = load_abcd_lookup()
    frames = []
    for fold in range(config.N_FOLDS):
        emb_dir = config.EMBEDDING_DIR
        if not (emb_dir / f"fold_{fold}_val_embeddings.npy").exists():
            raise FileNotFoundError(f"[fold {fold}] embeddings missing under {emb_dir} -- "
                                    f"run train_cnn.py then extract_embeddings.py first.")
        train_emb = np.load(emb_dir / f"fold_{fold}_train_embeddings.npy")
        train_labels = np.load(emb_dir / f"fold_{fold}_train_labels.npy")
        train_ids = np.load(emb_dir / f"fold_{fold}_train_isic_ids.npy", allow_pickle=True)
        val_emb = np.load(emb_dir / f"fold_{fold}_val_embeddings.npy")
        val_labels = np.load(emb_dir / f"fold_{fold}_val_labels.npy")
        val_ids = np.load(emb_dir / f"fold_{fold}_val_isic_ids.npy", allow_pickle=True)

        cnn, stored_paauc = cnn_head_scores(config.CHECKPOINT_DIR / f"fold_{fold}_best.pt", val_emb)
        recomputed = compute_paauc(val_labels, cnn, min_tpr=config.MIN_TPR)
        stored_text = f"{stored_paauc:.4f}" if stored_paauc is not None else "n/a"
        print(f"[fold {fold}] CNN val pAUC recomputed={recomputed:.4f} stored-in-checkpoint={stored_text}")
        if stored_paauc is not None and abs(recomputed - stored_paauc) > 0.005:
            raise ValueError(f"[fold {fold}] recomputed CNN pAUC {recomputed:.4f} differs from the checkpoint's "
                             f"{stored_paauc:.4f}; embeddings and checkpoint do not belong together.")

        train_abcd, train_valid = align_abcd_features(train_ids, abcd_features, abcd_lookup)
        val_abcd, val_valid = align_abcd_features(val_ids, abcd_features, abcd_lookup)
        y_train, y_val = train_labels[train_valid], val_labels[val_valid]

        fusion, source = fusion_scores(fold, train_emb[train_valid], train_abcd[train_valid], y_train,
                                       val_emb[val_valid], val_abcd[val_valid], y_val)
        abcd = abcd_only_scores(train_abcd[train_valid], y_train, val_abcd[val_valid], y_val)
        print(f"[fold {fold}] n_val={len(val_ids)} abcd_valid={int(val_valid.sum())} fusion: {source}")

        frame = pd.DataFrame({
            "isic_id": val_ids.astype(str),
            "fold": fold,
            "target": val_labels.astype(int),
            "abcd_valid": val_valid,
            "cnn": cnn,
            "abcd": np.nan,
            "fusion": np.nan,
        })
        frame.loc[val_valid, "abcd"] = abcd
        frame.loc[val_valid, "fusion"] = fusion
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def print_table(title: str, results: dict) -> None:
    print(f"\n=== {title} ===")
    any_result = next(iter(results.values()))
    print(f"n={any_result['n']}  malignant={any_result['n_malignant']}")
    rows = [("ROC-AUC (pooled OOF)", lambda r: r["roc_auc_pooled_oof"]),
            ("ROC-AUC (fold mean)", lambda r: r["roc_auc_per_fold_mean"]),
            ("pAUC TPR>=0.80 (raw)", lambda r: r["pauc_tpr80_raw_pooled_oof"])]
    for key, label in ((f"at_threshold_{FIXED_THRESHOLD}", f"@{FIXED_THRESHOLD}"),
                       ("at_cross_validated_threshold", "@cv-thr")):
        for metric in ("accuracy", "balanced_accuracy", "precision", "recall_sensitivity", "specificity", "f1"):
            rows.append((f"{metric} {label}", lambda r, k=key, m=metric: r[k][m]))
    print(f"{'metric':32s}" + "".join(f"{name:>10s}" for name in results))
    for label, get in rows:
        print(f"{label:32s}" + "".join(f"{get(r):10.4f}" for r in results.values()))
    for key, label in ((f"at_threshold_{FIXED_THRESHOLD}", f"@{FIXED_THRESHOLD}"),
                       ("at_cross_validated_threshold", "@cv-thr")):
        for name, r in results.items():
            m = r[key]
            print(f"confusion {label} {name}: TP={m['TP']} TN={m['TN']} FP={m['FP']} FN={m['FN']}")


def main() -> None:
    config.ensure_dirs()
    seed_everything(config.SEED)

    oof = collect_oof()
    oof.to_csv(config.OUTPUT_DIR / "oof_predictions.csv", index=False)

    # Main comparison: rows where all three models have a score (ABCD
    # features extracted), so every model is scored on identical images.
    common = oof[oof["abcd_valid"]]
    y, folds = common["target"].to_numpy(), common["fold"].to_numpy()
    comparison = {name: evaluate(y, common[name].to_numpy(), folds) for name in MODELS}

    # CNN alone can also score images where YOLO found no lesion.
    cnn_all = {"cnn": evaluate(oof["target"].to_numpy(), oof["cnn"].to_numpy(), oof["fold"].to_numpy())}

    report = {"same_images_all_three_models": comparison, "cnn_on_all_images": cnn_all}
    with open(config.OUTPUT_DIR / "metrics_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print_table("Same images, all three models (rows with valid ABCD features)", comparison)
    print_table("CNN on every validation image (incl. no-lesion-detected)", cnn_all)
    print(f"\nSaved {config.OUTPUT_DIR / 'metrics_report.json'} and {config.OUTPUT_DIR / 'oof_predictions.csv'}")


if __name__ == "__main__":
    main()
