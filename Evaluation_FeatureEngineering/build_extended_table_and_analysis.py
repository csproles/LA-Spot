"""
Merge the new candidate features with V4's existing base feature table,
then produce:
  - extended_feature_table.csv (image_id, group_id, ground_truth_binary,
    V4's 6 base features, all new candidate features)
  - feature_analysis_table.csv (per-feature: benign/melanoma medians, effect
    size, univariate ROC-AUC, missing %)
  - correlation_matrix.csv and redundancy_report.md (near-duplicate /
    near-zero-variance / highly-correlated / high-missingness flags)

Read-only with respect to V4 and its existing feature table/OOF results.
"""

import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

BENCH_DIR = Path(__file__).resolve().parent
ROOT = BENCH_DIR.parent
V4_FEATURE_TABLE = ROOT / "Evaluation_V4" / "FeatureTable" / "development_feature_table.csv"
RAW_CANDIDATES = BENCH_DIR / "candidate_features_raw.csv"

BASE_FEATURES = ["A_value", "B_circularity", "C_value", "D_px", "confidence", "lesion_fraction"]


def to_float(x):
    if x in (None, "", "nan"):
        return np.nan
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def load_base():
    with open(V4_FEATURE_TABLE, newline="", encoding="utf-8") as f:
        return {r["image_id"]: r for r in csv.DictReader(f)}


def load_candidates():
    with open(RAW_CANDIDATES, newline="", encoding="utf-8") as f:
        return {r["image_id"]: r for r in csv.DictReader(f)}


def build_extended_table():
    base = load_base()
    cand = load_candidates()

    missing_in_cand = set(base.keys()) - set(cand.keys())
    missing_in_base = set(cand.keys()) - set(base.keys())
    if missing_in_cand:
        print(f"WARNING: {len(missing_in_cand)} base-feature-table images have no candidate-feature row "
              f"(sample: {sorted(missing_in_cand)[:5]})")
    if missing_in_base:
        print(f"WARNING: {len(missing_in_base)} candidate-feature rows have no base-feature-table match "
              f"(sample: {sorted(missing_in_base)[:5]})")

    common_ids = sorted(set(base.keys()) & set(cand.keys()))
    print(f"Base feature table: {len(base)} rows. Candidate features: {len(cand)} rows. "
          f"Merged on {len(common_ids)} common image_ids.")

    new_feature_cols = [k for k in next(iter(cand.values())).keys() if k not in ("image_id", "_errors")]

    rows = []
    for iid in common_ids:
        b = base[iid]
        c = cand[iid]
        row = {
            "image_id": iid,
            "group_id": b["group_id"],
            "ground_truth_binary": int(b["ground_truth_binary"]),
        }
        for f in BASE_FEATURES:
            row[f] = to_float(b[f])
        for f in new_feature_cols:
            row[f] = to_float(c[f])
        row["_errors"] = c.get("_errors", "")
        rows.append(row)

    fieldnames = ["image_id", "group_id", "ground_truth_binary"] + BASE_FEATURES + new_feature_cols + ["_errors"]
    with open(BENCH_DIR / "extended_feature_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote extended_feature_table.csv ({len(rows)} rows, {len(fieldnames)-4} feature columns)")
    return rows, BASE_FEATURES + new_feature_cols


def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    pooled_std = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if pooled_std < 1e-12:
        return np.nan
    return float((a.mean() - b.mean()) / pooled_std)


def feature_analysis(rows, feature_cols):
    y = np.array([r["ground_truth_binary"] for r in rows])
    out = []
    for f in feature_cols:
        vals = np.array([r[f] for r in rows], dtype=float)
        missing_pct = 100.0 * np.isnan(vals).sum() / len(vals)
        valid = ~np.isnan(vals)
        v, yv = vals[valid], y[valid]
        benign = v[yv == 0]
        malignant = v[yv == 1]
        if len(benign) < 2 or len(malignant) < 2:
            out.append({"feature": f, "benign_median": np.nan, "melanoma_median": np.nan,
                        "cohens_d": np.nan, "univariate_auc": np.nan, "missing_pct": round(missing_pct, 2)})
            continue
        d = cohens_d(malignant, benign)
        try:
            auc = roc_auc_score(yv, v)
        except ValueError:
            auc = np.nan
        out.append({
            "feature": f,
            "benign_median": round(float(np.median(benign)), 4),
            "melanoma_median": round(float(np.median(malignant)), 4),
            "cohens_d": round(d, 4) if d == d else np.nan,
            "univariate_auc": round(float(auc), 4) if auc == auc else np.nan,
            "missing_pct": round(missing_pct, 2),
        })

    out.sort(key=lambda r: -abs(r["univariate_auc"] - 0.5) if r["univariate_auc"] == r["univariate_auc"] else 0)
    with open(BENCH_DIR / "feature_analysis_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["feature", "benign_median", "melanoma_median", "cohens_d",
                                           "univariate_auc", "missing_pct"])
        w.writeheader()
        for r in out:
            w.writerow(r)
    print(f"Wrote feature_analysis_table.csv ({len(out)} features)")
    return out


def correlation_and_redundancy(rows, feature_cols):
    y = np.array([r["ground_truth_binary"] for r in rows])
    mat = np.array([[r[f] for f in feature_cols] for r in rows], dtype=float)

    # near-zero-variance / high-missingness flags computed on valid values only
    nzv_flags = {}
    missing_flags = {}
    variances = {}
    for j, f in enumerate(feature_cols):
        col = mat[:, j]
        valid = col[~np.isnan(col)]
        missing_pct = 100.0 * (len(col) - len(valid)) / len(col)
        missing_flags[f] = missing_pct
        if len(valid) > 1 and np.abs(valid.mean()) > 1e-9:
            cv = np.std(valid) / (np.abs(np.mean(valid)) + 1e-9)
        else:
            cv = np.std(valid) if len(valid) > 1 else 0.0
        variances[f] = float(np.std(valid)) if len(valid) > 1 else 0.0
        nzv_flags[f] = bool(len(valid) > 1 and np.std(valid) < 1e-6)

    # pairwise Spearman correlation (rank-based, robust to nonlinear monotonic
    # relationships like circularity vs isoperimetric_ratio)
    from scipy.stats import spearmanr
    n = len(feature_cols)
    corr = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i, n):
            a, b = mat[:, i], mat[:, j]
            valid = ~np.isnan(a) & ~np.isnan(b)
            if valid.sum() < 30:
                continue
            rho, _p = spearmanr(a[valid], b[valid])
            corr[i, j] = corr[j, i] = rho

    with open(BENCH_DIR / "correlation_matrix.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([""] + feature_cols)
        for i, f_i in enumerate(feature_cols):
            w.writerow([f_i] + [round(corr[i, j], 3) if corr[i, j] == corr[i, j] else "" for j in range(n)])

    # redundancy: |spearman| > 0.90 with a different, earlier-listed feature
    redundant_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if corr[i, j] == corr[i, j] and abs(corr[i, j]) > 0.90:
                redundant_pairs.append((feature_cols[i], feature_cols[j], round(float(corr[i, j]), 3)))

    lines = ["# Redundancy / quality-control report\n"]
    lines.append("## Near-zero-variance features (std ~ 0, flagged, not usable)\n")
    nzv = [f for f, v in nzv_flags.items() if v]
    lines.append(", ".join(nzv) if nzv else "(none)")
    lines.append("\n\n## High-missingness features (>5% NaN)\n")
    high_miss = [(f, round(v, 1)) for f, v in missing_flags.items() if v > 5.0]
    for f, v in sorted(high_miss, key=lambda x: -x[1]):
        lines.append(f"- {f}: {v}% missing")
    if not high_miss:
        lines.append("(none)")
    lines.append("\n\n## Highly redundant pairs (|Spearman rho| > 0.90)\n")
    for a, b, r in sorted(redundant_pairs, key=lambda x: -abs(x[2])):
        lines.append(f"- {a} <-> {b}: rho={r}")
    if not redundant_pairs:
        lines.append("(none)")

    with open(BENCH_DIR / "redundancy_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote correlation_matrix.csv and redundancy_report.md "
          f"({len(redundant_pairs)} redundant pairs, {len(nzv)} near-zero-variance, {len(high_miss)} high-missing)")
    return corr, redundant_pairs, nzv, missing_flags


def main():
    rows, feature_cols = build_extended_table()
    feature_analysis(rows, feature_cols)
    correlation_and_redundancy(rows, feature_cols)


if __name__ == "__main__":
    main()
