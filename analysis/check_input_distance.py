#!/usr/bin/env python3
"""Nearest-neighbour distance diagnostics for duplicate / leakage detection.

Loads a CSV (or Excel) dataset, builds a standardised feature matrix, and
computes pairwise nearest-neighbour distances to identify exact duplicates
and near-duplicates.  Optionally performs a random train/val/test split and
reports test-to-train nearest-neighbour distances.

Example::

    python analysis/check_input_distance.py --file data/DDA_dataset.csv --do_split --seed 0
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


DEFAULT_FEATURES = ["Npp", "V/V0", "coating_RI_imag", "Xve", "core_Df"]
DEFAULT_TARGETS  = ["Qext", "SSA", "g"]


def load_table(path: str, sheet: str | None = None) -> pd.DataFrame:
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path, sheet_name=sheet)
    return pd.read_csv(path)


def extract_imag_from_coating_RI(col: pd.Series) -> np.ndarray:
    """Parse imaginary part from coating_RI column (robust to a few formats)."""
    if np.issubdtype(col.dtype, np.number):
        return col.to_numpy(dtype=float)

    def _parse_one(v):
        if pd.isna(v):
            return np.nan
        if isinstance(v, complex):
            return float(v.imag)

        s = str(v).strip().lower()
        s = s.replace("i", "j").strip("()")
        try:
            return float(complex(s).imag)
        except Exception:
            # last resort for strings like "1.6+0.05j"
            if "+" in s and "j" in s:
                try:
                    return float(s.split("+")[-1].replace("j", ""))
                except Exception:
                    pass
            raise ValueError(f"Could not parse coating_RI value: {v}")

    return np.array([_parse_one(v) for v in col.values], dtype=float)


def build_X(df: pd.DataFrame,
            features: list[str],
            targets: list[str],
            include_hs: bool = True) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Returns:
      X: feature matrix after dropping NaNs (float ndarray)
      df_kept: df subset aligned with X, includes 'orig_row'
    """
    df2 = df.copy()

    # engineered feature Xve
    if "Xve" in features:
        if not {"Dve", "wavelength"}.issubset(df2.columns):
            raise KeyError("Need columns Dve and wavelength to compute Xve = pi*Dve/wavelength")
        df2["Xve"] = np.pi * df2["Dve"].astype(float) / df2["wavelength"].astype(float)

    cols = []
    for f in features:
        if f == "coating_RI_imag":
            if "coating_RI" not in df2.columns:
                raise KeyError("Feature coating_RI_imag requested but column coating_RI not found")
            cols.append(extract_imag_from_coating_RI(df2["coating_RI"]))
        else:
            if f not in df2.columns:
                raise KeyError(f"Feature column not found: {f}")
            cols.append(df2[f].to_numpy(dtype=float))

    if include_hs:
        for t in targets:
            hs_col = f"{t}_HS"
            if hs_col not in df2.columns:
                raise KeyError(f"include_hs=True but missing column: {hs_col}")
            cols.append(df2[hs_col].to_numpy(dtype=float))

    X = np.column_stack(cols)

    # drop rows with NaNs/infs and track original row ids
    mask = np.isfinite(X).all(axis=1)
    keep_idx = np.where(mask)[0]
    X = X[mask]

    df_kept = df2.iloc[keep_idx].copy()
    df_kept["orig_row"] = keep_idx  # original index within the loaded table
    df_kept.reset_index(drop=True, inplace=True)  # align with X row numbering

    return X, df_kept


def quantiles(x: np.ndarray, qs=(0, 0.01, 0.05, 0.1, 0.5, 0.9, 0.99, 1.0)) -> dict:
    qv = np.quantile(x, qs)
    return {q: float(v) for q, v in zip(qs, qv)}


def dataset_wide_nn(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (dist, ind) from kneighbors with n_neighbors=2 in standardized space."""
    Xs = StandardScaler().fit_transform(X)
    nn = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(Xs)
    dist, ind = nn.kneighbors(Xs)
    return dist, ind  # dist[:,0]=0 self; dist[:,1]=nearest other


def print_pairs(df_kept: pd.DataFrame,
                ind: np.ndarray,
                dist: np.ndarray,
                eps: float,
                max_pairs: int,
                title: str,
                cols_to_show: list[str]) -> None:
    """
    Print unique pairs (i, j) where the nearest-neighbor distance <= eps.

    Fixes:
      - skip bogus self-pairs (i == j)
      - avoid double-printing symmetric pairs
    """
    pairs = []
    n = len(df_kept)

    for i in range(n):
        j = int(ind[i, 1])
        d = float(dist[i, 1])

        # guard against self being returned as "nearest neighbor"
        if j == i:
            continue

        if d <= eps:
            a, b = (i, j) if i < j else (j, i)
            pairs.append((a, b, d))

    # unique + sort
    pairs = sorted(set(pairs), key=lambda x: (x[2], x[0], x[1]))

    print(f"\n{title}: {len(pairs)} pairs with distance <= {eps}")
    for k, (a, b, d) in enumerate(pairs[:max_pairs], start=1):
        ra = df_kept.iloc[a]
        rb = df_kept.iloc[b]
        print(
            f"\nPair {k}: kept_rows=({a}, {b}), dist={d:.6g}, "
            f"orig_rows=({int(ra['orig_row'])}, {int(rb['orig_row'])})"
        )
        for c in cols_to_show:
            if c in df_kept.columns:
                print(f"  {c:12s}  A={ra[c]}   B={rb[c]}")
            else:
                print(f"  {c:12s}  (missing)")

    if len(pairs) > max_pairs:
        print(f"\n... truncated to {max_pairs} pairs")

def split_indices(n: int, train_frac: float, val_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    idx_tr = idx[:n_train]
    idx_va = idx[n_train:n_train + n_val]
    idx_te = idx[n_train + n_val:]
    return idx_tr, idx_va, idx_te


def test_to_train_nn(X: np.ndarray, idx_tr: np.ndarray, idx_te: np.ndarray):
    """Compute test→train NN distances (train-fitted scaling) and nearest train indices."""
    scaler = StandardScaler().fit(X[idx_tr])
    Xtr = scaler.transform(X[idx_tr])
    Xte = scaler.transform(X[idx_te])

    nn = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(Xtr)
    dist, ind = nn.kneighbors(Xte)
    # ind is index into Xtr; map back to original kept-row indices
    nn_train_rows = idx_tr[ind[:, 0]]
    return dist[:, 0], nn_train_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="Path to CSV/XLSX dataset")
    ap.add_argument("--sheet", default=None, help="Excel sheet name (if using .xlsx)")
    ap.add_argument("--features", nargs="+", default=DEFAULT_FEATURES)
    ap.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    ap.add_argument("--no_hs", action="store_true", help="Exclude *_HS features from distance computation")
    ap.add_argument("--do_split", action="store_true", help="Also compute test→train NN distances for a random split")
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--near_thresh", type=float, default=1e-3, help="Near-duplicate threshold in standardized space")
    ap.add_argument("--dup_eps", type=float, default=0.0, help="Exact-duplicate threshold (use 0.0 for exact)")
    ap.add_argument("--print_top", type=int, default=20, help="Max number of pairs/rows to print")
    args = ap.parse_args()

    df = load_table(args.file, args.sheet)
    include_hs = not args.no_hs
    X, df_kept = build_X(df, args.features, args.targets, include_hs=include_hs)

    print(f"\nLoaded rows: {len(df)}")
    print(f"Rows used after dropping NaNs in X: {len(X)}")
    if include_hs:
        print(f"Features used: {args.features} + ({args.targets} HS)")
    else:
        print(f"Features used: {args.features} (no HS)")

    # --- dataset-wide NN ---
    dist, ind = dataset_wide_nn(X)
    d1 = dist[:, 1]

    print("\nDataset-wide nearest-neighbor distances (standardized feature space):")
    print("Quantiles:", quantiles(d1))
    print(f"Fraction with NN distance < {args.near_thresh}: {(d1 < args.near_thresh).mean():.4f}")

    cols_to_show = ["orig_row", "Npp", "V/V0", "core_Df", "Dve", "wavelength", "coating_RI"]
    print_pairs(df_kept, ind, dist, eps=args.dup_eps, max_pairs=args.print_top,
                title="Exact duplicates (distance <= dup_eps)", cols_to_show=cols_to_show)

    if args.near_thresh > args.dup_eps:
        print_pairs(df_kept, ind, dist, eps=args.near_thresh, max_pairs=args.print_top,
                    title=f"Near-duplicates (distance <= {args.near_thresh})", cols_to_show=cols_to_show)

    # --- split-specific ---
    if args.do_split:
        idx_tr, idx_va, idx_te = split_indices(len(X), args.train_frac, args.val_frac, args.seed)
        d_te, nn_train_rows = test_to_train_nn(X, idx_tr, idx_te)

        print("\nRandom-split test→train nearest-neighbor distances (train-only scaling):")
        print(f"Split sizes: train={len(idx_tr)}, val={len(idx_va)}, test={len(idx_te)}")
        print("Quantiles:", quantiles(d_te))
        print(f"Fraction test points with NN distance < {args.near_thresh}: {(d_te < args.near_thresh).mean():.4f}")

        # print the closest test points
        order = np.argsort(d_te)
        print(f"\nClosest test points (top {min(args.print_top, len(order))}):")
        for k in order[:args.print_top]:
            te_row = int(idx_te[k])
            tr_row = int(nn_train_rows[k])
            ra = df_kept.iloc[te_row]
            rb = df_kept.iloc[tr_row]
            print(f"\nTest kept_row={te_row} (orig_row={int(ra['orig_row'])}) "
                  f"nearest train kept_row={tr_row} (orig_row={int(rb['orig_row'])}), dist={d_te[k]:.6g}")
            for c in cols_to_show:
                if c in df_kept.columns:
                    print(f"  {c:12s}  test={ra[c]}   train={rb[c]}")

    print("\nDone.")


if __name__ == "__main__":
    main()