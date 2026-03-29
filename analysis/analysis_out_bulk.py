#!/usr/bin/env python3
"""Bulk metric collection and ranking across all runs in a sweep.

Iterates over every sub-directory under a sweep root, computes extended
metrics (epi/total decomposition, coverage, calibration error), and
writes per-run and aggregated CSVs.  Identifies the best run by a
weighted accuracy-vs-calibration score.

Example::

    python analysis/analysis_out_bulk.py --root artifacts/20260329_134948
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis import analysis as ana


# -------------------------------------------------------------------------
def load_predictions(path: Path) -> pd.DataFrame:
    """Load a predictions CSV (or parquet) with clean column names."""
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in {".csv", ".gz"}:
        df = pd.read_csv(path, skipinitialspace=True)
        df.columns = df.columns.str.strip()
        return df
    raise ValueError(f"Unsupported file format: {path}")


def metrics_per_run(run_dir: Path, targets: list[str] | None = None) -> pd.DataFrame | None:
    """Compute extended metrics for a single run, or None if no predictions file."""
    pred_file = next(
        (run_dir / p for p in ("predictions.parquet", "predictions.csv", "predictions_test.csv")
         if (run_dir / p).exists()),
        None,
    )
    if pred_file is None:
        return None
    df = load_predictions(pred_file)
    mtable = ana.metrics_for_run(df, targets=targets)
    mtable.insert(0, "run", run_dir.name)
    return mtable


# -------------------------------------------------------------------------
def collect_all(root: Path, targets: list[str] | None = None) -> pd.DataFrame:
    """Collect extended metrics from every run sub-directory under *root*."""
    rows = []
    for rd in sorted(p for p in root.iterdir() if p.is_dir()):
        m = metrics_per_run(rd, targets)
        if m is not None:
            rows.append(m)
        else:
            print(f"[skip] {rd.name}: no prediction file")
    if not rows:
        sys.exit("No runs found.")
    return pd.concat(rows, ignore_index=True)


# -------------------------------------------------------------------------
def plot_bar(df: pd.DataFrame, metric: str, save: Path):
    """One bar per run showing the chosen metric averaged over targets."""
    d = df.groupby("run")[metric].mean().sort_values()
    plt.figure(figsize=(max(4, 0.25 * len(d)), 3))
    d.plot(kind="barh")
    plt.xlabel(metric)
    plt.tight_layout()
    plt.savefig(save)
    plt.close()
    print(f"saved {save}")


def pareto_front(df: pd.DataFrame, x: str, y: str, save: Path):
    """Scatter plot with Pareto front highlighted (lower is better on both axes)."""
    g = df.groupby("run")[[x, y]].mean()
    xs, ys = g[x].values, g[y].values

    is_efficient = np.ones(len(xs), dtype=bool)
    for i, (cx, cy) in enumerate(zip(xs, ys)):
        if is_efficient[i]:
            is_efficient[is_efficient] = (xs[is_efficient] < cx) | (ys[is_efficient] < cy)
            is_efficient[i] = True
    front = g[is_efficient]

    plt.figure(figsize=(4, 4))
    plt.scatter(xs, ys, c="grey", alpha=0.5)
    plt.scatter(front[x], front[y], c="red")
    plt.plot(front[x].sort_values(), front[y].loc[front[x].sort_values().index],
             c="red", lw=0.8)
    for run, row in front.iterrows():
        plt.annotate(run, (row[x], row[y]), fontsize=7)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    plt.savefig(save)
    plt.close()
    print(f"saved {save}")


# -------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Collect metrics from all runs in a sweep.")
    ap.add_argument("--root", type=Path, required=True,
                    help="Sweep directory containing run sub-folders")
    ap.add_argument("--targets", nargs="*", default=["Qext", "SSA", "g"])
    args = ap.parse_args()

    out_dir = args.root
    out_dir.mkdir(exist_ok=True)

    df = collect_all(args.root, targets=args.targets)
    csv_file = out_dir / "all_metrics.csv"
    df.to_csv(csv_file, index=False)
    print(f"Wrote {csv_file}  (shape={df.shape})")

    # Aggregate over targets per run
    agg = (df
           .groupby("run")
           .agg(RMSE_mean=("RMSE_pred", "mean"),
                MAE_mean=("MAE_pred", "mean"),
                NLL_mean=("NLL_total", "mean"),
                sharp_mean=("sharp_total", "mean"),
                zvar_mean=("z_var_tot", "mean"),
                cov68_mean=("cov68_tot", "mean"),
                cov90_mean=("cov90_tot", "mean"),
                cov95_mean=("cov95_tot", "mean"),
                frac_epi_m=("frac_epi", "mean"))
           .reset_index())

    # Calibration error (distance of empirical coverage from nominal)
    agg["cal_err"] = np.sqrt((agg["cov68_mean"] - 0.68) ** 2 +
                             (agg["cov90_mean"] - 0.90) ** 2 +
                             (agg["cov95_mean"] - 0.95) ** 2)

    # Combined score (accuracy vs calibration — adjust weights as needed)
    alpha, beta = 0.7, 0.3
    agg["score"] = (alpha * agg["RMSE_mean"] / agg["RMSE_mean"].mean() +
                    beta * agg["cal_err"] / agg["cal_err"].mean())

    # Write run-level CSV
    run_csv = out_dir / "run_metrics.csv"
    agg.to_csv(run_csv, index=False)
    print(f"wrote {run_csv}  (shape={agg.shape})")

    # Best run
    best_run = agg.loc[agg["score"].idxmin(), "run"]
    print(f"\n*** Best run by weighted score = {best_run} ***")
