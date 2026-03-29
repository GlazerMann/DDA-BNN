#!/usr/bin/env python3
"""Prediction-vs-reference scatter plots with confidence interval bars.

Generates per-target scatter plots comparing predictions (q50, pred_map,
or pred_mean) against truth or the HS baseline, with optional vertical
bars showing the credible interval (e.g. 5th–95th quantile).

Example::

    python analysis/plot_bounds.py --csv artifacts/20260329_134948/20260329_134949/predictions_test_cal.csv
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_results_sampling(df: pd.DataFrame,
                          targets: tuple[str, ...] | list[str] = ("Qext", "SSA", "g"),
                          x_mode: str = "true",
                          pred_mode: str = "q50",
                          interval: tuple[str, str] = ("q05", "q95"),
                          out_path: Path = Path("pred_plot_sampling.png"),
                          max_bars: int = 250):
    """Scatter plot of predictions vs reference with optional CI bars.

    Parameters
    ----------
    x_mode    : ``"true"`` (DDA truth) or ``"hs"`` (low-res baseline).
    pred_mode : ``"q50"``, ``"pred_map"``, or ``"pred_mean"``.
    interval  : Pair of quantile column prefixes, e.g. ``("q05", "q95")``.
    """

    n = len(targets)
    fig, axes = plt.subplots(n, 1, figsize=(7, 4.5 * n), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, t in zip(axes, targets):
        # x-axis choice
        if x_mode == "true":
            x = df[f"true_{t}"].to_numpy()
            xlab = f"True {t} (DDA)"
        elif x_mode == "hs":
            x = df[f"{t}_HS"].to_numpy()
            xlab = f"{t}_HS (baseline)"
        else:
            raise ValueError("x_mode must be 'true' or 'hs'")

        # baseline (red)
        y_base = df[f"{t}_HS"].to_numpy()
        ax.scatter(x, y_base, s=18, alpha=0.5, color="tab:red", label="Low-res (HS)")

        # prediction (green)
        if pred_mode == "q50":
            y_pred = df[f"q50_{t}"].to_numpy()
            pred_label = "Pred (q50)"
        elif pred_mode == "pred_map":
            y_pred = df[f"pred_map_{t}"].to_numpy()
            pred_label = "Pred (map f^{-1}(E[mu_lat]))"
        elif pred_mode == "pred_mean":
            y_pred = df[f"pred_mean_{t}"].to_numpy()
            pred_label = "Pred (mean of samples)"
        else:
            raise ValueError("pred_mode must be 'q50', 'pred_map', or 'pred_mean'")

        ax.scatter(x, y_pred, s=20, alpha=0.8, color="tab:green", label=pred_label)

        # uncertainty bars
        lo_tag, hi_tag = interval
        lo_col = f"{lo_tag}_{t}"
        hi_col = f"{hi_tag}_{t}"
        if lo_col in df.columns and hi_col in df.columns:
            lo = df[lo_col].to_numpy()
            hi = df[hi_col].to_numpy()
            ok = np.isfinite(x) & np.isfinite(lo) & np.isfinite(hi)
            idx = np.where(ok)[0]
            if len(idx) > 0:
                step = max(1, len(idx) // max_bars)
                sel = idx[::step]
                ax.vlines(x[sel], lo[sel], hi[sel],
                          color="tab:green", alpha=0.25, linewidth=1)

        # y=x line (use combined range from x and y_pred)
        mn = np.nanmin(np.r_[x, y_pred, y_base])
        mx = np.nanmax(np.r_[x, y_pred, y_base])
        ax.plot([mn, mx], [mn, mx], "b--", lw=1)

        ax.set_title(t)
        ax.set_xlabel(xlab)
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] saved → {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Prediction scatter with CI bars.")
    ap.add_argument("--csv", required=True, type=str,
                    help="Predictions CSV (e.g. predictions_test_cal.csv)")
    ap.add_argument("--out", default=None, type=str)
    ap.add_argument("--x_mode", choices=["true", "hs"], default="true")
    ap.add_argument("--pred_mode", choices=["q50", "pred_map", "pred_mean"], default="q50")
    ap.add_argument("--interval", default="q05,q95",
                    help="Comma-separated quantile column prefixes, e.g. 'q05,q95'")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out) if args.out else (csv_path.parent / f"pred_plot_{args.x_mode}_{args.pred_mode}.png")

    df = pd.read_csv(csv_path)

    lo, hi = args.interval.split(",")
    plot_results_sampling(
        df,
        targets=["Qext", "SSA", "g"],
        x_mode=args.x_mode,
        pred_mode=args.pred_mode,
        interval=(lo, hi),
        out_path=out_path
    )


if __name__ == "__main__":
    main()