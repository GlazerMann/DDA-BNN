#!/usr/bin/env python3
"""Master analysis script that runs the full diagnostic suite on a single run.

Computes basic and extended metrics, generates z-histograms, calibration
curves, sharpness-vs-error plots, and calibrated prediction scatter plots.
All outputs are saved to ``<run_dir>/analysis_plots/``.

Example::

    python analysis/run_analysis.py
    python analysis/run_analysis.py --run_dir artifacts/20260329_134948/20260329_134949
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

# ── make sure the project root is on sys.path ──────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from analysis import analysis as ana
from analysis.plot_bounds import plot_results_sampling


# ── resolve run dir (reuse the same logic as inference_api) ─────────
def _find_latest_run_dir(root: Path | None = None) -> Path:
    root_path = root or (_PROJECT_ROOT / "artifacts")
    if not root_path.is_dir():
        raise FileNotFoundError(f"Artifact root '{root_path}' does not exist")

    candidates: list[Path] = []
    for child in sorted(root_path.iterdir(), reverse=True):
        if not child.is_dir():
            continue
        if (child / "model.pth").exists():
            candidates.append(child)
            continue
        for inner in sorted(child.iterdir(), reverse=True):
            if inner.is_dir() and (inner / "model.pth").exists():
                candidates.append(inner)
                break
    if not candidates:
        raise FileNotFoundError(
            f"No run directory with model.pth found under '{root_path}'"
        )
    return max(candidates, key=lambda p: p.name)


# ── main ────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Run full analysis suite on a single run.")
    ap.add_argument("--run_dir", default=None, type=str,
                    help="Path to run directory. Default: most recent run under artifacts/")
    ap.add_argument("--targets", nargs="*", default=["Qext", "SSA", "g"])
    args = ap.parse_args()

    # ── resolve run dir ──
    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
    else:
        run_dir = _find_latest_run_dir()
    print(f"[INFO] run_dir → {run_dir}")

    out_dir = run_dir / "analysis_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = args.targets
    pred_csv = run_dir / "predictions_test.csv"
    cal_csv  = run_dir / "predictions_test_cal.csv"

    # ── 1. Basic metrics table (from raw predictions) ────────────────
    if not pred_csv.exists():
        print(f"[WARN] {pred_csv.name} not found — skipping raw metrics")
    else:
        df = pd.read_csv(pred_csv)
        print(f"\n{'='*60}")
        print(f"  Raw predictions: {len(df)} samples, {len(df.columns)} columns")
        print(f"{'='*60}")

        print("\n── make_metrics_table (aleatoric-only, with HS baseline) ──")
        m1 = ana.make_metrics_table(df, targets)
        print(m1.to_string(float_format="%.4g"))

        print("\n── metrics_for_run (epi + total decomposition) ──")
        m2 = ana.metrics_for_run(df, targets)
        print(m2.to_string(float_format="%.4g"))

        # save metrics CSVs
        m1.to_csv(out_dir / "metrics_basic.csv")
        m2.to_csv(out_dir / "metrics_extended.csv", index=False)
        print(f"\n[INFO] saved metrics → {out_dir / 'metrics_basic.csv'}")
        print(f"[INFO] saved metrics → {out_dir / 'metrics_extended.csv'}")

        # ── 2. run_all plots (z-histograms, calibration curve, sharpness) ──
        print(f"\n── Generating plots ──")
        ana.run_all(df, save_dir=str(out_dir), targets=targets, show_metrics=False)

    # ── 3. Calibrated bounds plot (from predictions_test_cal.csv) ────
    if not cal_csv.exists():
        print(f"\n[WARN] {cal_csv.name} not found — skipping calibrated bounds plot")
    else:
        df_cal = pd.read_csv(cal_csv)
        print(f"\n{'='*60}")
        print(f"  Calibrated predictions: {len(df_cal)} samples")
        print(f"{'='*60}")

        # q50 as prediction, q05–q95 interval
        has_q50 = all(f"q50_{t}" in df_cal.columns for t in targets)
        has_interval = all(f"q05_{t}" in df_cal.columns and f"q95_{t}" in df_cal.columns
                           for t in targets)

        if has_q50 and has_interval:
            print("\n── plot_results_sampling (true vs q50, with 90% CI) ──")
            plot_results_sampling(
                df_cal,
                targets=targets,
                x_mode="true",
                pred_mode="q50",
                interval=("q05", "q95"),
                out_path=out_dir / "pred_vs_true_cal.png",
            )

            # also plot HS baseline comparison
            if all(f"{t}_HS" in df_cal.columns for t in targets):
                print("── plot_results_sampling (HS baseline vs q50) ──")
                plot_results_sampling(
                    df_cal,
                    targets=targets,
                    x_mode="hs",
                    pred_mode="q50",
                    interval=("q05", "q95"),
                    out_path=out_dir / "pred_vs_hs_cal.png",
                )
        else:
            print("[WARN] calibrated CSV missing q50/q05/q95 columns — skipping bounds plot")

        # ── 4. Calibrated metrics (using mean_* and std_tot_* from cal CSV) ──
        # Build a df compatible with metrics_for_run by mapping calibrated columns
        has_mean = all(f"mean_{t}" in df_cal.columns for t in targets)
        has_std  = all(f"std_tot_{t}" in df_cal.columns for t in targets)
        has_true = all(f"true_{t}" in df_cal.columns for t in targets)

        if has_mean and has_std and has_true:
            # Create a df with pred_* = mean_*, std_ale_* = std_tot_* for metrics
            df_m = df_cal.copy()
            for t in targets:
                df_m[f"pred_{t}"]    = df_m[f"mean_{t}"]
                df_m[f"std_ale_{t}"] = df_m.get(f"std_ale_{t}", df_m[f"std_tot_{t}"])
                df_m[f"std_epi_{t}"] = df_m.get(f"std_epi_{t}", pd.Series(0.0, index=df_m.index))

            print("\n── metrics_for_run (calibrated) ──")
            m_cal = ana.metrics_for_run(df_m, targets)
            print(m_cal.to_string(float_format="%.4g"))
            m_cal.to_csv(out_dir / "metrics_calibrated.csv", index=False)
            print(f"[INFO] saved → {out_dir / 'metrics_calibrated.csv'}")

    # ── summary ──
    print(f"\n{'='*60}")
    print(f"  All outputs saved to: {out_dir}")
    print(f"{'='*60}")
    for f in sorted(os.listdir(out_dir)):
        print(f"  • {f}")


if __name__ == "__main__":
    main()

