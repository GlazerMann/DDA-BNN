# analysis/

Post-training analysis and diagnostics for aerosol BNN runs.

All scripts are run from the **project root** (`aerosol_bnn/`).

## Scripts

| Script | Purpose | Usage |
|---|---|---|
| `run_analysis.py` | **Master script** — runs the full analysis suite on a single run | `python analysis/run_analysis.py --run_dir artifacts/20260329_134948/20260329_134949` |
| `analysis.py` | Core metrics library — RMSE, MAE, NLL, coverage, z-histograms, calibration curves | Imported by other scripts; also usable via `ana.run_all(df, save_dir)` |
| `analysis_out_bulk.py` | Sweep-level analysis — harvest metrics from all runs, aggregate, rank | `python analysis/analysis_out_bulk.py --root artifacts/20260329_134948` |
| `plot_bounds.py` | Prediction-vs-truth scatter with confidence interval bars | `python analysis/plot_bounds.py --csv artifacts/20260329_134948/20260329_134949/predictions_test_cal.csv` |
| `check_input_distance.py` | Data QA — nearest-neighbour distance diagnostics | `python analysis/check_input_distance.py --file data/DDA_dataset.csv` |

## Quick start

> **Note:** The example artifact `20260329_134948` is a small 5-epoch test run
> included for demonstration purposes. Metrics and plots from it are not
> representative of a fully trained model.

```bash
# Full analysis on the most recent run (auto-detected)
python analysis/run_analysis.py

# Full analysis on a specific run
python analysis/run_analysis.py --run_dir artifacts/20260329_134948/20260329_134949

# Sweep-level comparison across all runs
python analysis/analysis_out_bulk.py --root artifacts/20260329_134948

# Standalone calibrated bounds plot
python analysis/plot_bounds.py --csv artifacts/20260329_134948/20260329_134949/predictions_test_cal.csv

# Dataset duplicate/leakage check
python analysis/check_input_distance.py --file data/DDA_dataset.csv --do_split
```

## Output

`run_analysis.py` saves everything to `<run_dir>/analysis_plots/`:

**Metrics (CSV)**
- `metrics_basic.csv` — RMSE/MAE for HS baseline and BNN, aleatoric NLL, coverage
- `metrics_extended.csv` — adds epistemic/total decomposition, `frac_epi`
- `metrics_calibrated.csv` — same metrics recomputed on tau-calibrated predictions

**Plots (PNG)**
- `z_hist_{Qext,SSA,g}.png` — normalised residual histograms vs N(0,1)
- `calibration_curve.png` — nominal vs empirical coverage
- `sharp_vs_err_{Qext,SSA,g}.png` — predicted variance vs squared error
- `pred_vs_true_cal.png` — true vs q50 prediction with 90% CI bars
- `pred_vs_hs_cal.png` — HS baseline vs q50 prediction with 90% CI bars

