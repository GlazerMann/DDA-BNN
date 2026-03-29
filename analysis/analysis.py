"""Metrics and visualisation helpers for evaluating model output.

Provides point-estimate metrics (RMSE, MAE), probabilistic metrics
(NLL, coverage, sharpness), and diagnostic plots (z-histograms,
calibration curves, sharpness-vs-error).  ``run_all`` is a convenience
wrapper that generates everything in one call.

Example::

    import analysis as ana

    df = pd.read_csv("predictions_test.csv")
    metrics = ana.run_all(df, save_dir="figures")
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

DEFAULT_TARGETS = ["Qext", "SSA", "g"]


# ------------------------------------------------------------------ #
# -----------------------  basic metrics ---------------------------- #
# ------------------------------------------------------------------ #
def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a, b): return float(np.mean(np.abs(a - b)))


def nll_gauss(y, mu, sigma):
    var = sigma ** 2
    return float(np.mean(0.5 * np.log(2 * np.pi * var) +
                         0.5 * ((y - mu) ** 2) / var))


def coverage(y, mu, sigma, z):
    """Empirical P(|y-mu| <= z * sigma)."""
    return float((np.abs(y - mu) <= z * sigma).mean())


# ------------------------------------------------------------------ #
# -------------------  1. Metrics table ----------------------------- #
# ------------------------------------------------------------------ #
def make_metrics_table(df: pd.DataFrame,
                       targets: list[str],
                       tf_eps: float = 1e-12) -> pd.DataFrame:
    """
    Compute point-estimate and (if available) probabilistic metrics.

    Returns
    -------
    pandas.DataFrame  indexed by target with columns
        RMSE_HS, RMSE_pred, MAE_HS, MAE_pred,
        [NLL, sharpness, z_var, cov_68, cov_90, cov_95]
    """
    rows = []
    for t in targets:
        y_true = df[f"true_{t}"].values
        y_hs = df[f"{t}_HS"].values
        mu = df[f"pred_{t}"].values

        row = dict(
            target=t,
            RMSE_HS=rmse(y_true, y_hs),
            RMSE_pred=rmse(y_true, mu),
            MAE_HS=mae(y_true, y_hs),
            MAE_pred=mae(y_true, mu)
        )

        std_col = f"std_ale_{t}"
        if std_col in df.columns:
            sigma = df[std_col].values + tf_eps
            row.update(
                NLL=nll_gauss(y_true, mu, sigma),
                sharpness=float(np.mean(sigma)),
                z_var=float(np.mean(((y_true - mu) / sigma) ** 2)),
                cov_68=coverage(y_true, mu, sigma, 1.0),
                cov_90=coverage(y_true, mu, sigma, norm.ppf(0.95)),
                cov_95=coverage(y_true, mu, sigma, norm.ppf(0.975)),
            )
        rows.append(row)

    return pd.DataFrame(rows).set_index("target")


# ------------------------------------------------------------------ #
# --------- 1b. Extended metrics (epi + total decomposition) ------- #
# ------------------------------------------------------------------ #
def metrics_for_run(df: pd.DataFrame,
                    targets: list[str] = ("Qext", "SSA", "g"),
                    tf_eps: float = 0.0) -> pd.DataFrame:
    """
    Like make_metrics_table but decomposes uncertainty into
    aleatoric / epistemic / total components.

    Returns a tidy DataFrame with one row per target (not indexed).
    """
    rows = []
    for t in targets:
        col_true = f"true_{t}"
        col_pred = f"pred_{t}"
        if col_true not in df.columns:
            continue
        y = df[col_true].values
        mu = df[col_pred].values
        sig_a = df.get(f"std_ale_{t}", pd.Series(tf_eps, index=df.index)).values + tf_eps
        sig_e = df.get(f"std_epi_{t}", pd.Series(0.0, index=df.index)).values
        sig_t = np.sqrt(sig_a ** 2 + sig_e ** 2)

        rows.append(dict(
            target=t,
            RMSE_pred=rmse(y, mu),
            MAE_pred=mae(y, mu),
            NLL_ale=nll_gauss(y, mu, sig_a),
            NLL_total=nll_gauss(y, mu, sig_t),
            sharp_ale=float(np.mean(sig_a)),
            sharp_epi=float(np.mean(sig_e)),
            sharp_total=float(np.mean(sig_t)),
            z_var_ale=float(np.mean(((y - mu) / sig_a) ** 2)),
            z_var_tot=float(np.mean(((y - mu) / sig_t) ** 2)),
            cov68_tot=coverage(y, mu, sig_t, 1.0),
            cov90_tot=coverage(y, mu, sig_t, norm.ppf(0.95)),
            cov95_tot=coverage(y, mu, sig_t, norm.ppf(0.975)),
            frac_epi=float(np.mean(sig_e ** 2) / np.mean(sig_t ** 2 + 1e-12)),
        ))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ #
# --------------- 2. Histogram of normalised residuals -------------- #
# ------------------------------------------------------------------ #
def plot_z_histograms(df: pd.DataFrame,
                      targets: list[str],
                      save_dir: Optional[Path] = None) -> None:
    """
    For every target plot histogram of z = (y_true - mu)/sigma.

    If `save_dir` is given the PNGs are stored there, otherwise shown inline.
    """
    save_dir = Path(save_dir) if save_dir is not None else None
    for t in targets:
        std_col = f"std_ale_{t}"
        if std_col not in df.columns:
            continue

        z = (df[f"true_{t}"] - df[f"pred_{t}"]) / df[std_col]
        plt.figure(figsize=(4, 3))
        plt.hist(z, bins=40, density=True, alpha=0.6, label="z histogram")
        xs = np.linspace(-4, 4, 200)
        plt.plot(xs, norm.pdf(xs), 'r', lw=2, label="N(0,1)")
        plt.title(f"{t}: normalised residuals")
        plt.xlabel("z")
        plt.ylabel("density")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if save_dir is None:
            plt.show()
        else:
            out = save_dir / f"z_hist_{t}.png"
            plt.savefig(out, bbox_inches="tight")
            print(f"saved {out}")
            plt.close()


# ------------------------------------------------------------------ #
# ------------------- 3. Calibration curves ------------------------- #
# ------------------------------------------------------------------ #
def plot_calibration_curves(df: pd.DataFrame,
                            targets: list[str],
                            save_dir: Optional[Path] = None) -> None:
    """Plot nominal vs empirical coverage curves."""
    qs = np.linspace(0.05, 0.95, 19)

    plt.figure(figsize=(4, 4))
    plt.plot([0, 1], [0, 1], 'k--', label="ideal")

    for t in targets:
        std_col = f"std_ale_{t}"
        if std_col not in df.columns:
            continue
        y = df[f"true_{t}"].values
        mu = df[f"pred_{t}"].values
        sig = df[std_col].values

        cov = _empirical_coverage(y, mu, sig, qs)
        plt.plot(qs, cov, marker='o', label=t)

    plt.xlabel("Nominal coverage probability")
    plt.ylabel("Empirical coverage")
    plt.title("Calibration curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_dir is None:
        plt.show()
    else:
        out = Path(save_dir) / "calibration_curve.png"
        plt.savefig(out, bbox_inches="tight")
        print(f"saved {out}")
        plt.close()


def _empirical_coverage(y, mu, sigma, qs):
    """Compute empirical coverage at each nominal level in *qs*."""
    z = norm.ppf(0.5 + np.asarray(qs) / 2)
    residuals = np.abs(y - mu)
    return np.array([float((residuals <= zi * sigma).mean()) for zi in z])


# ------------------------------------------------------------------ #
# ---------------- 4. Sharpness vs squared error -------------------- #
# ------------------------------------------------------------------ #
def plot_sharpness_vs_error(df: pd.DataFrame,
                            targets: list[str],
                            save_dir: Optional[Path] = None,
                            quant: float = 0.99,
                            log: bool = True):
    """Scatter of predicted variance vs squared error."""
    for t in targets:
        std_col = f"std_ale_{t}"
        if std_col not in df.columns:
            continue

        err2 = (df[f"true_{t}"] - df[f"pred_{t}"]) ** 2
        sig2 = df[std_col] ** 2

        # -------- axis limits (percentile clipping) ----------------
        x_max = np.quantile(sig2, quant)
        y_max = np.quantile(err2, quant)

        plt.figure(figsize=(4, 3))
        plt.scatter(sig2, err2, alpha=0.5)

        # reference y = x line (ideal calibration)
        xs = np.linspace(0, x_max, 200)
        plt.plot(xs, xs, 'k--', lw=1)

        if log:
            plt.xscale('symlog', linthresh=1e-6)  # avoids log(0)
            plt.yscale('symlog', linthresh=1e-6)

        plt.xlim(0, x_max)
        plt.ylim(0, y_max)

        plt.xlabel("Predicted variance σ²")
        plt.ylabel("Squared error")
        plt.title(f"{t}: sharpness vs error")
        plt.grid(True)
        plt.tight_layout()

        if save_dir is None:
            plt.show()
        else:
            out = Path(save_dir) / f"sharp_vs_err_{t}.png"
            plt.savefig(out, bbox_inches="tight")
            print(f"saved {out}")
            plt.close()


# ------------------------------------------------------------------ #
# ------------------- 5. Convenience wrapper ------------------------ #
# ------------------------------------------------------------------ #
def run_all(df: pd.DataFrame,
            save_dir: str | Path | None = None,
            targets: list[str] | None = None,
            show_metrics: bool = True) -> pd.DataFrame:
    """Run all analyses (metrics table + 3 plot groups).

    Parameters
    ----------
    df           : DataFrame with columns true_<T>, <T>_HS, pred_<T>,
                   [std_ale_<T>] for every T in *targets*.
    save_dir     : If given, figures are saved there; otherwise shown inline.
    targets      : Target names.  Default ``DEFAULT_TARGETS``.
    show_metrics : If True, display the metrics DataFrame.

    Returns
    -------
    pandas.DataFrame with metrics.
    """
    if targets is None:
        targets = list(DEFAULT_TARGETS)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    # 1) metrics table
    metrics = make_metrics_table(df, targets)
    if show_metrics:
        try:
            from IPython.display import display
            display(metrics.style.format("{:.3g}"))
        except Exception:  # outside notebook
            print(metrics.to_string(float_format="%.3g"))

    # 2) histograms
    plot_z_histograms(df, targets, save_dir)

    # 3) calibration curve
    plot_calibration_curves(df, targets, save_dir)

    # 4) sharpness vs error
    plot_sharpness_vs_error(df, targets, save_dir)

    return metrics
