#!/usr/bin/env python3
"""Temperature calibration (tau fitting) for BNN uncertainty estimates.

Fits a per-target scaling factor tau on the validation set so that the
model's predictive intervals achieve nominal coverage.  Writes
``taus.json`` and ``predictions_test_cal.csv`` into the run directory.

Called automatically at the end of ``train.py``, or standalone::

    python training/temp_tune.py                    # auto-detect latest run
    python training/temp_tune.py --run_dir <path>
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import numpy as np
import pandas as pd
import yaml
from scipy.special import expit
from scipy.optimize import minimize_scalar
import training.config as cfg

# ---------------- config helper ----------------
def apply_config_used(cfg_module, cfg_dict: dict):
    for k, v in cfg_dict.items():
        if k.isupper() and hasattr(cfg_module, k):
            setattr(cfg_module, k, v)

# ---------------- transforms ----------------
def logit_np(y, eps):
    y = np.clip(y, eps, 1 - eps)
    return np.log(y / (1 - y))

def forward_phys_tf(y_phys: np.ndarray, tf_info, tf_eps: float):
    """physical -> latent"""
    out = y_phys.copy()
    idx = {c: i for i, c in enumerate(cfg.TARGETS)}
    for col, tf in tf_info.items():
        j = idx[col]
        if tf == "log":
            out[:, j] = np.log(out[:, j] + tf_eps)
        elif tf == "logit":
            out[:, j] = logit_np(out[:, j], tf_eps)
        elif tf == "none":
            pass
        else:
            raise ValueError(f"Unknown tf '{tf}' for target '{col}'")
    return out

def inverse_phys_tf(t_lat: np.ndarray, tf_info, tf_eps: float):
    """latent -> physical"""
    out = t_lat.copy()
    idx = {c: i for i, c in enumerate(cfg.TARGETS)}
    for col, tf in tf_info.items():
        j = idx[col]
        if tf == "log":
            z = np.clip(out[:, j], -50.0, 80.0)
            out[:, j] = np.exp(z) - tf_eps
        elif tf == "logit":
            z = np.clip(out[:, j], -30.0, 30.0)
            out[:, j] = expit(z)
        elif tf == "none":
            pass
        else:
            raise ValueError(f"Unknown tf '{tf}' for target '{col}'")
    return out

# ---------------- tau fitting (latent, total std) ----------------
def gaussian_nll(mu, sigma, y):
    sigma = np.maximum(sigma, 1e-12)
    return 0.5 * np.log(2 * np.pi * sigma**2) + 0.5 * ((y - mu) ** 2) / (sigma**2)

def fit_tau_per_target(mu_lat, std_tot_lat, t_true):
    """Fit per-target tau to scale total latent std by minimizing Gaussian NLL."""
    taus = []
    for j in range(mu_lat.shape[1]):
        mu = mu_lat[:, j]
        sig = std_tot_lat[:, j]
        y = t_true[:, j]
        def obj(log_tau):
            tau = np.exp(log_tau)
            return float(np.mean(gaussian_nll(mu, tau * sig, y)))
        res = minimize_scalar(obj, bounds=(-5, 5), method="bounded")
        taus.append(np.exp(res.x))
    return np.array(taus)

# ---------------- utilities: detect whether per-sample posterior columns exist ----------------
def _find_S_from_columns(df: pd.DataFrame, targets) -> int | None:
    # look for mu_lat_s0_<t> columns
    for t in targets:
        if f"mu_lat_s0_{t}" not in df.columns:
            return None
    # count s until missing
    s = 0
    while True:
        ok = all(f"mu_lat_s{s}_{t}" in df.columns for t in targets)
        if not ok:
            break
        s += 1
    return s if s > 0 else None

def load_latent_components(df: pd.DataFrame, targets):
    """
    Returns either:
      (mode="posterior", mu_lat_s: (S,N,D), std_ale_lat_s: (S,N,D))
    or
      (mode="gaussian",  mu_lat: (N,D), std_epi_lat: (N,D), std_ale_lat: (N,D))
    """
    S = _find_S_from_columns(df, targets)
    N = len(df)
    D = len(targets)
    if S is not None:
        # posterior mode: load mu_lat_s and std_ale_lat_s
        mu_lat_s = np.zeros((S, N, D), dtype=float)
        std_ale_lat_s = np.zeros((S, N, D), dtype=float)
        for j, t in enumerate(targets):
            for s in range(S):
                mu_lat_s[s, :, j] = df[f"mu_lat_s{s}_{t}"].to_numpy()
                std_ale_lat_s[s, :, j] = df[f"std_ale_lat_s{s}_{t}"].to_numpy()
        return "posterior", mu_lat_s, std_ale_lat_s
    # fallback Gaussian mode
    mu_lat = np.stack([df[f"mu_lat_mean_{t}"].to_numpy() for t in targets], axis=1)
    std_epi_lat = np.stack([df[f"std_epi_lat_{t}"].to_numpy() for t in targets], axis=1)
    std_ale_lat = np.stack([df[f"std_ale_lat_{t}"].to_numpy() for t in targets], axis=1)
    return "gaussian", mu_lat, std_epi_lat, std_ale_lat

# ---------------- sampling + stats in physical space ----------------
def sample_phys_posterior(mu_lat_s, std_ale_lat_s, tf_info, tf_eps, L: int, seed: int):
    """
    mu_lat_s: (S,N,D) posterior latent means
    std_ale_lat_s: (S,N,D) aleatoric std per posterior draw
    Draw L aleatoric samples per S and return y_samp: (S,L,N,D) in physical space.
    """
    rng = np.random.default_rng(seed)
    S, N, D = mu_lat_s.shape
    eps = rng.standard_normal(size=(S, L, N, D))
    t = mu_lat_s[:, None, :, :] + std_ale_lat_s[:, None, :, :] * eps  # (S,L,N,D)
    y = inverse_phys_tf(t.reshape(S * L * N, D), tf_info, tf_eps).reshape(S, L, N, D)
    return y

def sample_phys_gaussian(mu_lat, std_epi_lat, std_ale_lat, tf_info, tf_eps, K: int, L: int, seed: int):
    """
    Gaussian-epistemic fallback:
      mu_k ~ N(mu_lat, std_epi_lat^2)    (K,N,D)
      t_kl ~ N(mu_k,  std_ale_lat^2)     (K,L,N,D)
    Returns y_kl: (K,L,N,D)
    """
    rng = np.random.default_rng(seed)
    N, D = mu_lat.shape
    eps_epi = rng.standard_normal(size=(K, N, D))
    mu_k = mu_lat[None, :, :] + std_epi_lat[None, :, :] * eps_epi
    eps_ale = rng.standard_normal(size=(K, L, N, D))
    t = mu_k[:, None, :, :] + std_ale_lat[None, None, :, :] * eps_ale
    y = inverse_phys_tf(t.reshape(K * L * N, D), tf_info, tf_eps).reshape(K, L, N, D)
    return y

def stats_from_nested_cloud(y_kl: np.ndarray, quantiles=(0.05, 0.5, 0.95)):
    """
    y_kl: (K,L,N,D) where K indexes epistemic groups (posterior draws), L aleatoric draws.
    """
    K, L, N, D = y_kl.shape
    y_pool = y_kl.reshape(K * L, N, D)
    mean = y_pool.mean(axis=0)
    var_tot = y_pool.var(axis=0, ddof=0)
    std_tot = np.sqrt(np.maximum(var_tot, 1e-24))
    var_ale = y_kl.var(axis=1, ddof=0).mean(axis=0)          # E_k[Var_l]
    var_epi = y_kl.mean(axis=1).var(axis=0, ddof=0)          # Var_k[E_l]
    std_ale = np.sqrt(np.maximum(var_ale, 1e-24))
    std_epi = np.sqrt(np.maximum(var_epi, 1e-24))
    q_out = {}
    for q in quantiles:
        tag = f"q{int(round(q * 100)):02d}"
        q_out[tag] = np.quantile(y_pool, q, axis=0)
    return mean, std_tot, std_ale, std_epi, q_out

def write_outputs(df_in: pd.DataFrame, targets, out_csv: Path,
                  mean, std_tot, std_ale, std_epi, q_out,
                  suffix: str = ""):
    df = df_in.copy()
    for j, t in enumerate(targets):
        df[f"mean_{t}{suffix}"] = mean[:, j]
        df[f"std_tot_{t}{suffix}"] = std_tot[:, j]
        df[f"std_ale_{t}{suffix}"] = std_ale[:, j]
        df[f"std_epi_{t}{suffix}"] = std_epi[:, j]
        for tag, arr in q_out.items():
            df[f"{tag}_{t}{suffix}"] = arr[:, j]
    df.to_csv(out_csv, index=False)
    print(f"[INFO] wrote → {out_csv}")
    return df

def write_sme(df_full: pd.DataFrame, out_sme_csv: Path):
    drop = []
    for c in df_full.columns:
        if (
            c.startswith("mu_lat_")
            or c.startswith("mu_lat_s")
            or c.startswith("std_ale_lat_")
            or c.startswith("std_epi_lat_")
            or c.startswith("std_tot_lat_")
            or "_lat_" in c
        ):
            drop.append(c)

    df_sme = df_full.drop(columns=drop, errors="ignore")
    df_sme.to_csv(out_sme_csv, index=False)
    print(f"[INFO] wrote SME → {out_sme_csv}")


def coverage_report(df: pd.DataFrame, targets):
    print("\n[CAL] Empirical coverage on apply_csv:")
    for t in targets:
        y = df[f"true_{t}"].to_numpy()
        q05 = df[f"q05_{t}"].to_numpy()
        q95 = df[f"q95_{t}"].to_numpy()
        cov90 = np.mean((y >= q05) & (y <= q95))
        w90 = np.mean(q95 - q05)
        print(f"  {t}: cov90={cov90:.3f} (ideal 0.90), w90={w90:.4g}")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _find_latest_run_dir(root: str | Path | None = None) -> Path:
    """Walk into artifacts/ and find the most recent directory that contains model.pth.

    Handles both flat (artifacts/<run>/) and nested (artifacts/<sweep>/<run>/) layouts.
    Directories are compared by name (timestamp strings sort chronologically).
    """
    root_path = Path(root) if root is not None else _PROJECT_ROOT / "artifacts"
    if not root_path.is_dir():
        raise FileNotFoundError(f"Artifact root '{root_path}' does not exist")

    candidates: list[Path] = []
    for child in sorted(root_path.iterdir(), reverse=True):
        if not child.is_dir():
            continue
        # flat layout: child itself contains model.pth
        if (child / "model.pth").exists():
            candidates.append(child)
            continue
        # nested layout: child/<inner_run>/model.pth
        for inner in sorted(child.iterdir(), reverse=True):
            if inner.is_dir() and (inner / "model.pth").exists():
                candidates.append(inner)
                break  # only latest inner run per sweep

    if not candidates:
        raise FileNotFoundError(
            f"No run directory with model.pth found under '{root_path}'"
        )
    # pick the one whose *name* (timestamp) is latest
    best = max(candidates, key=lambda p: p.name)
    return best


def run_temp_tune(
    run_dir: Path,
    val_csv: Path | None = None,
    apply_csv: Path | None = None,
    out_csv: Path | None = None,
    out_sme_csv: Path | None = None,
    seed: int = 0,
    K: int = 200,
    L: int = 50,
) -> dict[str, float]:
    """Fit per-target tau on validation predictions and apply to test predictions.

    Returns the fitted taus dict (also saved to run_dir/taus.json).
    Can be called programmatically after training without argparse.
    """
    val_csv = val_csv or (run_dir / "predictions_val.csv")
    apply_csv = apply_csv or (run_dir / "predictions_test.csv")
    out_csv = out_csv or (apply_csv.parent / (apply_csv.stem + "_cal.csv"))

    # load config used
    with (run_dir / "config_used.yaml").open("r") as f:
        apply_config_used(cfg, yaml.safe_load(f))
    # load tf_info
    import torch
    meta = torch.load(run_dir / "data_meta.pt", map_location="cpu", weights_only=False)
    tf_info = meta["tf_info"]
    tf_eps = cfg.TF_EPS
    targets = cfg.TARGETS
    # ---------- fit tau on val (latent total std) ----------
    df_val = pd.read_csv(val_csv)
    for t in targets:
        for col in (f"mu_lat_mean_{t}", f"std_tot_lat_{t}", f"true_{t}"):
            if col not in df_val.columns:
                raise KeyError(f"{val_csv} missing required column '{col}'")
    mu_lat_val = np.stack([df_val[f"mu_lat_mean_{t}"].to_numpy() for t in targets], axis=1)
    std_tot_lat_val = np.stack([df_val[f"std_tot_lat_{t}"].to_numpy() for t in targets], axis=1)
    y_phys_val = np.stack([df_val[f"true_{t}"].to_numpy() for t in targets], axis=1)
    t_true_val = forward_phys_tf(y_phys_val, tf_info, tf_eps)
    taus = fit_tau_per_target(mu_lat_val, std_tot_lat_val, t_true_val)
    print("[INFO] fitted tau per target:")
    for t, tau in zip(targets, taus):
        print(f"  {t}: tau={tau:.6f}")

    # save taus to run_dir for reuse by inference_api
    taus_dict = {t: float(tau) for t, tau in zip(targets, taus)}
    taus_path = run_dir / "taus.json"
    with taus_path.open("w") as fh:
        json.dump(taus_dict, fh, indent=2)
    print(f"[INFO] saved taus → {taus_path}")

    # ---------- apply tau + sample ----------
    df_app = pd.read_csv(apply_csv)
    mode, *parts = load_latent_components(df_app, targets)

    def run_with_taus(taus_used: np.ndarray):
        """Return df_out with q05/q50/q95 and mean/std columns added (in physical space)."""
        if mode == "posterior":
            mu_lat_s, std_ale_lat_s = parts

            # scale epistemic spread (posterior means) around their ensemble mean
            mu_bar = mu_lat_s.mean(axis=0, keepdims=True)  # (1,N,D)
            mu_lat_s_cal = mu_bar + taus_used[None, None, :] * (mu_lat_s - mu_bar)

            # scale aleatoric std per posterior draw
            std_ale_lat_s_cal = std_ale_lat_s * taus_used[None, None, :]

            y_kl = sample_phys_posterior(mu_lat_s_cal, std_ale_lat_s_cal, tf_info, tf_eps,
                                         L=L, seed=seed)
        else:
            mu_lat, std_epi_lat, std_ale_lat = parts
            std_epi_use = std_epi_lat * taus_used[None, :]
            std_ale_use = std_ale_lat * taus_used[None, :]
            y_kl = sample_phys_gaussian(mu_lat, std_epi_use, std_ale_use,
                                        tf_info, tf_eps, K=K, L=L, seed=seed)

        mean, std_tot, std_ale, std_epi, q_out = stats_from_nested_cloud(
            y_kl, quantiles=(0.05, 0.5, 0.95)
        )
        # build dataframe in-memory (do not write)
        df_out = df_app.copy()
        for j, t in enumerate(targets):
            df_out[f"mean_{t}"] = mean[:, j]
            df_out[f"std_tot_{t}"] = std_tot[:, j]
            df_out[f"std_ale_{t}"] = std_ale[:, j]
            df_out[f"std_epi_{t}"] = std_epi[:, j]
            for tag, arr in q_out.items():
                df_out[f"{tag}_{t}"] = arr[:, j]
        return df_out

    # ---- PRE calibration (tau = 1) ----
    df_pre = run_with_taus(np.ones_like(taus))
    print("\n[CAL] Pre-calibration (tau=1):")
    coverage_report(df_pre, targets)

    # ---- POST calibration (fitted taus) ----
    df_post = run_with_taus(taus)
    print("\n[CAL] Post-calibration (fitted tau):")
    coverage_report(df_post, targets)

    # write only post-cal outputs
    df_post.to_csv(out_csv, index=False)
    print(f"[INFO] wrote → {out_csv}")

    if out_sme_csv:
        write_sme(df_post, Path(out_sme_csv))

    return taus_dict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=None, type=str,
                    help="Run directory containing config_used.yaml and data_meta.pt. "
                         "Default: most recent run under artifacts/")
    ap.add_argument("--val_csv", default=None, type=str,
                    help="Validation CSV to FIT tau from. Default: <run_dir>/predictions_val.csv")
    ap.add_argument("--apply_csv", default=None, type=str,
                    help="CSV to APPLY tau to. Must contain latent columns. "
                         "Default: <run_dir>/predictions_test.csv")
    ap.add_argument("--out_csv", default=None, type=str)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--K", default=200, type=int, help="Epistemic samples for gaussian fallback")
    ap.add_argument("--L", default=50, type=int, help="Aleatoric samples per epistemic draw")
    ap.add_argument("--out_sme_csv", default=None, type=str,
                    help="Optional SME CSV (drops latent/sample columns; keeps calibrated summaries).")
    args = ap.parse_args()

    # --- resolve run_dir (auto-detect if not given) ---
    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
    else:
        run_dir = _find_latest_run_dir()
        print(f"[INFO] auto-detected run_dir → {run_dir}")

    run_temp_tune(
        run_dir=run_dir,
        val_csv=Path(args.val_csv) if args.val_csv else None,
        apply_csv=Path(args.apply_csv) if args.apply_csv else None,
        out_csv=Path(args.out_csv) if args.out_csv else None,
        out_sme_csv=Path(args.out_sme_csv) if args.out_sme_csv else None,
        seed=args.seed,
        K=args.K,
        L=args.L,
    )
if __name__ == "__main__":
    main()