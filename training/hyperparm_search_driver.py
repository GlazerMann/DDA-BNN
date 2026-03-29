"""Grid search driver for hyperparameter sweeps.

Reads a YAML file of parameter lists, generates the Cartesian product,
and runs ``train.main()`` for each combination.  Results are collected
into ``summary.csv`` under the sweep directory.

Example::

    python training/hyperparm_search_driver.py \\
        --grid training/configs/grid_search.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import importlib
import itertools
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
import pyro

import training.config as cfg


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------- CLI -------------------------------------------------------
def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--grid", default=str(_PROJECT_ROOT / "training" / "configs" / "grid_search.yaml"),
                   help="YAML file that lists values to sweep")
    p.add_argument("--base", default=str(_PROJECT_ROOT / "training" / "configs" / "default.yaml"),
                   help="Optional base override.yaml")
    p.add_argument("--root", default=str(_PROJECT_ROOT / "artifacts"),
                   help="Top-level artefact directory (default: <project>/artifacts/)")
    p.add_argument("--name", default="{}".format(datetime.now().strftime("%Y%m%d_%H%M%S")),
                   help="Name of this sweep; default: time-stamp")
    return p.parse_args()


# ---------- small helpers ---------------------------------------------
def load_yaml(path: str | Path | None) -> dict:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as fh:   # specify encoding!
        return yaml.safe_load(fh) or {}


def cartesian(d: dict[str, list]) -> list[dict]:
    """Yield every combination from a dict of lists."""
    keys, vals = zip(*d.items())
    for combo in itertools.product(*vals):
        yield dict(zip(keys, combo))


def dump_yaml(d: dict, dest: Path):
    """Write a dict to a YAML file."""
    with dest.open("w") as fh:
        yaml.safe_dump(d, fh, sort_keys=False)


# ---------- main -------------------------------------------------------
def main():
    args = _cli()

    grid_cfg = load_yaml(args.grid)
    base_cfg = load_yaml(args.base)

    # sweep directory
    sweep = args.name or datetime.now().strftime("grid_%Y%m%d_%H%M%S")
    sweep_dir = Path(args.root) / sweep
    sweep_dir.mkdir(parents=True, exist_ok=False)
    print(f"[GRID] All runs will be saved under {sweep_dir.resolve()}")

    summary = []

    for i, combo in enumerate(cartesian(grid_cfg)):
        # ----------------------------------------------------------------
        # Build config for *this* run
        run_cfg = dict(base_cfg)
        run_cfg.update(combo)
        run_cfg["ARTIFACT_DIR"] = str(sweep_dir.resolve())  # absolute so _post() won't prepend ROOT_DIR

        # Store override so we know what we did
        ov_yaml = sweep_dir / f"override_{i:03d}.yaml"
        dump_yaml(run_cfg, ov_yaml)

        # Load into global cfg
        cfg.load(ov_yaml)

        # (re-)seed RNGs
        torch.manual_seed(cfg.SEED)
        np.random.seed(cfg.SEED)
        random.seed(cfg.SEED)
        pyro.clear_param_store()

        # ----------------------------------------------------------------
        # Train
        train_mod = importlib.import_module("training.train")
        train_mod = importlib.reload(train_mod)
        artefact_dir: Path = train_mod.main()

        # obtain last validation loss
        loss_df = pd.read_csv(artefact_dir / "loss.csv")
        val_loss = float(loss_df["val_tot"].min())
        summary.append(dict(run=artefact_dir.name, val=val_loss, **combo))

    # --------------------------------------------------------------------
    pd.DataFrame(summary).sort_values("val").to_csv(sweep_dir / "summary.csv",
                                                    index=False)
    print(f"[GRID] Finished – summary at {sweep_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()