"""Global configuration loaded from YAML with optional run-time overrides.

Reads ``configs/default.yaml`` at import time and exposes every key as a
module-level global (``LR``, ``EPOCHS``, …) and via ``cfg.ns``.

Example::

    import training.config as cfg
    cfg.load("experiments/try_lr_5e4.yaml")
"""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import json
import yaml

import torch

# ------------------------------------------------------------------ #
# helpers                                                            #
# ------------------------------------------------------------------ #
def _device(val: str | None) -> torch.device:
    if val in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(val)

def _apply(d: dict[str, Any]) -> None:
    """Copy key–value pairs into module globals *and* ns."""
    g = globals()
    for k, v in d.items():
        g[k] = v
        setattr(ns, k, v)

def _post() -> None:
    """Resolve paths & device after every load/override."""
    # ROOT_DIR is relative to training/ (where this config.py lives), not CWD
    _this_dir = Path(__file__).resolve().parent
    root = (_this_dir / globals()["ROOT_DIR"]).resolve()
    globals()["ROOT_DIR"] = root
    ns.ROOT_DIR = root

    for key in ("DATA_FILE", "ARTIFACT_DIR"):
        p = Path(globals()[key])
        if not p.is_absolute():
            p = root / p
        globals()[key] = ns.__dict__[key] = p

    ns.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    globals()["DEVICE"] = ns.DEVICE = _device(globals()["DEVICE"])

def _read_file(path: Path) -> dict[str, Any]:
    """Return dict from JSON or YAML file, chosen by extension."""
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        return yaml.safe_load(path.read_text()) or {}
    if suffix == ".json":
        return json.loads(path.read_text()) or {}
    raise ValueError(f"Unsupported config format: '{path.suffix}'")


# Print out the current configuration nicely
def print_cfg():
    """
    Nicely print current configuration
    """
    from pprint import pformat
    print("\nCurrent configuration\n" + "-" * 70)
    g = globals()
    for k in _baseline.keys():
        v = g[k]
        if isinstance(v, (list, dict, tuple, Path)):
            pretty = pformat(v, compact=True, width=80)
            print(f"{k} = {pretty}")
        else:
            print(f"{k} = {v}")
    print("-" * 70 + "\n")

# ------------------------------------------------------------------ #
# namespace & baseline load                                          #
# ------------------------------------------------------------------ #
ns = SimpleNamespace()

_DEFAULT_FILE = Path(__file__).resolve().parent / "configs" / "default.yaml"
_baseline = _read_file(_DEFAULT_FILE)        # dict[str, Any]

_apply(_baseline)
_post()

# ------------------------------------------------------------------ #
# run-time override                                                  #
# ------------------------------------------------------------------ #
def load(path: str | Path) -> None:
    """
    Override the current configuration with another YAML or JSON file.
    Unknown keys raise KeyError.
    """
    path = Path(path).expanduser().resolve()
    data = _read_file(path)

    unknown = data.keys() - _baseline.keys()
    if unknown:
        raise KeyError(f"Unknown config keys: {', '.join(unknown)}")

    _apply(data)
    _post()

# ------------------------------------------------------------------ #
# public symbols                                                     #
# ------------------------------------------------------------------ #
__all__ = ["ns", "load"] + list(_baseline.keys())