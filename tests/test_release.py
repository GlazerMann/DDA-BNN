"""Fast validation and smoke tests for the bundled DDA-BNN release."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import nbformat
import numpy as np
import yaml

from release.inference_api import run_inference_phys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "release" / "chosen_model"


class RepositoryValidationTests(unittest.TestCase):
    def test_configuration_files_are_valid_yaml(self) -> None:
        config_files = sorted(PROJECT_ROOT.glob("**/configs/*.[Yy][Aa][Mm][Ll]"))
        self.assertGreater(len(config_files), 0)

        for path in config_files:
            with self.subTest(path=path.relative_to(PROJECT_ROOT)):
                with path.open(encoding="utf-8") as stream:
                    self.assertIsInstance(yaml.safe_load(stream), dict)

    def test_release_notebook_is_valid(self) -> None:
        notebook_path = PROJECT_ROOT / "release" / "inference_notebook.ipynb"
        with notebook_path.open(encoding="utf-8") as stream:
            notebook = nbformat.read(stream, as_version=4)
        nbformat.validate(notebook)


class ReleaseInferenceTests(unittest.TestCase):
    def test_bundled_model_runs_inference(self) -> None:
        required_artifacts = {
            "config_used.yaml",
            "data_meta.pt",
            "model.pth",
            "pyro_params.pt",
            "taus.json",
        }
        self.assertTrue(required_artifacts.issubset({path.name for path in MODEL_DIR.iterdir()}))

        # One representative row in the documented feature order:
        # Npp, V/V0, coating_RI_imag, Xve, core_Df, Qext_HS, SSA_HS, g_HS.
        x_raw = np.array(
            [[28.0, 1.0, 0.0, np.pi * 0.121537 / 0.7, 1.76891,
              0.816859706, 0.121279497, 0.066140142]],
            dtype=np.float32,
        )
        with (MODEL_DIR / "taus.json").open(encoding="utf-8") as stream:
            taus = json.load(stream)

        mean, std_ale, std_epi, std_tot, quantiles = run_inference_phys(
            MODEL_DIR,
            x_raw,
            num_mc=4,
            seed=0,
            taus=taus,
            L=4,
        )

        for name, values in {
            "mean": mean,
            "aleatoric standard deviation": std_ale,
            "epistemic standard deviation": std_epi,
            "total standard deviation": std_tot,
            **quantiles,
        }.items():
            with self.subTest(output=name):
                self.assertEqual(values.shape, (1, 3))
                self.assertTrue(np.isfinite(values).all())

        self.assertTrue((std_ale >= 0).all())
        self.assertTrue((std_epi >= 0).all())
        self.assertTrue((std_tot >= 0).all())
        self.assertTrue((mean[:, 0] > 0).all())
        self.assertTrue(((mean[:, 1:] > 0) & (mean[:, 1:] < 1)).all())
        self.assertTrue((quantiles["q05"] <= quantiles["q50"]).all())
        self.assertTrue((quantiles["q50"] <= quantiles["q95"]).all())


if __name__ == "__main__":
    unittest.main()
