"""Fast validation and smoke tests for the bundled DDA-BNN release."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import nbformat
import numpy as np
import yaml

from release.inference_api import run_inference_latent, run_inference_phys
import training.inference_api as training_api

from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "release" / "chosen_model"

class TrainingInferenceTests(unittest.TestCase):
    @staticmethod
    def representative_input() -> np.ndarray:
        return np.array(
            [[
                28.0,
                1.0,
                0.0,
                np.pi * 0.121537 / 0.7,
                1.76891,
                0.816859706,
                0.121279497,
                0.066140142,
            ]],
            dtype=np.float32,
        )

    def test_physical_inference_disables_nested_tau_scaling(self) -> None:
        """Physical inference must apply tau calibration exactly once."""

        with patch.object(
            training_api,
            "run_inference_latent",
            wraps=training_api.run_inference_latent,
        ) as latent_mock:
            training_api.run_inference_phys(
                MODEL_DIR,
                self.representative_input(),
                num_mc=4,
                seed=0,
                L=4,
            )

        latent_mock.assert_called_once()

        self.assertIn("taus", latent_mock.call_args.kwargs)
        self.assertIsNone(latent_mock.call_args.kwargs["taus"])

  def test_physical_none_taus_disables_all_tau_scaling(self) -> None:
    with patch.object(
        training_api,
        "run_inference_latent",
        wraps=training_api.run_inference_latent,
    ) as latent_mock:
        training_api.run_inference_phys(
            MODEL_DIR,
            self.representative_input(),
            num_mc=4,
            seed=0,
            taus=None,
            L=4,
        )

    latent_mock.assert_called_once()
    self.assertIsNone(latent_mock.call_args.kwargs["taus"])
