"""Only marginal standard deviations are retained by split_output()."""
import unittest
from unittest.mock import patch
try:
    import torch
    from training import model as train_model
    from release import model as release_model
    from training.train import split_output as training_split
    from release.inference_api import split_output as release_split
except (ImportError, FileNotFoundError) as exc:
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

@unittest.skipIf(_IMPORT_ERROR is not None, f"Repository dependencies/config unavailable: {_IMPORT_ERROR}")
class MarginalCovarianceTests(unittest.TestCase):
    def test_marginal_std_agrees_with_covariance_diagonal(self):
        for module, splitter in ((train_model, training_split), (release_model, release_split)):
            with self.subTest(module=module.__name__), \
                 patch.object(module.cfg, 'TARGETS', ['A','B','C']), \
                 patch.object(module.cfg, 'UNCERTAINTY_MODE', 'full'):
                out = torch.ones((3, 9)) * 0.7  # D=3, 3 means + 6 parameters
                params = out[:, 3:]
                sigma = torch.exp(0.5 * (module.cfg.LOG_VAR_MIN +
                        (module.cfg.LOG_VAR_MAX - module.cfg.LOG_VAR_MIN) *
                        torch.sigmoid(params[:, :3])))
                cov = module.build_cov_from_params(sigma, params[:, 3:])
                mean, std = splitter(out)
                torch.testing.assert_close(mean, out[:, :3])
                torch.testing.assert_close(std, torch.diagonal(cov, dim1=-2, dim2=-1).sqrt())
                self.assertTrue(bool((cov[:, 1, 0].abs() > 0).all()))
