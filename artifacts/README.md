# artifacts/

Output directory for training runs. Each run produces a timestamped
sub-directory containing the trained model, predictions, and diagnostics.

## Example run

`20260329_134948/` is a sample sweep (single-config) kept in version control
as a reference. It contains:

```
20260329_134948/
├── override_000.yaml              # config overrides for this sweep entry
├── summary.csv                    # sweep-level summary (val loss per run)
└── 20260329_134949/               # the actual training run
    ├── config_used.yaml           # full config snapshot
    ├── model.pth                  # trained model weights
    ├── data_meta.pt               # scalers, transform info, split indices
    ├── pyro_params.pt             # Bayesian variational parameters
    ├── taus.json                  # temperature calibration factors
    ├── loss.csv                   # per-epoch train/val losses
    ├── loss_components.png        # loss curve plot
    ├── pred_plot.png              # prediction scatter plot
    ├── predictions_val.csv        # validation set predictions
    ├── predictions_test.csv       # test set predictions
    └── predictions_test_cal.csv   # tau-calibrated test predictions
```

## New runs

All other run directories created by `train.py` or `hyperparm_search_driver.py`
are git-ignored. Only the example above is tracked.

