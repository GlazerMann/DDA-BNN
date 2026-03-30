# DDA-BNN: A Morphology-Aware Bayesian Neural Network Surrogate for Black Carbon Optical Properties

DDA-BNN is a surrogate modeling framework for predicting the optical properties of black carbon–containing particles from particle morphology, coating state, and composition. Trained on discrete dipole approximation (DDA) simulations, it predicts extinction efficiency, single-scattering albedo, and asymmetry parameter while also quantifying aleatoric and epistemic uncertainty.

This branch is a stand alone Jupyter notebook implementation for model training and evaluation.

## Repository structure

```
aerosol_bnn/
├── release/             # standalone inference package (start here)
│   ├── README.md        # usage instructions
│   ├── inference_notebook.ipynb
│   ├── inference_api.py
│   ├── model.py
│   ├── config.py
│   └── chosen_model/    # pre-trained weights + calibration
├── training/            # full training pipeline
│   ├── train.py
│   ├── hyperparm_search_driver.py
│   ├── temp_tune.py
│   ├── inference_api.py
│   └── ...
├── analysis/            # post-training diagnostics
│   ├── run_analysis.py
│   └── ...
├── data/                # datasets
└── artifacts/           # training run outputs
```

## Using a pre-trained model

See [`release/README.md`](release/README.md) for setup and usage instructions. This is the recommended starting point for most users who want to use the pre-trained model for inference.

## Training a new model

See [`training/README.md`](training/README.md) for the full pipeline:

```
hyperparm_search_driver.py → train.py → temp_tune.py → inference_api.py
```

## Environment setup

```bash
conda env create -f environment.yml
conda activate bnn_notebook_env
```

## Reference
Archived release: https://doi.org/10.5281/zenodo.19324375
