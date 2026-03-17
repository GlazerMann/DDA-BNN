# DDA-BNN: A Morphology-Aware Bayesian Neural Network Surrogate for Black Carbon Optical Properties

DDA-BNN is a surrogate modeling framework for predicting the optical properties of black carbon–containing particles from particle morphology, coating state, and composition. Trained on discrete dipole approximation (DDA) simulations, it predicts extinction efficiency, single-scattering albedo, and asymmetry parameter while also quantifying aleatoric and epistemic uncertainty.

This branch is a stand alone Jupyter notebook implementation for model training and evaluation.

---

# Interactive Jupyter notebook

Prerequisite: Conda installed. Run these from the repo root (where environment.yml lives).

1) Create the Conda environment
```bash
conda env create -f environment.yml
```

2) Activate the environment
```bash
conda activate bnn_notebook_env
```

3) Register the Jupyter kernel (one-time)
```bash
python -m ipykernel install --user --name bnn_notebook_env --display-name "Python (bnn_notebook_env)"
```
4) Launch Jupyter and open the notebook file (model_explorer.ipynb)
```bash
jupyter notebook
```
In the UI: Kernel → Change Kernel → select "Python (bnn_notebook)"

---
