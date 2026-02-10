# MIE aerosol property correction via Neural Networks with Uncertainty Quantification

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