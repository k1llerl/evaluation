# Evaluation

A lightweight evaluation toolkit for ML/quant research: data loading → model training → metric reporting, organized with YAML configs for reproducible experiments.

---

## What this repo does

This repository is used to **evaluate predictive models** (e.g., LightGBM / LSTM) on financial datasets and factor-based features.

It is designed to keep the workflow clear and reproducible:

- **Config-driven experiments** (YAML)
- **Modular data loading**
- **Training scripts for different model families**
- **Consistent evaluation outputs** for iteration and comparison

---

## Project structure

evaluation/
├── dataloader.py                  # dataset loading / preprocessing utilities  
├── LGB_train*                      # LightGBM training scripts (variants for different settings)  
├── run_regression_LSTM.py          # LSTM regression training script  
├── default.yaml                    # default experiment configuration  
├── data_yaml.yaml                  # dataset/feature configuration  
├── useful_factor_list*.csv         # factor/feature list used for modeling  
└── README.md  

> Notes  
> - `LGB_train*` may include multiple versions for different time windows or experiment settings.  
> - `useful_factor_list*.csv` defines the factor set used by the pipeline.

---

## Quick start

Install dependencies:

    pip install -r requirements.txt

(Recommended) Create a clean virtual environment:

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

Run training / evaluation:

LightGBM (example):

    python LGB_train.py

LSTM regression (example):

    python run_regression_LSTM.py

---

## Configuration

Experiments are controlled via YAML configs:

- `default.yaml`: model settings / training parameters
- `data_yaml.yaml`: dataset path, feature list, split rules, etc.

Typical workflow:

1. Update `data_yaml.yaml` for dataset and factor list
2. Update `default.yaml` for model hyperparameters
3. Run a training script and compare results across configs

---

## Intended use cases

- Factor signal modeling and feature selection
- Cross-sectional or time-series regression evaluation
- Comparing ML baselines (LGBM vs LSTM) under consistent data splits
- Rapid iteration during quant research

---

## Notes / best practices

- Keep large datasets out of the repo (use links / external storage).
- Track experiment settings through YAML to ensure reproducibility.
- When adding new models, follow the existing modular structure:
  dataloader → model script → evaluation outputs

---

## License

For research / educational use.