# Jet Tagging Deep Learning Pipeline

[![CI Pipeline](https://github.com/ianfly1/jet-tag/actions/workflows/ci.yml/badge.svg)](https://github.com/ianfly1/jet-tag/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch Lightning](https://img.shields.io/badge/PyTorch-Lightning-7015A8.svg)](https://lightning.ai/)

A machine learning repository classifying simulated hadronically decaying particles (gluons, light quarks, top quarks) from the **JetNet** benchmark dataset using PyTorch, Hydra, and PyTorch Lightning pipelines.

## Key Results & Physics Benchmarks

| Architecture | Representation Input | Model Parameters | Test Accuracy | ROC-AUC | Background Rejection ($R_{50}$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Joint Autoencoder + MLP** | Particle Constituents ($30 \times 3$) | 17.1 K | **87.1%** | **0.937** | **32.5** |
| **JetCNN** | Jet Images ($25 \times 25 \times 1$) | 37.9 K | **87.4%** | **0.944** | **51.8** |

*Note: Background Rejection Factor ($R_{50} = 1/\epsilon_B$) is measured at a fixed Signal Efficiency ($\epsilon_S = 0.50$).*

## Physics Context $\rightarrow$ Industry ML Mapping
- **Jet Tagging**: Binary/Multi-class classification of unordered particle point clouds.
- **Constituent Particles**: Spatial graph node coordinates ($p_T, \eta, \phi$).
- **Background Rejection**: Inverse False Positive Rate ($1 / \text{FPR}$) at fixed target recall.

## Setup & Quick Start

```bash
# 1. Clone & create virtual environment
git clone [https://github.com/ianfly1/jet-tagging-dl.git](https://github.com/ianfly1/jet-tagging-dl.git)
cd jet-tagging-dl
python3.11 -m venv .venv
source .venv/bin/activate

# 2. Install package in editable mode
pip install -e ".[dev]"

# 3. Run unit test suite
pytest tests/ -v

# 4. Train model via Hydra CLI
python train.py model=encdec data=jet_particle trainer.epochs=15