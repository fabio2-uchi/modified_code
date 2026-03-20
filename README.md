# AI Emulation of Stochastic Sudden Stratospheric Warming with Interpretable Latent Structure

## Overview
This repository contains the implementation for a **Conditional Variational Autoencoder (CVAE)** designed to probabilistically emulate the stochastic **Holton–Mass stratospheric model**, with a focus on reproducing **Sudden Stratospheric Warming (SSW)** dynamics.

Key contributions:
- **ResNet-inspired CVAE** that autoregressively forecasts regime transitions with one-day lead time.
- Accurate reproduction of short-term dynamics, steady-state PDFs, regime persistence, rare transition rates, committor functions, and expected lead times.
- **Interpretable latent space**: PCA reveals four well-separated clusters corresponding to strong/weak vortex states and transition pathways (A→B, B→A).
- KL-divergence annealing and Huber loss for stable training with rare-event fidelity.

---

## Project Structure
```
├── train.py                # Training loop with KL annealing and logging
├── inference.py            # Inference pipeline for stochastic/deterministic tests
├── model.py                # Conditional VAE model implementation
├── analysis_latent.py      # PCA, clustering, and latent diagnostics
├── transitions.py          # Transition detection and statistics
├── plots/                  # Figures for PCA, histograms, predictions
├── save_folder/            # Model checkpoints and results
├── long_run_310k.npy       # Stratospheric training dataset (Holton–Mass simulation)
└── README.md               # Project documentation
```

---

## Getting Started

### Prerequisites
- Python 3.10+
- PyTorch (CUDA recommended)
- NumPy, SciPy, scikit-learn
- Seaborn, Matplotlib
- Weights & Biases (`wandb`)

### Data
Place `long_run_310k.npy` in the repository root (or set `HM_DATA_PATH` environment variable). This file contains a 3×10⁵-day simulation of the stochastic Holton-Mass model.

### Training
```bash
python train.py
```

### Inference
```bash
python inference.py
```

### Latent Analysis
```bash
python analysis_latent.py
```

---

## Results
- The CVAE captures **bimodal steady-state distributions**, **transition return periods**, and **committor functions**.
- Latent vectors exhibit **structured clustering** into four physically interpretable regimes without supervision.
- PC1 separates strong vs weak vortex states; PC2 distinguishes stable states from transition-prone configurations.

---

## Authors
- **C. Daniel Boscu**, **Daniel Hernandez**, **Fabio Alvarez Ventura** (equal contribution)
- Justin Finkel, Ashesh Chattopadhyay, Pedram Hassanzadeh, Dorian S. Abbot

University of Chicago & UC Santa Cruz

---

## Citation
```
Boscu, Hernandez, Alvarez Ventura et al. (2025). "AI Emulation of Stochastic Sudden
Stratospheric Warming with Interpretable Latent Structure." JGR: Machine Learning
and Computation.
```
