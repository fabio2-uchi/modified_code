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
├── model.py                # Conditional VAE model implementation
├── train.py                # Training loop with KL annealing and logging
├── inference.py            # Autoregressive inference pipeline
├── transitions.py          # Transition detection and statistics
├── analysis_latent.py      # PCA, clustering, and latent diagnostics
├── plots/
│   ├── generate_plots/     # Code to reproduce paper figures
│   │   ├── holton_mass.py              # Holton-Mass model (for one-step tests)
│   │   ├── double_one_step_test.ipynb  # Fig. 3: One-step RMSE vs altitude
│   │   ├── rmse_calcs.ipynb            # Fig. 4: Forecast error growth by altitude
│   │   ├── steady_state_density.ipynb  # Fig. 5: Steady-state density (U vs IHF)
│   │   ├── timeseries_pdf.ipynb        # Fig. 2: Time series & PDF comparison
│   │   ├── committor.ipynb             # Fig. 6: Committor function q+(x)
│   │   ├── lead_time.ipynb             # Fig. 7: Expected lead time η+_B(x)
│   │   ├── ccdf.ipynb                  # Fig. 8: CCDF of transition durations
│   │   └── latent_pca.ipynb            # Fig. 9: Latent space PCA
│   └── graphs_for_paper/   # Output PNGs for the paper
├── save_folder/            # Model checkpoints
├── long_run_310k.npy       # Stratospheric training dataset (Holton–Mass simulation)
└── README.md
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
