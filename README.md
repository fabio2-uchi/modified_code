# AI Emulation of Stochastic Sudden Stratospheric Warming with Interpretable Latent Structure

## Overview
This repository contains the implementation for a **Conditional Variational Autoencoder (CVAE)** designed to probabilistically emulate the stochastic **Holton–Mass stratospheric model**, with a focus on reproducing **Sudden Stratospheric Warming (SSW)** dynamics.

Key contributions:
- **ResNet-inspired CVAE** that autoregressively forecasts regime transitions with one-day lead time.
- Accurate reproduction of short-term dynamics, steady-state PDFs, regime persistence, rare transition rates, committor functions, and expected lead times.
- **Interpretable latent space**: PCA reveals four well-separated clusters corresponding to strong/weak vortex states and transition pathways (A→B, B→A), recovered **without labels** by a Gaussian mixture model. A LASSO analysis maps the principal components back to physical variables.
- KL-divergence annealing and Huber loss for stable training with rare-event fidelity.

---

## Project Structure
```
├── model.py                # Conditional VAE model implementation
├── train.ipynb                # Training loop with KL annealing and logging
├── inference.ipynb            # Autoregressive inference pipeline
├── plots/
│   ├── generate_plots/     # Code to reproduce paper figures
│   │   ├── holton_mass.py              # Holton-Mass model (for one-step tests)
│   │   ├── timeseries_pdf.ipynb        # Fig. 2: Time series & PDF comparison
│   │   ├── double_one_step_test.ipynb  # Fig. 3: One-step RMSE vs altitude
│   │   ├── rmse_calcs.ipynb            # Fig. 4: Forecast error growth by altitude
│   │   ├── steady_state_density.ipynb  # Fig. 5: Steady-state density (U vs IHF)
│   │   ├── ccdf.ipynb                  # Figs. 6 & 7: CCDF of persistence times + transition durations
│   │   ├── committor.ipynb             # Fig. 8: Committor function q+(x)
│   │   ├── lead_time.ipynb             # Fig. 9: Expected lead time η+_B(x)
│   │   └── latent_pca.ipynb            # Figs. 10 & 11: Latent-space PCA, GMM clustering, LASSO paths
│   └── graphs_for_paper/   # Output PNGs for the paper
└── README.md
```

All figure notebooks load the canonical weights `checkpoint_11` and read the same
data arrays; each writes its PNG(s) into `plots/graphs_for_paper/`.

---

## Getting Started

### Prerequisites
- Python 3.10+
- PyTorch (CUDA recommended)
- NumPy, SciPy, scikit-learn
- Seaborn, Matplotlib

---

## Results
- The CVAE captures **bimodal steady-state distributions**, **transition return periods**, and **committor functions**.
- Latent vectors exhibit **structured clustering** into four physically interpretable regimes without supervision: a Gaussian mixture model fit to the latent coordinates alone recovers the four regimes (silhouette score independently maximized at k=4), reaching ≈96% (emulator) / ≈89% (Holton–Mass) agreement with the physically defined classes on the full latent space.
- A LASSO regression maps the components back to physics: **PC1 is essentially the vortex strength** (zonal wind at 30 km, r=0.98), **PC3 tracks the wave forcing** (heat flux, r=0.66) that precedes transitions, and **PC2 is the wind-profile shape**.

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
