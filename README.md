# Conditional Variational Autoencoder for Stochastic Climate Dynamics

## Overview
This repository contains the implementation and experiments for a **Conditional Variational Autoencoder (CVAE)** designed to emulate the stochastic **Holton–Mass stratospheric model**, with a focus on **Sudden Stratospheric Warmings (SSWs)**.  
The project investigates **AI interpretability**, **rare-event modeling**, and the ability of deep generative models to reproduce **stochastic transitions in complex dynamical systems**.

Key contributions:
- Developed a **novel CVAE architecture** that autoregressively forecasts regime transitions.
- Applied **KL-divergence annealing** and **posterior collapse mitigation** to preserve meaningful latent structure.
- Identified **latent space clustering** aligned with physically distinct regimes.
- Demonstrated competitive stochastic modeling compared to traditional numerical approaches.
- First-author published work advancing the understanding of **AI interpretability in climate emulation**.

---

## Features
- **Training Framework**
  - Variational inference with KL annealing (`beta-VAE` style).
  - CRPS and Smooth L1 reconstruction loss options.
  - Mixed precision training with gradient scaling.
  - Model checkpointing and logging with Weights & Biases.

- **Inference**
  - Stochastic and deterministic tests of latent variable influence.
  - `z`-nullification and perturbation experiments to evaluate latent usage.
  - Transition statistics: mean transition duration, CCDF slope, exponential fits.

- **Latent Space Analysis**
  - PCA decomposition and visualization of latent means (`mu`).
  - K-means clustering to evaluate alignment with dynamical regimes.
  - Clear clustering structure observed in latent space.

- **Transition Diagnostics**
  - Transition detection functions for both `A→B` and `B→A`.
  - Empirical distribution analysis of transition durations.
  - Histogram and exponential fit comparisons between model and true system.

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
- PyTorch (CUDA enabled for GPU acceleration)
- NumPy, SciPy, scikit-learn
- Seaborn, Matplotlib, Plotly
- Weights & Biases (`wandb`)

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
- The CVAE can **reproduce transition dynamics** between regimes.
- Latent vectors exhibit **structured clustering**, rare in VAE studies.
- With stochastic latent sampling (`z ~ N(0,I)`), the model captures PDFs more accurately.
- With `z=0` (nullification), transitions persist but stochastic variability is lost → evidence of **partial posterior collapse**.

---

## Why It Matters
- Addresses **AI interpretability** in generative models applied to climate.
- Demonstrates that **latent structure aligns with physical regimes**, a novel finding.
- Opens pathways for **rare-event emulation** and **robust climate forecasting**.

---

## Authors
- **First Author:** [Your Name]  
  Undergraduate Researcher, University of Chicago  
- Supervised by: [Advisor’s Name]  

---

## Citation
If you use this code, please cite our work:
```
et al. (2025). "Interpretable Latent Representations of Stochastic Stratospheric Dynamics via Conditional Variational Autoencoders." AGU Fall Meeting 2025.
```
