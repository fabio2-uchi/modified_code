"""
inference.py — Inference pipeline for stochastic/deterministic tests

Loads a trained CVAE checkpoint and runs:
  1. Autoregressive long-run inference (stochastic z sampling)
  2. Time series comparison (real vs predicted)
  3. Single-step profile comparisons (U, Re{Psi}, Im{Psi})
"""

import os
import random

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
from torch.amp import autocast

from model import ConditionalVAE

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_PATH = os.environ.get("HM_DATA_PATH", "long_run_310k.npy")
WEIGHTS_PATH = os.environ.get(
    "HM_WEIGHTS_PATH",
    "save_folder/best_weights/checkpoint_11"
)

LATENT_DIM = 32
OUTPUT_DIM = 75
CONDITION_DIM = 50
LEVEL = 63
TIME_STEPS = 30_000
NUM_PROFILE_SAMPLES = 5

# ============================================================================
# LOAD DATA
# ============================================================================
print("Loading data...")
psi_raw = np.load(DATA_PATH)            # (N, 2, 75)
psi = psi_raw[:, 1, :]
mean_psi = np.mean(psi, axis=0, keepdims=True)
std_psi = np.std(psi, axis=0, keepdims=True)
psi_norm = (psi - mean_psi) / std_psi

start, end = 0, CONDITION_DIM

# ============================================================================
# LOAD MODEL
# ============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConditionalVAE(LATENT_DIM, OUTPUT_DIM, CONDITION_DIM).to(device)

if os.path.exists(WEIGHTS_PATH):
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device, weights_only=True))
    print(f"Loaded weights from {WEIGHTS_PATH}")
else:
    raise FileNotFoundError(f"Checkpoint not found: {WEIGHTS_PATH}")

model.eval()

# ============================================================================
# AUTOREGRESSIVE INFERENCE
# ============================================================================
print(f"Running autoregressive inference for {TIME_STEPS} steps...")
initial_cond = torch.tensor(psi_norm[0, start:end]).reshape(1, CONDITION_DIM)
z = torch.zeros(1, LATENT_DIM)
pred_norm = np.zeros((TIME_STEPS, OUTPUT_DIM))

for k in tqdm(range(TIME_STEPS), desc="Inference"):
    with torch.inference_mode():
        with autocast(device_type=device.type):
            z_sample = torch.randn(1, LATENT_DIM).to(device)
            if k == 0:
                cond = initial_cond.float().to(device)
            else:
                cond = torch.tensor(
                    pred_norm[k - 1, start:end]
                ).reshape(1, CONDITION_DIM).float().to(device)

            y = model.decode(z_sample, cond).detach().cpu().numpy()
            pred_norm[k, :] = y

# Denormalize
pred = pred_norm * std_psi + mean_psi
actual = psi * std_psi + mean_psi

np.save("plots/best_pred.npy", pred)
print("Predictions saved to plots/best_pred.npy")

# ============================================================================
# PLOT 1: Time series comparison
# ============================================================================
os.makedirs("plots", exist_ok=True)

fig, ax = plt.subplots(figsize=(20, 8))
ax.plot(actual[:TIME_STEPS, LEVEL], 'b', label='Holton-Mass', alpha=0.7)
ax.plot(pred[:TIME_STEPS, LEVEL], 'r', label='Emulator', alpha=0.7)
ax.set_xlabel("Time [days]", fontsize=14)
ax.set_ylabel("U(30 km) [m/s]", fontsize=14)
ax.set_title("Time Series: Holton-Mass vs Emulator", fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("plots/timeseries_comparison.png")
plt.close(fig)
print("Saved plots/timeseries_comparison.png")

# ============================================================================
# PLOT 2: Bimodal PDF comparison
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))
sns.histplot(actual[:TIME_STEPS, LEVEL], bins=50, kde=True, color='blue',
             alpha=0.6, element='step', label='Holton-Mass', ax=ax)
sns.histplot(pred[:, LEVEL], bins=50, kde=True, color='red',
             alpha=0.6, element='step', label='Emulator', ax=ax)
ax.set_xlabel("U(30 km) [m/s]", fontsize=14)
ax.set_ylabel("Density", fontsize=14)
ax.set_title("Steady-State PDF Comparison", fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("plots/pdf_comparison.png")
plt.close(fig)
print("Saved plots/pdf_comparison.png")

# ============================================================================
# PLOT 3: Single-step profile comparisons
# ============================================================================
time_indices = random.sample(range(0, len(psi) - 2), NUM_PROFILE_SAMPLES)
print(f"Profile sample indices: {time_indices}")

z_levels_25 = np.linspace(0, 70, 25)

for i, t_idx in enumerate(time_indices):
    real_current = actual[t_idx, :]
    real_next = actual[t_idx + 1, :]

    # One-step prediction from normalized current state
    cond = torch.tensor(psi_norm[t_idx, start:end]).reshape(1, CONDITION_DIM)
    with torch.no_grad():
        z_sample = torch.randn(1, LATENT_DIM).to(device)
        cond_dev = cond.float().to(device)
        y = model.decode(z_sample, cond_dev).detach().cpu().numpy()
    pred_next = y.squeeze() * std_psi.squeeze() + mean_psi.squeeze()

    # U profiles (indices 50:75)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].plot(real_current[50:75], z_levels_25, 'x-', label="Current (real)")
    axes[0].plot(real_next[50:75], z_levels_25, 'd-', label="Next (real)")
    axes[0].plot(pred_next[50:75], z_levels_25, 's--', label="Next (pred)")
    axes[0].set_xlabel("U [m/s]", fontsize=13)
    axes[0].set_ylabel("Altitude [km]", fontsize=13)
    axes[0].set_title(f"U Profiles @ step {t_idx}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    dU_real = real_next[50:75] - real_current[50:75]
    dU_pred = pred_next[50:75] - real_current[50:75]
    axes[1].plot(dU_real, z_levels_25, 'xb', label="Real ΔU")
    axes[1].plot(dU_pred, z_levels_25, 'o--r', label="Pred ΔU")
    axes[1].set_xlabel("ΔU [m/s]", fontsize=13)
    axes[1].set_title("U Difference (Next − Current)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f"Single-Step Profile @ t={t_idx}", fontsize=15)
    plt.tight_layout()
    plt.savefig(f"plots/profile_step_{t_idx}.png")
    plt.close(fig)

print(f"Saved {NUM_PROFILE_SAMPLES} profile plots to plots/")
