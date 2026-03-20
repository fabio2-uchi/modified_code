"""
train.py — Training loop with KL annealing and logging

Trains the Conditional VAE on the Holton-Mass stochastic simulation data.
Uses cyclic KL annealing, Huber (smooth L1) reconstruction loss, mixed-precision
training, and model selection based on long-term climatological metrics.
"""

import os
import sys
import datetime
import shutil
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
from tqdm import tqdm
from scipy.stats import linregress

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast

import wandb

from model import ConditionalVAE
from transitions import calculate_transition_durations

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================
DATA_PATH = os.environ.get(
    "HM_DATA_PATH", "long_run_310k.npy"
)

psi_raw = np.load(DATA_PATH)          # (N, 2, 75)
psi = psi_raw[:, 1, :]                # trajectory from fixed point B
mean_psi = np.mean(psi, axis=0, keepdims=True)
std_psi = np.std(psi, axis=0, keepdims=True)
psi_norm = (psi - mean_psi) / std_psi

# Conditioning: use streamfunction Psi (first 50 components)
LEAD = 1
TRAIN_N = 250000
VAL_N = 50000
CONDITION_DIM = 50   # Re{Psi} + Im{Psi}
OUTPUT_DIM = 75
VARIABLE_RANGE = (0, CONDITION_DIM - 1)  # indices 0..49

np.random.seed(42)
valid_indices = np.arange(0, TRAIN_N - LEAD)
shuffled_indices = np.random.permutation(valid_indices)

start, end = VARIABLE_RANGE[0], VARIABLE_RANGE[1] + 1
psi_train_input = torch.tensor(psi_norm[shuffled_indices, start:end])
psi_train_label = torch.tensor(psi_norm[shuffled_indices + LEAD, :])
psi_val_input = torch.tensor(psi_norm[TRAIN_N:TRAIN_N + VAL_N, start:end])
psi_val_label = torch.tensor(psi_norm[TRAIN_N + LEAD:TRAIN_N + VAL_N + LEAD, :])


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def frange_cycle_linear(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    """Cyclic linear KL annealing schedule."""
    L = np.ones(n_epoch)
    period = n_epoch / n_cycle
    step = (stop - start) / (period * ratio)
    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i + c * period) < n_epoch):
            L[int(i + c * period)] = v
            v += step
            i += 1
    return L


def normalize_transition_time(slope_value, delta, transition_real):
    return 1 - np.exp(-np.abs((slope_value - transition_real)) / delta)


def crps_score(predictions, actual):
    actual = actual.unsqueeze(0)
    mae = torch.cdist(actual, predictions, 1).mean()
    ens_var = torch.cdist(predictions, predictions, 1).mean()
    return mae - 0.5 * ens_var


def inference(model, psi_data, mean_psi, std_psi, time_step, latent_dim):
    """Autoregressive inference for long-term evaluation."""
    start, end = VARIABLE_RANGE[0], VARIABLE_RANGE[1] + 1
    num_vars = end - start
    initial_cond = torch.tensor(psi_data[0, start:end]).reshape(1, num_vars)
    z = torch.zeros(1, latent_dim)
    pred = np.zeros((time_step, OUTPUT_DIM))

    for k in tqdm(range(time_step), desc="Inference"):
        with torch.inference_mode():
            model.eval()
            with autocast(device_type='cuda'):
                z = torch.randn_like(z).float().cuda(non_blocking=True)
                if k == 0:
                    initial_cond = initial_cond.float().cuda(non_blocking=True)
                else:
                    initial_cond = torch.tensor(
                        pred[k - 1, start:end]
                    ).reshape(1, num_vars).float().cuda(non_blocking=True)

                y = model.decode(z, initial_cond).detach().cpu().numpy()
                pred[k, :] = y

    return pred


# ============================================================================
# PLOTTING HELPERS (used during training)
# ============================================================================
def Timeseries_plot(actual, pred, epoch, ax):
    ax.plot(actual, 'b', label='Actual')
    ax.plot(pred, 'r', label='Predictions')
    ax.set_title(f"Timeseries | Epoch {epoch}", fontsize=16)
    ax.set_xlabel('Time Step', fontsize=14)
    ax.set_ylabel('Zonal Wind Value', fontsize=14)
    ax.legend()
    ax.grid(True)


def PDF_plot(actual, pred, epoch, pdf_distance, ax):
    sns.histplot(actual, bins=50, kde=True, color='black', alpha=0.6,
                 element='step', label='Real Data', ax=ax)
    sns.histplot(pred, bins=50, kde=True, color='red', alpha=0.6,
                 element='step', label='Predictions', ax=ax)
    ax.set_title(f"PDFs | Epoch {epoch} | KL Error: {pdf_distance:.4f}", fontsize=16)
    ax.set_xlabel('Zonal Wind (m/s)', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.axvline(np.mean(actual), color='black', linestyle='--',
               label=f'Real Mean: {np.mean(actual):.2f}')
    ax.axvline(np.mean(pred), color='red', linestyle='--',
               label=f'Pred Mean: {np.mean(pred):.2f}')
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)


def Exp_fit_plot(x_pred, y_pred, x_real, y_real, fit_pred, fit_real,
                 epoch, exp_dist, range_dist, ax):
    ax.plot(x_pred, y_pred, 'r-',
            label=f'Pred Exp Fit (slope={fit_pred:.4f})', linewidth=2)
    ax.plot(x_real, y_real, 'b-',
            label=f'Real Exp Fit (slope={fit_real:.4f})', linewidth=2)
    ax.set_xlabel('Time Duration (Steps)')
    ax.set_ylabel('Exponential Fit')
    ax.set_title(f"Exp Fits | Epoch {epoch} | Exp Err: {exp_dist:.4f} "
                 f"| Range Err: {range_dist:.4f}", fontsize=16)
    ax.grid()
    ax.legend()


def all_plot(actual, pred, x_pred, y_pred, x_real, y_real,
             fit_pred, fit_real, pdf_dist, exp_dist, range_dist, epoch, folder):
    fig = plt.figure(figsize=(24, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig)
    ax_ts = fig.add_subplot(gs[:, 0])
    ax_pdf = fig.add_subplot(gs[0, 1])
    ax_exp = fig.add_subplot(gs[1, 1])

    Timeseries_plot(actual, pred, epoch, ax_ts)
    PDF_plot(actual, pred, epoch, pdf_dist, ax_pdf)
    Exp_fit_plot(x_pred, y_pred, x_real, y_real,
                 fit_pred, fit_real, epoch, exp_dist, range_dist, ax_exp)

    distance = np.sqrt(pdf_dist**2 + exp_dist**2 + range_dist**2)
    fig.suptitle(f"Epoch {epoch} | Euclidean Error: {distance:.4f}", fontsize=20)
    plt.subplots_adjust(wspace=0.2, hspace=0.35)
    fig.tight_layout(pad=2.0)
    os.makedirs(os.path.join(folder, "plots"), exist_ok=True)
    plt.savefig(os.path.join(folder, f"plots/all_plots_epoch_{epoch}.png"))
    plt.close(fig)


# ============================================================================
# MAIN TRAINING
# ============================================================================
if __name__ == "__main__":

    # --- Hyperparameters ---
    LATENT_DIM = 32
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 1500
    BATCH_SIZE = 1024
    TIME_STEP_EVAL = 40000
    LEVEL = 63
    UPPER_BOUND = 53.8 / (2.5e5 / 86400.0)
    LOWER_BOUND = 7.41 / (2.5e5 / 86400.0)

    beta_kl_coef = frange_cycle_linear(0.01, 0.3, NUM_EPOCHS, n_cycle=1, ratio=1.0)

    # --- W&B ---
    run = wandb.init(
        project="ssw_research",
        config={
            "architecture": "ResNet-CVAE",
            "dataset": "Holton-Mass",
            "learning_rate": LEARNING_RATE,
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "latent_dim": LATENT_DIM,
            "neurons": 1024,
        },
    )

    # --- Reference statistics from real data ---
    real_data_1d = psi_raw[:, 1, LEVEL]
    real_durations = calculate_transition_durations(real_data_1d, UPPER_BOUND, LOWER_BOUND)
    real_data_sorted = np.sort(real_durations)
    transition_real = np.mean(real_data_sorted)
    slope_real, _, *_ = linregress(
        real_data_sorted[1 - np.arange(1, len(real_data_sorted) + 1) / len(real_data_sorted) > 0],
        np.log((1 - np.arange(1, len(real_data_sorted) + 1) / len(real_data_sorted))[
            1 - np.arange(1, len(real_data_sorted) + 1) / len(real_data_sorted) > 0
        ])
    )
    print(f"Reference transition mean: {transition_real:.1f} days")
    print(f"Reference CCDF slope: {slope_real:.6f}")

    # --- Model & optimizer ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConditionalVAE(LATENT_DIM, OUTPUT_DIM, CONDITION_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler()

    master_folder = f"save_folder/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(master_folder, exist_ok=True)

    # --- Training loop ---
    for epoch in range(NUM_EPOCHS):
        model.train()
        for batch_start in tqdm(range(0, TRAIN_N, BATCH_SIZE),
                                desc=f"Train epoch {epoch + 1}"):
            inp = psi_train_input[batch_start:batch_start + BATCH_SIZE].float().to(device)
            lbl = psi_train_label[batch_start:batch_start + BATCH_SIZE].float().to(device)

            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                output, mu, logvar = model(lbl, inp)
                rec_loss = F.smooth_l1_loss(output, lbl, reduction="mean")
                kl_loss = 0.5 * (mu**2 + torch.exp(logvar) - 1 - logvar).sum()
                loss = rec_loss + beta_kl_coef[epoch] * kl_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # --- Validation ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_start in range(0, VAL_N, BATCH_SIZE):
                inp = psi_val_input[batch_start:batch_start + BATCH_SIZE].float().to(device)
                lbl = psi_val_label[batch_start:batch_start + BATCH_SIZE].float().to(device)
                with autocast(device_type='cuda'):
                    output, mu, logvar = model(lbl, inp)
                    val_rec = F.smooth_l1_loss(output, lbl, reduction="mean")
                    val_kl = 0.5 * (mu**2 + torch.exp(logvar) - 1 - logvar).sum()
                    val_losses.append((val_rec + beta_kl_coef[epoch] * val_kl).item())

        avg_val = np.mean(val_losses)
        run.log({"Loss": rec_loss.item(), "KL-Loss": kl_loss.item(),
                 "Val Loss": avg_val}, step=epoch + 1)

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}  "
              f"rec={rec_loss.item():.5f}  kl={kl_loss.item():.5f}  "
              f"val={avg_val:.5f}")

        # --- Save checkpoint ---
        ckpt_path = os.path.join(master_folder, f"checkpoint_{epoch + 1}")
        torch.save(model.state_dict(), ckpt_path)

    run.finish()
    print(f"Training complete. Checkpoints in {master_folder}")
