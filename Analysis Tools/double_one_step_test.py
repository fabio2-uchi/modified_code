"""
double_one_step_test.py

Compare one-step predictions of zonal wind (Delta U) from:
  - The Holton-Mass (HM) stochastic model
  - The VAE emulator

Toggle RUN_HM and RUN_EMULATOR at the top to control which models run.
Both produce ensemble predictions with mean +/- 2sigma error bars.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================================
# TOGGLE: Choose which models to run
# ============================================================================
RUN_HM = True
RUN_EMULATOR = True
NUM_SAMPLES = 1000  # Ensemble size for uncertainty quantification
NUM_COMPARISONS = 10  # Number of different one-step comparisons to generate
SEED = 42  # Master seed for reproducibility

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = '/home/fabio/work/HM_and_AI_models/VAE_Model/data/long_run_310k.npy'
EMU_MODEL_PATH = '/home/fabio/work/HM_and_AI_models/VAE_Model/data/best_weights/checkpoint_11'
SAVE_DIR = os.path.join(BASE_DIR, f'one_step_{NUM_COMPARISONS}_preds')
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================================
# PHYSICAL SCALES
# ============================================================================
LENGTH_SCALE = 2.5e5        # [m]
TIME_SCALE = 24 * 3600.0    # [s]
VELOCITY_SCALE = LENGTH_SCALE / TIME_SCALE  # ~2.89 m/s

NZ = 26
VERTICAL_LEVELS = np.linspace(0, 70e3, NZ + 1)[1:-1] / 1000  # Interior levels [km]

# Thresholds for equilibrium states (non-dimensional)
UPPER_BOUND = 53.8 / VELOCITY_SCALE
LOWER_BOUND = 7.41 / VELOCITY_SCALE

# ============================================================================
# LOAD DATA
# ============================================================================
np.random.seed(SEED)
print("Loading data...")
real_data = np.load(DATA_PATH)   # (N, 2, 75)
psi = real_data[:, 1, :]         # Trajectory 1, shape (N, 75)
zonal_wind = psi[:, 63]          # U at reference altitude (non-dim)

# Normalization stats (needed for emulator input)
mean_psi = np.mean(psi, axis=0)
std_psi = np.std(psi, axis=0)

# ============================================================================
# PRE-LOAD MODELS (once, outside the loop)
# ============================================================================
valid = np.arange(len(psi) - 1)  # Exclude last index (need valid next step)

hm_model = None
if RUN_HM:
    sys.path.insert(0, BASE_DIR)
    from holton_mass import HoltonMassModel

    print("Initializing HM model...")
    hm_model = HoltonMassModel(HoltonMassModel.default_config())
    t_save = np.array([0.0, 1.0])  # One time step

vae = None
if RUN_EMULATOR:
    import torch
    import torch.nn as nn
    torch.manual_seed(SEED)

    LATENT_DIM = 32
    OUTPUT_DIM = 75
    CONDITION_DIM = 50
    NUM_NEURONS = 1024

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(OUTPUT_DIM, NUM_NEURONS)
            self.fc2 = nn.Linear(NUM_NEURONS, NUM_NEURONS)
            self.fc3 = nn.Linear(NUM_NEURONS, NUM_NEURONS)
            self.fc4 = nn.Linear(NUM_NEURONS, NUM_NEURONS)
            self.fc5 = nn.Linear(NUM_NEURONS, NUM_NEURONS)
            self.fc6 = nn.Linear(NUM_NEURONS, NUM_NEURONS)
            self.fc_mu = nn.Linear(NUM_NEURONS, LATENT_DIM)
            self.fc_logvar = nn.Linear(NUM_NEURONS, LATENT_DIM)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x)) + x
            x = torch.relu(self.fc3(x)) + x
            x = torch.relu(self.fc4(x)) + x
            x = torch.relu(self.fc5(x)) + x
            x = torch.relu(self.fc6(x)) + x
            return self.fc_mu(x), self.fc_logvar(x)

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(LATENT_DIM + CONDITION_DIM, NUM_NEURONS)
            self.fc2 = nn.Linear(NUM_NEURONS, NUM_NEURONS)
            self.fc3 = nn.Linear(NUM_NEURONS, NUM_NEURONS)
            self.fc4 = nn.Linear(NUM_NEURONS, NUM_NEURONS)
            self.fc5 = nn.Linear(NUM_NEURONS, NUM_NEURONS)
            self.fc6 = nn.Linear(NUM_NEURONS, NUM_NEURONS)
            self.fc_output = nn.Linear(NUM_NEURONS, OUTPUT_DIM)

        def forward(self, z, cond):
            x = torch.cat((z, cond), dim=1)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x)) + x
            x = torch.relu(self.fc3(x)) + x
            x = torch.relu(self.fc4(x)) + x
            x = torch.relu(self.fc5(x)) + x
            x = torch.relu(self.fc6(x)) + x
            return self.fc_output(x)

    class ConditionalVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Encoder()
            self.decoder = Decoder()

        def decode(self, z, cond):
            return self.decoder(z, cond)

        def forward(self, x, cond):
            mu, logvar = self.encoder(x)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
            return self.decode(z, cond), mu, logvar

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae = ConditionalVAE().to(device)

    if os.path.exists(EMU_MODEL_PATH):
        vae.load_state_dict(torch.load(EMU_MODEL_PATH, map_location=device, weights_only=True))
        vae.eval()
        print("Emulator model loaded.")
    else:
        print(f"WARNING: checkpoint not found at {EMU_MODEL_PATH}, skipping emulator.")
        RUN_EMULATOR = False

# ============================================================================
# MAIN LOOP: Generate NUM_COMPARISONS different one-step comparisons
# ============================================================================
for comp_idx in range(NUM_COMPARISONS):
    print(f"\n{'='*60}")
    print(f"  Comparison {comp_idx + 1}/{NUM_COMPARISONS}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # SELECT INITIAL CONDITIONS (one from each equilibrium basin)
    # ------------------------------------------------------------------
    ic_A = np.random.choice(valid[zonal_wind[valid] > UPPER_BOUND])
    ic_B = np.random.choice(valid[zonal_wind[valid] < LOWER_BOUND])

    state_A, next_A = psi[ic_A], psi[ic_A + 1]
    state_B, next_B = psi[ic_B], psi[ic_B + 1]

    print(f"IC A: index={ic_A}, U(ref)={zonal_wind[ic_A] * VELOCITY_SCALE:.2f} m/s")
    print(f"IC B: index={ic_B}, U(ref)={zonal_wind[ic_B] * VELOCITY_SCALE:.2f} m/s")

    # ------------------------------------------------------------------
    # HM ONE-STEP PREDICTIONS
    # ------------------------------------------------------------------
    hm_pred = {}
    if RUN_HM:
        for label, state, seed in [('A', state_A, None), ('B', state_B, None)]:
            x0 = np.tile(state, (NUM_SAMPLES, 1))  # (NUM_SAMPLES, 75)
            x_ens = hm_model.integrate_euler_maruyama(x0, t_save, seed=seed)
            hm_pred[label] = x_ens[1]  # (NUM_SAMPLES, 75)
            print(f"  HM ensemble for state {label}: done")

    # ------------------------------------------------------------------
    # EMULATOR ONE-STEP PREDICTIONS
    # ------------------------------------------------------------------
    emu_pred = {}
    if RUN_EMULATOR:
        for label, state in [('A', state_A), ('B', state_B)]:
            x_norm = ((state - mean_psi) / std_psi)[:CONDITION_DIM].astype(np.float32)
            cond = torch.tensor(x_norm).unsqueeze(0).expand(NUM_SAMPLES, -1).to(device)
            z = torch.randn(NUM_SAMPLES, LATENT_DIM, device=device)
            with torch.no_grad():
                y = vae.decode(z, cond).cpu().numpy()
            emu_pred[label] = y * std_psi + mean_psi  # Denormalize -> (NUM_SAMPLES, 75)
            print(f"  Emulator ensemble for state {label}: done")

    # ------------------------------------------------------------------
    # PLOT: Delta-U profiles with error bars
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    panels = [
        (axes[0], 'A', state_A, next_A, 'Equilibrium Point A'),
        (axes[1], 'B', state_B, next_B, 'Equilibrium Point B'),
    ]

    for ax, label, state, truth, title in panels:
        U_current = state[50:75] * VELOCITY_SCALE
        delta_truth = truth[50:75] * VELOCITY_SCALE - U_current

        # HM ensemble
        if RUN_HM and label in hm_pred:
            delta = hm_pred[label][:, 50:75] * VELOCITY_SCALE - U_current
            mu, sigma = delta.mean(axis=0), delta.std(axis=0)
            ax.plot(mu, VERTICAL_LEVELS, 'o', color='tab:green',
                    label=r'HM $\Delta U$', markersize=5, alpha=0.8)
            for i in range(len(VERTICAL_LEVELS)):
                ax.plot([mu[i] - 2*sigma[i], mu[i] + 2*sigma[i]],
                        [VERTICAL_LEVELS[i]]*2,
                        color='tab:green', linewidth=1.5, alpha=0.6)

        # Emulator ensemble
        if RUN_EMULATOR and label in emu_pred:
            delta = emu_pred[label][:, 50:75] * VELOCITY_SCALE - U_current
            mu, sigma = delta.mean(axis=0), delta.std(axis=0)
            ax.plot(mu, VERTICAL_LEVELS, 's', color='tab:red',
                    label=r'Emulator $\Delta U$', markersize=5, alpha=0.8)
            for i in range(len(VERTICAL_LEVELS)):
                ax.plot([mu[i] - 2*sigma[i], mu[i] + 2*sigma[i]],
                        [VERTICAL_LEVELS[i]]*2,
                        color='tab:red', linewidth=1.5, alpha=0.6)

        ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel(r'$\Delta U$ [m/s]', fontsize=14)
        ax.set_ylabel('Altitude [km]', fontsize=14)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(labelsize=11)

    plt.tight_layout()
    savepath = os.path.join(SAVE_DIR, f'one_step_comparison_{comp_idx + 1:02d}.png')
    fig.savefig(savepath, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)
    print(f"Plot saved to {savepath}")

    # ------------------------------------------------------------------
    # DIAGNOSTICS
    # ------------------------------------------------------------------
    for label, truth in [('A', next_A), ('B', next_B)]:
        U_truth = truth[50:75] * VELOCITY_SCALE
        print(f"\n--- State {label} ---")
        if RUN_HM and label in hm_pred:
            U_hm_mean = hm_pred[label][:, 50:75].mean(axis=0) * VELOCITY_SCALE
            print(f"  HM      RMSE: {np.sqrt(np.mean((U_hm_mean - U_truth)**2)):.4f} m/s")
        if RUN_EMULATOR and label in emu_pred:
            U_emu_mean = emu_pred[label][:, 50:75].mean(axis=0) * VELOCITY_SCALE
            print(f"  Emulator RMSE: {np.sqrt(np.mean((U_emu_mean - U_truth)**2)):.4f} m/s")

print(f"\nAll {NUM_COMPARISONS} comparisons saved to {SAVE_DIR}")
