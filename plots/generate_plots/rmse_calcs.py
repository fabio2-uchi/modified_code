"""
rmse_calcs.py

Compute forecast error growth (RMSE vs lead time) for the VAE emulator.
Iteratively forecasts from many initial conditions and compares to ground truth.
Includes climatological RMSE as a skill baseline.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = '/home/fabio/work/HM_and_AI_models/VAE_Model/data/long_run_310k.npy'
EMU_MODEL_PATH = '/home/fabio/work/HM_and_AI_models/VAE_Model/data/best_weights/checkpoint_11'
SAVE_DIR = BASE_DIR

LENGTH_SCALE = 2.5e5        # [m]
TIME_SCALE = 24 * 3600.0    # [s]
VELOCITY_SCALE = LENGTH_SCALE / TIME_SCALE  # ~2.89 m/s

LATENT_DIM = 32
OUTPUT_DIM = 75
CONDITION_DIM = 50
NUM_NEURONS = 1024

NUM_FORECAST_DAYS = 4000
NUM_ENSEMBLE = 50
NUM_ICS = 80

# ============================================================================
# VAE MODEL DEFINITION
# ============================================================================
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

# ============================================================================
# ITERATIVE FORECAST FUNCTION
# ============================================================================
def run_iterative_forecast(vae, initial_cond_50, num_days, num_ensemble, device):
    """
    Iteratively forecast num_days steps using the VAE emulator.

    Args:
        vae: Loaded ConditionalVAE model.
        initial_cond_50: (1, 50) tensor, normalized condition for Day 0.
        num_days: Number of days to forecast.
        num_ensemble: Number of ensemble members per step.
        device: torch device.

    Returns:
        predictions: (num_days, num_ensemble, 75) in normalized units.
    """
    predictions = np.zeros((num_days, num_ensemble, OUTPUT_DIM))
    current_cond = initial_cond_50.clone()

    for day in range(num_days):
        with torch.no_grad():
            z = torch.randn(num_ensemble, LATENT_DIM, device=device)
            cond_expanded = current_cond.expand(num_ensemble, -1).to(device)
            y = vae.decode(z.float(), cond_expanded.float())
            y_np = y.cpu().numpy()
            predictions[day] = y_np
            # Recurse: feed ensemble mean back as next input
            mean_pred = np.mean(y_np, axis=0, keepdims=True)
            current_cond = torch.tensor(mean_pred[:, :CONDITION_DIM]).float()

    return predictions

# ============================================================================
# LOAD DATA AND MODEL
# ============================================================================
print("Loading data...")
real_data = np.load(DATA_PATH)  # (N, 2, 75)
psi = real_data[:, 1, :]        # Trajectory 1, shape (N, 75)

mean_psi = np.mean(psi, axis=0)
std_psi = np.std(psi, axis=0)

# Climatological mean of zonal wind [m/s]
U_climatology = mean_psi[50:75] * VELOCITY_SCALE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vae = ConditionalVAE().to(device)
vae.load_state_dict(torch.load(EMU_MODEL_PATH, map_location=device, weights_only=True))
vae.eval()
print("Emulator model loaded.")

# ============================================================================
# SELECT INITIAL CONDITIONS
# ============================================================================
max_valid_idx = len(real_data) - NUM_FORECAST_DAYS - 1
selected_indices = np.random.choice(max_valid_idx, NUM_ICS, replace=False)

print(f"\nConfiguration:")
print(f"  Forecast horizon: {NUM_FORECAST_DAYS} days")
print(f"  Initial conditions: {NUM_ICS}")
print(f"  Ensemble members: {NUM_ENSEMBLE}")

# ============================================================================
# MAIN LOOP: FORECAST RMSE + CLIMATOLOGICAL RMSE
# ============================================================================
all_forecast_rmse = []
all_clim_rmse = []

for i, idx in enumerate(selected_indices):
    if i % 50 == 0:
        print(f"Processing IC {i}/{NUM_ICS}...", end='\r')

    # Ground truth: zonal wind for days 1..400 [m/s]
    truth_U = real_data[idx + 1 : idx + NUM_FORECAST_DAYS + 1, 1, 50:75] * VELOCITY_SCALE

    # Prepare normalized IC for emulator
    raw_ic = real_data[idx, 1, :]
    ic_norm = (raw_ic - mean_psi) / std_psi
    initial_cond_50 = torch.tensor(ic_norm[:CONDITION_DIM]).float().unsqueeze(0)

    # Run iterative forecast -> (400, ensemble, 75) normalized
    preds_norm = run_iterative_forecast(vae, initial_cond_50, NUM_FORECAST_DAYS, NUM_ENSEMBLE, device)

    # Denormalize and extract zonal wind [m/s]
    preds_U = (preds_norm * std_psi + mean_psi)[:, :, 50:75] * VELOCITY_SCALE

    # Ensemble mean prediction [m/s]
    preds_mean_U = preds_U.mean(axis=1)  # (400, 25)

    # Forecast RMSE at each lead time (RMS over 25 altitude levels)
    forecast_rmse = np.sqrt(np.mean((preds_mean_U - truth_U) ** 2, axis=1))  # (400,)
    all_forecast_rmse.append(forecast_rmse)

    # Climatological RMSE: how far is climatology from truth at each lead time
    clim_rmse = np.sqrt(np.mean((U_climatology - truth_U) ** 2, axis=1))  # (400,)
    all_clim_rmse.append(clim_rmse)

print(f"\nProcessing complete!")

all_forecast_rmse = np.array(all_forecast_rmse)  # (NUM_ICS, 400)
all_clim_rmse = np.array(all_clim_rmse)          # (NUM_ICS, 400)

# ============================================================================
# PLOT
# ============================================================================
time_days = np.arange(1, NUM_FORECAST_DAYS + 1)

# Forecast RMSE statistics
forecast_median = np.median(all_forecast_rmse, axis=0)
forecast_q25 = np.percentile(all_forecast_rmse, 25, axis=0)
forecast_q75 = np.percentile(all_forecast_rmse, 75, axis=0)
forecast_p2_5 = np.percentile(all_forecast_rmse, 2.5, axis=0)
forecast_p97_5 = np.percentile(all_forecast_rmse, 97.5, axis=0)

# Climatological RMSE statistics
clim_median = np.median(all_clim_rmse, axis=0)
clim_q25 = np.percentile(all_clim_rmse, 25, axis=0)
clim_q75 = np.percentile(all_clim_rmse, 75, axis=0)

fig, ax = plt.subplots(figsize=(10, 6))

# Forecast RMSE
ax.fill_between(time_days, forecast_p2_5, forecast_p97_5,
                color='tab:blue', alpha=0.15, label='Forecast 95% CI')
ax.fill_between(time_days, forecast_q25, forecast_q75,
                color='tab:blue', alpha=0.4, label='Forecast IQR')
ax.plot(time_days, forecast_median, color='tab:blue', linewidth=2.5,
        label='Forecast Median RMSE')

# Climatological RMSE
ax.fill_between(time_days, clim_q25, clim_q75,
                color='tab:red', alpha=0.2, label='Climatology IQR')
ax.plot(time_days, clim_median, color='tab:red', linewidth=2, linestyle='--',
        label='Climatology Median RMSE')

ax.set_xlabel('Forecast Lead Time [Days]', fontsize=14)
ax.set_ylabel('Zonal Wind RMSE [m/s]', fontsize=14)
ax.set_title(f'Forecast Error Growth ({NUM_ICS} ICs, {NUM_ENSEMBLE} ensemble members)', fontsize=14)
ax.set_ylim(bottom=0)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=11)
ax.tick_params(labelsize=11)

plt.tight_layout()
savepath = os.path.join(SAVE_DIR, 'rmse_forecast.png')
fig.savefig(savepath, dpi=300, bbox_inches='tight', pad_inches=0.2)
plt.close(fig)

# ============================================================================
# DIAGNOSTICS
# ============================================================================
print(f"\n{'='*60}")
print(f"Forecast RMSE Summary")
print(f"{'='*60}")
print(f"  Day 1 median RMSE:   {forecast_median[0]:.4f} m/s")
print(f"  Day 100 median RMSE: {forecast_median[99]:.4f} m/s")
print(f"  Day 400 median RMSE: {forecast_median[-1]:.4f} m/s")
print(f"  Climatology median:  {np.median(clim_median):.4f} m/s")
print(f"{'='*60}")
print(f"Plot saved to {savepath}")
