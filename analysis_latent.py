"""
analysis_latent.py — PCA, clustering, and latent diagnostics

Encodes Holton-Mass and/or Emulator states through the CVAE encoder,
applies PCA to the latent mean vectors, and visualizes the four dynamical
regime clusters (A, B, A→B, B→A) in the leading principal components.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import torch

from model import ConditionalVAE
from transitions import classify_states, detect_AB_transitions, detect_BA_transitions

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
VELOCITY_SCALE = 2.5e5 / (24 * 3600.0)
U_A = 53.8   # m/s
U_B = 1.75   # m/s
LEVEL = 63
SAMPLES_PER_CLASS = 750

# ============================================================================
# LOAD DATA & MODEL
# ============================================================================
print("Loading data...")
psi_raw = np.load(DATA_PATH)
psi = psi_raw[:, 1, :]
mean_psi = np.mean(psi, axis=0, keepdims=True)
std_psi = np.std(psi, axis=0, keepdims=True)
psi_norm = (psi - mean_psi) / std_psi

# Dimensional zonal wind at reference level
u_dim = psi[:, LEVEL] * VELOCITY_SCALE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConditionalVAE(LATENT_DIM, OUTPUT_DIM, CONDITION_DIM).to(device)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device, weights_only=True))
model.eval()
print(f"Loaded model from {WEIGHTS_PATH}")

# ============================================================================
# CLASSIFY STATES & SAMPLE
# ============================================================================
labels = classify_states(u_dim, u_a=U_A, u_b=U_B)
ab_indices = detect_AB_transitions(u_dim, u_a=U_A, u_b=U_B)
ba_indices = detect_BA_transitions(u_dim, u_a=U_A, u_b=U_B)

idx_A = np.where(labels == 'A')[0]
idx_B = np.where(labels == 'B')[0]

np.random.seed(42)
sample_A = np.random.choice(idx_A, min(SAMPLES_PER_CLASS, len(idx_A)), replace=False)
sample_B = np.random.choice(idx_B, min(SAMPLES_PER_CLASS, len(idx_B)), replace=False)
sample_AB = np.array(ab_indices[:SAMPLES_PER_CLASS]) if len(ab_indices) >= SAMPLES_PER_CLASS else np.array(ab_indices)
sample_BA = np.array(ba_indices[:SAMPLES_PER_CLASS]) if len(ba_indices) >= SAMPLES_PER_CLASS else np.array(ba_indices)

all_indices = np.concatenate([sample_A, sample_B, sample_AB, sample_BA])
all_labels = (
    ['A'] * len(sample_A) +
    ['B'] * len(sample_B) +
    ['AB'] * len(sample_AB) +
    ['BA'] * len(sample_BA)
)

print(f"Samples — A: {len(sample_A)}, B: {len(sample_B)}, "
      f"A→B: {len(sample_AB)}, B→A: {len(sample_BA)}")

# ============================================================================
# ENCODE TO LATENT SPACE
# ============================================================================
print("Encoding states to latent space...")
states = torch.tensor(psi_norm[all_indices]).float().to(device)
with torch.no_grad():
    mu, logvar = model.encode(states)

mu_np = mu.cpu().numpy()
logvar_np = logvar.cpu().numpy()

# ============================================================================
# PCA ON LATENT MEANS
# ============================================================================
pca = PCA(n_components=3)
pc = pca.fit_transform(mu_np)

print(f"Explained variance: PC1={pca.explained_variance_ratio_[0]*100:.2f}%, "
      f"PC2={pca.explained_variance_ratio_[1]*100:.2f}%, "
      f"PC3={pca.explained_variance_ratio_[2]*100:.2f}%")

# ============================================================================
# PLOT: PC1 vs PC2
# ============================================================================
os.makedirs("plots", exist_ok=True)

color_map = {'A': 'tab:red', 'B': 'tab:blue', 'AB': 'tab:green', 'BA': 'tab:purple'}
label_map = {'A': 'State A (strong vortex)', 'B': 'State B (weak vortex)',
             'AB': 'A→B (SSW)', 'BA': 'B→A (recovery)'}

fig, ax = plt.subplots(figsize=(10, 8))
for cls in ['A', 'B', 'AB', 'BA']:
    mask = np.array(all_labels) == cls
    ax.scatter(pc[mask, 0], pc[mask, 1], c=color_map[cls],
               label=label_map[cls], alpha=0.5, s=20)
    centroid = pc[mask].mean(axis=0)
    ax.scatter(centroid[0], centroid[1], c=color_map[cls],
               marker='X', s=200, edgecolors='black', linewidths=1.5)

ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=14)
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)", fontsize=14)
ax.set_title("Latent Space PCA — Holton-Mass States", fontsize=15)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/latent_pca_pc1_pc2.png", dpi=150)
plt.close(fig)
print("Saved plots/latent_pca_pc1_pc2.png")

# ============================================================================
# OPTIONAL: K-Means clustering evaluation
# ============================================================================
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(mu_np)

# Compute cluster purity
label_to_int = {'A': 0, 'B': 1, 'AB': 2, 'BA': 3}
true_ints = np.array([label_to_int[l] for l in all_labels])
purity = 0
for c in range(4):
    mask = cluster_labels == c
    if mask.sum() > 0:
        counts = np.bincount(true_ints[mask], minlength=4)
        purity += counts.max()
purity /= len(all_labels)
print(f"K-Means cluster purity: {purity:.3f}")
