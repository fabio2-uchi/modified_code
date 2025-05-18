# ── Paths ─────────────────────────────────────────────────────────────────
ORIG_PATH      = "/home/fabio/work/HM_and_AI_models/VAE_Model/x_stoch.npy"
SHUFFLED_PATH  = "/home/fabio/work/HM_and_AI_models/VAE_Model/combined_shuffled_3.npy"

# ── Constants ─────────────────────────────────────────────────────────────
TRAIN_N   = 200_000
VAL_N     = 50_000
CHANNEL   = 1            # take column‑index 1 of the 2‑channel data
PLOT_COL  = 63           # column to eyeball

# ──────────────────────────────────────────────────────────────────────────
# 1. load *original* series and compute normalisation
# ──────────────────────────────────────────────────────────────────────────
psi_raw = np.load(ORIG_PATH)[:, CHANNEL, :]          # shape ≈ (300 k, 75)

mean_psi = psi_raw.mean(axis=0, keepdims=True)
std_psi  = psi_raw.std(axis=0,  keepdims=True)
psi      = (psi_raw - mean_psi) / std_psi            # normalised copy

# ──────────────────────────────────────────────────────────────────────────
# 2. build the lead‑row dictionary  x(t) → x(t+1)
#    (omit final row; it has no successor)
# ──────────────────────────────────────────────────────────────────────────
lead_dict: dict[bytes, np.ndarray] = {
    psi[i].tobytes(): psi[i + 1]
    for i in range(len(psi) - 1)
}

# ──────────────────────────────────────────────────────────────────────────
# 3. load the PRE‑SHUFFLED data, take same channel, normalise identically
# ──────────────────────────────────────────────────────────────────────────
shuf_raw   = np.load(SHUFFLED_PATH)[:, CHANNEL, :]
shuf       = (shuf_raw - mean_psi) / std_psi        # shape (N*, 75)

#  -- recover x(t+1) rows via the dictionary ------------------------------
try:
    shuf_lead = np.stack([lead_dict[row.tobytes()] for row in shuf], axis=0)
except KeyError as e:          # only happens if the very last original row
    raise RuntimeError("Shuffled set contains a row with no successor "
                       "(probably the final row of the original series).") from e

assert shuf_lead.shape == shuf.shape

# ──────────────────────────────────────────────────────────────────────────
# 4. split into train / validation tensors
# ──────────────────────────────────────────────────────────────────────────
psi_train_input = torch.from_numpy(shuf[:TRAIN_N])
psi_train_label = torch.from_numpy(shuf_lead[:TRAIN_N])

psi_val_input   = torch.from_numpy(shuf[TRAIN_N:TRAIN_N + VAL_N])
psi_val_label   = torch.from_numpy(shuf_lead[TRAIN_N:TRAIN_N + VAL_N])

# ──────────────────────────────────────────────────────────────────────────
# 5. quick sanity prints / plots
# ──────────────────────────────────────────────────────────────────────────
print("train X:", psi_train_input.shape,
      "train y:", psi_train_label.shape)
print("val   X:", psi_val_input.shape,
      "val   y:", psi_val_label.shape)

plt.figure(figsize=(8, 3))
plt.plot(psi_train_input[:TRAIN_N, PLOT_COL], label="train")
plt.plot(psi_val_input[:VAL_N,   PLOT_COL], label="val")
plt.title("Normalised channel‑%d col‑%d" % (CHANNEL, PLOT_COL))
plt.legend(); plt.tight_layout(); plt.show()
