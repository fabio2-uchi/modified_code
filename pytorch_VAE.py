import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

# Set device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
torch.cuda.empty_cache()

# Model weights path
model_weights_path = r"/home/constantino-daniel-boscu/Documents/research/AI-RES/modified-code-main3/CNN_models"

# Load and preprocess data
F_data = np.load(r'/home/constantino-daniel-boscu/Documents/research/AI-RES/modified-code-main3/x_stoch.npy')
psi = F_data[3500:, 0, :]

# Normalize data
mean_psi = np.mean(psi, axis=0, keepdims=True)
std_psi = np.std(psi, axis=0, keepdims=True)
psi = (psi - mean_psi) / std_psi

# Data preparation
train_size = 200000
val_size = 50000
test_time = 15000
lead = 1

train_end = train_size
val_start = train_end
val_end = val_start + val_size

def reshape_data(data):
    return data.reshape(-1, 75)

psi_input_Tr = reshape_data(psi[:train_end, :])
psi_label_Tr = reshape_data(psi[:train_end, :])
psi_input_val = reshape_data(psi[val_start:val_end, :])
psi_label_val = reshape_data(psi[val_start + lead:val_end + lead, :])
test_size = int(0.1 * psi.shape[0])
psi_test_input = reshape_data(psi[val_end:val_end + test_size, :])
psi_test_label = reshape_data(psi[val_end + lead:val_end + lead + test_size, :])

class AtmosphereDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

# Data loaders
train_dataset = AtmosphereDataset(psi_input_Tr, psi_label_Tr)
val_dataset = AtmosphereDataset(psi_input_val, psi_label_val)
test_dataset = AtmosphereDataset(psi_test_input, psi_test_label)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# Define the VAE model
class VariationalEncoder(nn.Module):
    def __init__(self):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(75, 75)
        self.linear2 = nn.Linear(75, 64)
        self.linear3 = nn.Linear(64, 64)
        self.linear4 = nn.Linear(64, 64)
        self.kl = 0
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten to (batch, 75)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        mu = self.linear3(x)
        log_var = self.linear4(x)
        std = torch.exp(0.5 * log_var)
        noise = torch.randn_like(std)
        z = mu + std * noise
        
        # KL divergence
        self.kl = (-0.5 * torch.mean(1 + log_var - mu.pow(2) - torch.exp(log_var)))
        return z

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(64, 64)
        self.linear2 = nn.Linear(64, 75)
        self.linear3 = nn.Linear(75, 75)
        
    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = F.relu(self.linear2(z))
        z = self.linear3(z) # should be (batch, 75)
        return z

class VariationalAutoencoder(nn.Module):
    def __init__(self):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder()
        self.decoder = Decoder()
        
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

vae = VariationalAutoencoder().to(device)

def train_model(model, train_loader, val_loader, epochs=5):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            x_hat = model(x)

            recon_loss = F.mse_loss(x_hat, y, reduction='mean')
            kl_loss = model.encoder.kl
            loss = recon_loss + kl_loss * 0.5
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        print(f'Epoch {epoch+1} Training Loss: {avg_train_loss:.6f}')
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                x_hat = model(x)
                recon_loss = F.mse_loss(x_hat, y, reduction='mean')
                kl_loss = model.encoder.kl
                loss = recon_loss + kl_loss
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch {epoch+1} Validation Loss: {avg_val_loss:.6f}')
        
        # Saving the model
        torch.save(model.state_dict(), os.path.join(model_weights_path, 'model_weights.pth'))
    return model

# Load model weights if available; otherwise, train the model
model_file = os.path.join(model_weights_path, 'model_weights.pth')
if os.path.exists(model_file):
    vae.load_state_dict(torch.load(model_file, map_location=device))
    print(f"Model weights loaded from {model_weights_path}.")
else:
    print("No pre-trained weights found. Training model...")
    vae = train_model(vae, train_loader, val_loader, epochs=5)
    print(f"Model weights saved to {model_weights_path}/model_weights.pth.")

# ---------------- Inference ----------------
vae.eval()
pred_mean = np.zeros((test_time, 75, 1))
std_psi = std_psi.reshape(1, 75)
mean_psi = mean_psi.reshape(1, 75)

with torch.no_grad():
    prev = torch.tensor(psi_test_input[:1, :].reshape(1, 75), dtype=torch.float32).to(device)
    
    for k in tqdm(range(test_time), desc="Inference Progress"):
        pred_step = vae(prev)
        pred_step_np = pred_step.cpu().numpy()  # shape: (1, 75)
        pred_mean[k, :] = pred_step_np.reshape(75, 1) # turn it into (75, 1)
        
        # Denormalize the prediction before feeding it back
        pred_denorm = (pred_step_np * std_psi + mean_psi).reshape(1, 75)
        
        # Normalize for the next input
        next_input = (pred_denorm - mean_psi) / std_psi
        prev = torch.tensor(next_input, dtype=torch.float32).to(device)

# Denormalize final preds
pred_mean = pred_mean.squeeze() * std_psi.reshape(1, -1) + mean_psi.reshape(1, -1)
pred_mean = pred_mean.reshape(test_time, 75, 1)

# Denormalize test labels
actual_values = psi_test_label[:test_time, :].squeeze()
actual_values = actual_values * std_psi + mean_psi

print(f"Shape of actual_values after denormalization: {actual_values.shape}")

# Calculate MSE
pred_flat = pred_mean.reshape(test_time, 75)
actual_flat = actual_values
mse_value = mean_squared_error(actual_flat, pred_flat)
print(f"\nMean Squared Error: {mse_value}")

# Plot predictions vs actual for index 63
zonal_wind_idx = 63
plt.figure(figsize=(15, 10))
plt.plot(actual_values[:, zonal_wind_idx], label="Actual", color="blue")
plt.plot(pred_mean[:, zonal_wind_idx], label="Predicted", linestyle="dashed", color="red")
plt.title("Predictions vs Actual at Zonal Wind Index")
plt.xlabel("Time Step")
plt.ylabel("Zonal Wind Speed")
plt.legend()
plt.grid(True)
plt.savefig('predictions_vs_actual.png')
plt.show()

# Save results (predictions, mean, std, and actual values)
np.savez(r'/home/constantino-daniel-boscu/Documents/research/AI-RES/modified-code-main3/3ep-jan12th/model_weights.weights.h5',
         predictions=pred_mean, mean_psi=mean_psi, std_psi=std_psi, actual_values=actual_values)
