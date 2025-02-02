from os import path
import os
import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.utils.data
import torch.distributions
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 200

# Data Loading and Preprocessing
F = np.load(r'C:\Users\danie\OneDrive\Desktop\Holton Mass Model\Stochastic-VAE-for-Digital-Twins\x_stoch.npy')
psi = F[3500:, 0, :]

# Normalize data
mean_psi = np.mean(psi, axis=0, keepdims=True)
std_psi = np.std(psi, axis=0, keepdims=True)
psi = (psi - mean_psi) / std_psi

# Data preparation
train_size = 100000
val_size = 25000
lead = 1  # Time steps to predict ahead

def reshape_data(data):
    return data.reshape(-1, 75)

# Training, validation, and test sets
train_end = train_size
val_start = train_end
val_end = val_start + val_size

# Prepare input-output pairs
psi_input_Tr = reshape_data(psi[:train_end, :])
psi_label_Tr = reshape_data(psi[lead:train_end + lead, :])
psi_input_val = reshape_data(psi[val_start:val_end, :])
psi_label_val = reshape_data(psi[val_start + lead:val_end + lead, :])

# Test set
test_size = int(0.2 * psi.shape[0])
psi_test_input = reshape_data(psi[val_end:val_end + test_size, :])
psi_test_label = reshape_data(psi[val_end + lead:val_end + lead + test_size, :])

class AtmosphereDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, labels):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

# Data loaders
batch_size = 256
train_dataset = AtmosphereDataset(psi_input_Tr, psi_label_Tr)
val_dataset = AtmosphereDataset(psi_input_val, psi_label_val)
test_dataset = AtmosphereDataset(psi_test_input, psi_test_label)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.input_dims = 75
        self.latent_dims = latent_dims
        
        # Encoder architecture
        self.encoder_hidden = nn.Sequential(
            nn.Linear(self.input_dims, 150),
            nn.ReLU(),
            nn.Linear(150, 100),
            nn.ReLU()
        )
        
        # Mean and log variance layers
        self.fc_mu = nn.Linear(100, latent_dims)
        self.fc_var = nn.Linear(100, latent_dims)
        
        # Distribution
        self.N = torch.distributions.Normal(0, 1)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        
        # Get hidden features
        hidden = self.encoder_hidden(x)
        
        # Get mean and log variance
        mu = self.fc_mu(hidden)
        log_var = self.fc_var(hidden)
        
        # Reparameterization trick
        sigma = torch.exp(0.5 * log_var)
        z = mu + sigma * self.N.sample(mu.shape).to(mu.device)
        
        # KL divergence
        self.kl = (-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())) / x.size(0)
        
        return z

class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dims, 100),
            nn.ReLU(),
            nn.Linear(100, 150),
            nn.ReLU(),
            nn.Linear(150, 75)
        )

    def forward(self, z):
        return self.decoder(z)

class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

def train(autoencoder, train_loader, val_loader, epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder = autoencoder.to(device)
    
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        autoencoder.train()
        train_loss = 0.0
        train_recon_loss = 0.0
        train_kl_loss = 0.0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            x_hat = autoencoder(x)
            
            # Simple reconstruction loss using squared error
            recon_loss = ((x_hat - y) ** 2).sum() / x.size(0)
            kl_loss = autoencoder.encoder.kl
            
            # Total loss
            loss = recon_loss + kl_loss
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_recon_loss += recon_loss.item()
            train_kl_loss += kl_loss.item()

        # Validation phase
        autoencoder.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                x_hat_val = autoencoder(x_val)
                recon_loss_val = ((x_hat_val - y_val) ** 2).sum() / x_val.size(0)
                val_loss += (recon_loss_val + autoencoder.encoder.kl).item()
        
        val_loss /= len(val_loader)
        
        # Print progress
        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"Train Loss: {train_loss/len(train_loader):.6f} "
              f"(Recon: {train_recon_loss/len(train_loader):.6f}, "
              f"KL: {train_kl_loss/len(train_loader):.6f})")
        print(f"Val Loss: {val_loss:.6f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(autoencoder.state_dict(), 'best_vae.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after epoch {epoch+1}")
                break

def evaluate_model(model, test_loader, std_psi, mean_psi):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    predictions = []
    ground_truth = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            
            # Store predictions and ground truth
            predictions.append(pred.cpu().numpy())
            ground_truth.append(y.cpu().numpy())
    
    # Combine batches
    predictions = np.concatenate(predictions, axis=0)
    ground_truth = np.concatenate(ground_truth, axis=0)
    
    # Denormalize
    predictions = predictions * std_psi + mean_psi
    ground_truth = ground_truth * std_psi + mean_psi
    
    # Compute metrics
    mse = np.mean((predictions - ground_truth) ** 2)
    mae = np.mean(np.abs(predictions - ground_truth))
    
    print(f"Test MSE: {mse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    
    return predictions, ground_truth

def plot_predictions(predictions, ground_truth, variable_idx=63):
    time_steps = range(len(ground_truth))
    
    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, ground_truth[:, variable_idx], label='Ground Truth', color='blue')
    plt.plot(time_steps, predictions[:, variable_idx], label='Predictions', 
             color='red', linestyle='dashed', alpha=0.7)
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title(f'One-Step-Ahead Predictions vs Ground Truth for Index {variable_idx}')
    plt.legend()
    plt.grid(True)
    plt.show()

# Main execution
if __name__ == "__main__":
    latent_dims = 75  # Match input dimensions
    vae = VariationalAutoencoder(latent_dims)
    
    # Check if old model exists and remove it to start fresh
    if os.path.exists("best_vae.pth"):
        try:
            # Try to load the model to check dimensions
            state_dict = torch.load("best_vae.pth", weights_only=True)
            # Check if dimensions match
            if state_dict['encoder.fc_mu.weight'].shape[0] != latent_dims:
                print("Found incompatible model checkpoint. Removing old checkpoint to train new model...")
                os.remove("best_vae.pth")
                train(vae, train_loader, val_loader)
            else:
                vae.load_state_dict(state_dict)
                print("Loaded pre-trained model")
        except (RuntimeError, KeyError):
            print("Found incompatible model checkpoint. Removing old checkpoint to train new model...")
            os.remove("best_vae.pth")
            train(vae, train_loader, val_loader)
    else:
        print("Training new model...")
        train(vae, train_loader, val_loader)
    
    # Evaluate and plot results
    predictions, ground_truth = evaluate_model(vae, test_loader, std_psi, mean_psi)
    plot_predictions(predictions, ground_truth)
    
    # Save predictions
    np.savez(
        "vae_predictions.npz",
        predictions=predictions,
        ground_truth=ground_truth,
        mean_psi=mean_psi,
        std_psi=std_psi
    )
    print("Predictions saved to 'vae_predictions.npz'")