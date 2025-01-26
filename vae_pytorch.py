import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data Loading and Preprocessing
F = np.load(r'C:\Users\danie\OneDrive\Desktop\Holton Mass Model\Stochastic-VAE-for-Digital-Twins\x_stoch.npy')
psi = F[3500:, 0, :]

# Normalize data
mean_psi = np.mean(psi, axis=0, keepdims=True)
std_psi = np.std(psi, axis=0, keepdims=True)
psi = (psi - mean_psi) / std_psi

# Data preparation
train_size = 10000
val_size = 5000
lead = 1

# Splitting indices
train_end = train_size
val_start = train_end
val_end = val_start + val_size

def reshape_data(data):
    return data.reshape(-1, 75)

# Training, validation, and test sets
psi_input_Tr = reshape_data(psi[:train_end, :])
psi_label_Tr = reshape_data(psi[:train_end, :])
psi_input_val = reshape_data(psi[val_start:val_end, :])
psi_label_val = reshape_data(psi[val_start + lead:val_end + lead, :])

# Test set
test_size = int(0.1 * psi.shape[0])
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
    
#data loaders
train_dataset = AtmosphereDataset(psi_input_Tr, psi_label_Tr)
val_dataset = AtmosphereDataset(psi_input_val, psi_label_val)
test_dataset = AtmosphereDataset(psi_test_input, psi_test_label)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=False)

class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(75, 150)
        print(f"linear 1 {self.linear1}")
        self.linear2 = nn.Linear(150, latent_dims+75)
        print(f"linear 2 {self.linear2}")
        self.linear3 = nn.Linear(150, latent_dims+75)
        print(f"linear 3 {self.linear3}")

        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cuda()
        self.N.loc = self.N.loc # hack to get sampling on the GPU
        self.N.scale = self.N.scale
        self.kl = 0

    def forward(self, x):
        print(f"valeu of x passed in forward encoder: {x}")
        print(x.shape)
        x = torch.flatten(x, start_dim=1)
        print(x.shape)
        x = self.linear1(x).relu()
        mu =  self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z
    
class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims+75, 100)
        self.linear2 = nn.Linear(100, 75)

    def forward(self, z):
        z = self.linear1(z).relu()
        z = torch.relu(self.linear2(z))
        return z.reshape((-1, 75))
    
class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        print(f"latent_dims in ecncoder: {latent_dims}")
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    
def train(autoencoder, data, epochs=20):
    opt = torch.optim.Adam(autoencoder.parameters())
    train_loss = 0.0
    for epoch in range(epochs):
        for x, y in data:
            # x = x.to(device) # GPU
            opt.zero_grad()
            x_hat = autoencoder(x)
            loss = ((x - x_hat)**2).sum() + autoencoder.encoder.kl
            loss.backward()
            opt.step()
            train_loss += loss.item()
    return autoencoder

latent_dims = 75

vae = VariationalAutoencoder(latent_dims) # GPU
vae = train(vae, train_loader)


# Switch the model to evaluation mode
vae.eval()

# Get test data
test_inputs = torch.tensor(psi_test_input, dtype=torch.float32)

# Forward pass through the VAE
with torch.no_grad():
    latent_representations = vae.encoder(test_inputs)  # Encode test inputs
    predictions = vae.decoder(latent_representations)  # Decode latent representations

# Convert predictions to NumPy
predictions = predictions.numpy()

# Denormalize predictions
denormalized_predictions = (predictions * std_psi + mean_psi)

# Ground truth labels
ground_truth = (psi_test_label * std_psi + mean_psi)

# Plot comparison for a specific variable (e.g., variable at level 25)
time_steps = range(len(ground_truth))
variable_idx = 63  # Change this to select a different variable

plt.figure(figsize=(10, 6))
plt.plot(time_steps, ground_truth[:, variable_idx], label="Ground Truth", color="blue")
plt.plot(time_steps, denormalized_predictions[:, variable_idx], label="Predictions", color="red", linestyle="dashed")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.title(f"Predictions vs Real for Index {variable_idx}")
plt.legend()
plt.grid(True)
plt.show()

np.savez(
    "vae_predictions.npz",
    predictions=denormalized_predictions,
    ground_truth=ground_truth,
    mean_psi=mean_psi,
    std_psi=std_psi,
)
print("Predictions saved to 'vae_predictions.npz'.")




# def plot_latent(autoencoder, data, num_batches=100):
#     for i, (x, y) in enumerate(data):
#         # z = autoencoder.encoder(x.to(device))
#         z = autoencoder.encoder(x)
#         # z = z.to('cpu').detach().numpy()
#         z = z.detach().numpy()
#         plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
#         if i > num_batches:
#             plt.colorbar()
#             break
# plot_latent(vae, val_loader)

# def plot_reconstructed(autoencoder, r0=(-5, 10), r1=(-10, 5), n=12):
#     w = 28
#     img = np.zeros((n*w, n*w))
#     for i, y in enumerate(np.linspace(*r1, n)):
#         for j, x in enumerate(np.linspace(*r0, n)):
#             # z = torch.Tensor([[x, y]]).to(device)
#             z = torch.Tensor([[x, y]])
#             x_hat = autoencoder.decoder(z)
#             # x_hat = x_hat.reshape(28, 28).to('cpu').detach().numpy()
#             x_hat = x_hat.reshape(28, 28).detach().numpy()
#             img[(n-1-i)*w:(n-1-i+1)*w, j*w:(j+1)*w] = x_hat
#     plt.imshow(img, extent=[*r0, *r1])
# plot_reconstructed(vae)