"""
model.py — Conditional VAE model implementation

ResNet-inspired Conditional Variational Autoencoder for emulating
the stochastic Holton-Mass model of stratospheric variability.

Architecture:
  - Encoder: 6-layer residual MLP (75 → latent_dim mean + logvar)
  - Decoder: 6-layer residual MLP (latent_dim + condition_dim → 75)
"""

import torch
import torch.nn as nn


NUM_NEURONS = 1024


class Encoder(nn.Module):
    """Maps the full system state X(t) to latent mean and log-variance."""

    def __init__(self, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(75, NUM_NEURONS)
        self.fc2 = nn.Linear(NUM_NEURONS, NUM_NEURONS)
        self.fc3 = nn.Linear(NUM_NEURONS, NUM_NEURONS)
        self.fc4 = nn.Linear(NUM_NEURONS, NUM_NEURONS)
        self.fc5 = nn.Linear(NUM_NEURONS, NUM_NEURONS)
        self.fc6 = nn.Linear(NUM_NEURONS, NUM_NEURONS)
        self.fc_mu = nn.Linear(NUM_NEURONS, latent_dim)
        self.fc_logvar = nn.Linear(NUM_NEURONS, latent_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x)) + x
        x = torch.relu(self.fc3(x)) + x
        x = torch.relu(self.fc4(x)) + x
        x = torch.relu(self.fc5(x)) + x
        x = torch.relu(self.fc6(x)) + x
        return self.fc_mu(x), self.fc_logvar(x)


class Decoder(nn.Module):
    """Maps latent z + condition Psi(t) to predicted next state X(t+dt)."""

    def __init__(self, latent_dim, output_dim, condition_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim + condition_dim, NUM_NEURONS)
        self.fc2 = nn.Linear(NUM_NEURONS, NUM_NEURONS)
        self.fc3 = nn.Linear(NUM_NEURONS, NUM_NEURONS)
        self.fc4 = nn.Linear(NUM_NEURONS, NUM_NEURONS)
        self.fc5 = nn.Linear(NUM_NEURONS, NUM_NEURONS)
        self.fc6 = nn.Linear(NUM_NEURONS, NUM_NEURONS)
        self.fc_output = nn.Linear(NUM_NEURONS, output_dim)

    def forward(self, z, condition):
        x = torch.cat((z, condition), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x)) + x
        x = torch.relu(self.fc3(x)) + x
        x = torch.relu(self.fc4(x)) + x
        x = torch.relu(self.fc5(x)) + x
        x = torch.relu(self.fc6(x)) + x
        return self.fc_output(x)


class ConditionalVAE(nn.Module):
    """
    Conditional Variational Autoencoder.

    Encoder:  q_phi(z | X(t))
    Decoder:  p_theta(X(t+dt) | Psi(t), z)
    """

    def __init__(self, latent_dim, output_dim, condition_dim):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim, output_dim, condition_dim)

    def encode(self, x):
        return self.encoder(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, condition):
        return self.decoder(z, condition)

    def forward(self, x, condition):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        output = self.decode(z, condition)
        return output, mu, logvar
