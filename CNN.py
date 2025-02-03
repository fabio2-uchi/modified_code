import torch.optim.adadelta
from tqdm import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt
import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn as F
import torch.utils
import torch.distributions
import torchvision
from torch.utils.data import TensorDataset, DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
torch.cuda.empty_cache()

# Model setup
model_weights_path = r'/home/constantino-daniel-boscu/Documents/research/AI-RES/modified-code-main3/100ep-jan10th/model_weights.weights.h5' 

# Load and preprocess data
F = np.load(r'/home/constantino-daniel-boscu/Documents/research/AI-RES/modified-code-main3/x_stoch.npy' )
psi = F[3500:, 0, :]

# Normalize data
mean_psi = np.mean(psi, axis=0, keepdims=True)
std_psi = np.std(psi, axis=0, keepdims=True)
psi = (psi - mean_psi) / std_psi

# Data preparation
train_size = 100000
val_size = 20000  # Dedicated validation set size
test_time = 1500

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
test_size = int(0.5 * psi.shape[0])
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

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

### Defining the CNN model

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.linear1 = nn.Linear(75, 100)
        self.linear2 = nn.Linear(100, 150)
        self.linear3 = nn.Linear(150, 75)

    def forward(self, z):
        z = self.linear1(z).relu()
        z = self.linear2(z).relu()
        z = self.linear3(z)
        return z

model = CNN()
loss_fn = nn.MSELoss()

def train(cnn, train_loader, val_loader, epochs=5):

    opt = torch.optim.Adam(cnn.parameters(), lr = 0.001)

    for e in range(epochs):

        cnn.train()
        train_loss = 0.0
        min_valid_loss = np.inf

        for x, y in train_loader:
            y = y.to(device)
            x = x.to(device) # GPU
            opt.zero_grad()
            
            # Get reconstruction, mu, and log_var from the autoencoder.
            x_hat = cnn(x)

            loss = ((x_hat - y) ** 2).sum() / x.size(0)

            loss.backward()
            opt.step()

            train_loss += loss.item()

        print("Training", train_loss)
    
        cnn.eval()
        valid_loss = 0.0
        
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            x_hat = cnn(x)
            valid_loss = ((x_hat - y) ** 2).sum() / x.size(0)
            
        print("Valid", valid_loss)

        print(f'Epoch {e+1} \t\t Training Loss: {\
        train_loss / len(train_loader)} \t\t Validation Loss: {\
        valid_loss / len(val_loader)}')
        
        if min_valid_loss > valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss}--->{valid_loss}) \t Saving The Model')
            min_valid_loss = valid_loss
            
            # Saving State Dict
            torch.save(cnn.state_dict(), 'saved_model.pth')
    return cnn
    
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


# Running everything 

cnn_with_loss = CNN().to(device) # GPU

if not os.path.exists(model_weights_path):
    cnn_with_loss.load_state_dict(torch.load(model_weights_path, weights_only=True))
    print(f"Model weights loaded from {model_weights_path}.")
else:
    print(f"No pre-trained weights found. Training model...")
    
    # Train model
    cnn_with_loss = train(cnn_with_loss, train_loader, val_loader)

    torch.save(cnn_with_loss.state_dict(), 'model_weights.pth')
    print(f"Model weights saved to {model_weights_path}.")

    # Evaluate and plot results
    predictions, ground_truth = evaluate_model(cnn_with_loss, test_loader, std_psi, mean_psi)
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