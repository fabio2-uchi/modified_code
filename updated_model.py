from tensorflow.keras.layers import Input, Dense, Lambda, Layer, Convolution1D, MaxPooling1D, Flatten, Reshape, UpSampling1D, Conv1DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse
from tensorflow.keras.callbacks import EarlyStopping
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt


# Set the latent dimension
latent_dim = 512
model_weights_path = r'C:\Users\Fabio Ventura\Desktop\WORK_STUFF\SHORT-main_V2\model\model_weights.weights.h5'  # File to save/load the model weights

# Load and preprocess data
F = np.load(r'C:\Users\Fabio Ventura\Desktop\WORK_STUFF\SHORT-main_V2\SHORT-main\holton_mass_essentials\Sample_data\150k_Long_run\x_stoch.npy')
psi = F[3500:, 0, :]  # Assuming the relevant data lies in this slice

# Normalize the data
mean_psi = np.mean(psi, axis=0, keepdims=True)
std_psi = np.std(psi, axis=0, keepdims=True)
psi = (psi - mean_psi) / std_psi

# Define lead time
lead = 1

# Prepare input and label data for training/testing
psi_test_input = psi[:-lead, :]
psi_test_label = psi[lead:, :]

# Reshape to match Conv1D input requirements
psi_test_input = psi_test_input.reshape(-1, 75, 1)
psi_test_label = psi_test_label.reshape(-1, 75, 1)

# Define dimensions for training
trainN = 7000  # Number of training samples
psi_input_Tr = psi[-trainN:, :].reshape(-1, 75, 1)  # Training inputs
psi_label_Tr = psi[-trainN:, :].reshape(-1, 75, 1)  # Training labels
print('Train input shape:', psi_input_Tr.shape)
print('Train label shape:', psi_label_Tr.shape)

# Define the VAE model
# Encoder
input_data = Input(shape=(75, 1))  # Input shape (time_steps, channels)
encoder = Convolution1D(64, 3, activation='relu', padding='same')(input_data)
encoder = MaxPooling1D(2)(encoder)  # Downsample: (75 -> 38)
encoder = Convolution1D(64, 3, activation='relu', padding='same')(encoder)
encoder = MaxPooling1D(2)(encoder)  # Downsample: (38 -> 19)
encoder = Convolution1D(64, 3, activation='relu', padding='same')(encoder)
encoder = MaxPooling1D(2)(encoder)  # Downsample: (19 -> 10)
encoder = Flatten()(encoder)

# Latent space
distribution_mean = Dense(latent_dim, name='mean')(encoder)
distribution_variance = Dense(latent_dim, name='log_variance')(encoder)

def sample_latent_features(distribution):
    mean, variance = distribution
    batch_size = tf.shape(variance)[0]
    random = tf.keras.backend.random_normal(shape=(batch_size, tf.shape(variance)[1]))
    return mean + tf.exp(0.5 * variance) * random

latent_encoding = Lambda(sample_latent_features)([distribution_mean, distribution_variance])

# Decoder
decoder_input = Input(shape=(latent_dim,))
decoder = Dense(10 * 64, activation='relu')(decoder_input)  # Match flattened encoder output
decoder = Reshape((10, 64))(decoder)  # Reshape to (time_steps, channels)
decoder = Conv1DTranspose(64, 3, activation='relu', padding='same')(decoder)
decoder = UpSampling1D(2)(decoder)  # Upsample: (10 -> 20)
decoder = Conv1DTranspose(64, 3, activation='relu', padding='same')(decoder)
decoder = UpSampling1D(2)(decoder)  # Upsample: (20 -> 40)
decoder = Conv1DTranspose(64, 3, activation='relu', padding='same')(decoder)
decoder = UpSampling1D(2)(decoder)  # Upsample: (40 -> 80)

# Dynamically crop to match input shape (75, 1)
decoder_output = Lambda(lambda x: x[:, :75, :])(decoder)
decoder_output = Conv1DTranspose(1, 3, activation='linear', padding='same')(decoder_output)  # Final output shape (75, 1)

# Build Models
encoder_model = Model(input_data, [latent_encoding, distribution_mean, distribution_variance], name="Encoder")
decoder_model = Model(decoder_input, decoder_output, name="Decoder")

# Define combined VAE model
latent_encoding, mean, variance = encoder_model(input_data)
decoded = decoder_model(latent_encoding)

# Define custom loss functions
class KLLossLayer(Layer):

    def call(self, inputs):
        mean, variance = inputs
        kl_loss = 1 + variance - tf.square(mean) - tf.exp(variance)
        return -0.5 * tf.reduce_mean(kl_loss)

def get_reconstruction_loss(y_true, y_pred):
    return tf.reduce_mean(mse(y_true, y_pred))

class TotalLossLayer(Layer):

    def __init__(self, beta=1.0, **kwargs):
        super(TotalLossLayer, self).__init__(**kwargs)
        self.beta = beta

    def call(self, inputs):
        y_true, y_pred, mean, variance = inputs
        reconstruction_loss = get_reconstruction_loss(y_true, y_pred)
        kl_loss = KLLossLayer()([mean, variance])
        return reconstruction_loss + self.beta * kl_loss

# Compile the model with TotalLossLayer
y_true = Input(shape=(75, 1))  # Placeholder for labels
total_loss_layer = TotalLossLayer(beta=0.5)([y_true, decoded, mean, variance])
vae_with_loss = Model(inputs=[input_data, y_true], outputs=total_loss_layer)
vae_with_loss.compile(optimizer='adam', loss=lambda y_true, y_pred: y_pred, run_eagerly=True)


if os.path.exists(model_weights_path):
    vae_with_loss.load_weights(model_weights_path)
    print(f"Model weights loaded from {model_weights_path}.")

else:
    print(f"No pre-trained weights found. Training model...")
    history = vae_with_loss.fit(
        [psi_input_Tr, psi_label_Tr],  # Input and labels
        np.zeros((psi_input_Tr.shape[0], 1)),  # Dummy target
        validation_data=([psi_test_input, psi_test_label], np.zeros((psi_test_input.shape[0], 1))),
        batch_size=100,
        epochs=4,
        shuffle=True,
        callbacks=[EarlyStopping(monitor='val_loss', patience=5)],

    )
    vae_with_loss.save_weights(model_weights_path)
    print(f"Model weights saved to {model_weights_path}.")

# Save loss history for plotting
try:
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    np.save('train_loss.npy', train_loss)
    np.save('val_loss.npy', val_loss)
except:
    print("No training history available (weights were loaded).")

# Earlier in the code, ensure test_time is properly set
test_time = min(100, psi_test_label.shape[0])  # Ensure it does not exceed available test samples
print(test_time)
# Prepare input and label data for testing
psi_test_input = psi[:-lead, :]
psi_test_label = psi[lead:, :]

# Reshape to match Conv1D input requirements
psi_test_input = psi_test_input.reshape(-1, 75, 1)
psi_test_label = psi_test_label.reshape(-1, 75, 1)

# Important: Slice the test data to match test_time before inference
psi_test_input = psi_test_input[:test_time]
psi_test_label = psi_test_label[:test_time]


# Initialize prediction array with correct dimensions
pred_mean = np.zeros((test_time, 75, 1))
# Your inference loop remains the same
ens = 100
sig_m = 0.10

# Start with the actual first input
# try to start training where the initial condition starts in the cold state
initial_point = psi_test_input[-100, :, :].reshape(1, 75, 1)

for k in tqdm(range(test_time), desc="Inference Progress"):
    latent_encoding, _, _ = encoder_model.predict(initial_point, verbose=0)
    noise = np.random.normal(0, sig_m, (ens, latent_dim))
    z_batch = latent_encoding + noise
    pred_ens = decoder_model.predict(z_batch, batch_size=ens, verbose=0)
    pred_mean[k, :, :] = np.mean(pred_ens, axis=0).reshape(75, 1)
    initial_point = pred_mean[k, :, :].reshape(1, 75, 1)

# Reshape predictions and actual values for metric calculations
pred_flat = pred_mean.reshape(test_time, 75)  # This should now work correctly
actual_flat = psi_test_label.reshape(test_time, 75)  # This should match pred_flat's shape

# Print shapes for verification
print("Prediction flat shape:", pred_flat.shape)
print("Actual flat shape:", actual_flat.shape)

# Calculate metrics
from sklearn.metrics import mean_squared_error
mse_value = mean_squared_error(actual_flat, pred_flat)
print("Mean Squared Error:", mse_value)

# Save predictions along with metadata
np.savez(r'C:\Users\Fabio Ventura\Desktop\WORK_STUFF\SHORT-main_V2\model\predictions.npz',
         predictions=pred_mean, mean_psi=mean_psi, std_psi=std_psi)

# Plot loss if available
try:
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss vs Epochs')
    plt.savefig('loss_plot.png')
    plt.show()

except:
    print("Loss history not available for plotting.")