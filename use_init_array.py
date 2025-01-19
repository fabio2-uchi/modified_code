from tensorflow.keras.layers import Input, Dense, Lambda, Layer, Convolution1D, MaxPooling1D, Flatten, Reshape, UpSampling1D, Conv1DTranspose, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse
from tensorflow.keras.callbacks import EarlyStopping
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt


# Custom callback for prediction evolution
class PredictionCallback(tf.keras.callbacks.Callback):
    def __init__(self, encoder_model, decoder_model, initial_point, test_time, mean_psi, std_psi, actual_values, zonal_wind_idx=63, verbose=False):
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model
        self.initial_point = initial_point
        self.test_time = test_time
        print(f"Debug: self.test_time = {self.test_time}")
        self.mean_psi = mean_psi
        self.std_psi = std_psi
        self.predictions_history = []
        self.actual_values = actual_values
        self.zonal_wind_idx = zonal_wind_idx

    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch + 1}: Loss = {logs.get('loss')}, Validation Loss = {logs.get('val_loss')}")
        pred_mean = np.zeros((self.test_time, 75, 1))
        initial = self.initial_point.copy()
        
        # Add labeled progress bar
        for k in tqdm(range(self.test_time), desc=f"Epoch {epoch + 1} Prediction Progress"):
            try:
                latent_encoding, _, _ = self.encoder_model.predict_step(initial)
                pred_ens = self.decoder_model.predict_step(latent_encoding)
                pred_step = np.mean(pred_ens, axis=0).reshape(75, 1)
                pred_mean[k, :, :] = pred_step
                pred_denorm = (pred_step.squeeze() * self.std_psi + self.mean_psi).reshape(1, 75, 1)
                initial = (pred_denorm - self.mean_psi.reshape(1, -1, 1)) / self.std_psi.reshape(1, -1, 1)
            except Exception as e:
                print(f"Error during prediction at step {k}: {e}")
                break

        pred_mean = pred_mean.squeeze() * self.std_psi.reshape(1, -1) + self.mean_psi.reshape(1, -1)
        pred_mean = pred_mean.reshape(self.test_time, 75, 1)
        self.predictions_history.append(pred_mean)
        
        np.save(f'predictions_epoch_{epoch}.npy', pred_mean)
        plt.figure(figsize=(15, 10))
        plt.plot(self.actual_values[:, self.zonal_wind_idx], 'b-', label='Actual', linewidth=2)
        plt.plot(pred_mean[:, self.zonal_wind_idx], 'r--', label=f'Predicted', linewidth=2)
        plt.title(f'Predictions vs Actual at Epoch {epoch+1}')
        plt.xlabel('Time Step')
        plt.ylabel('Zonal Wind Speed')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'prediction_epoch_{epoch+1}.png')
        plt.close()

def plot_prediction_evolution(predictions_history, actual_values, zonal_wind_idx=63):
        n_epochs = len(predictions_history)
        plt.figure(figsize=(15, 10))
        plt.plot(actual_values[:, zonal_wind_idx], 'k-', label='Actual', linewidth=2)
        
        for i, pred in enumerate(predictions_history):
            alpha = (i + 1) / n_epochs
            plt.plot(pred[:, zonal_wind_idx], alpha=alpha, 
                    label=f'Epoch {i+1}', linestyle='--')
        
        plt.title(f'Evolution of Predictions at Index {zonal_wind_idx}')
        plt.xlabel('Time Step')
        plt.ylabel('Zonal Wind Speed')
        plt.legend()
        plt.grid(True)
        plt.savefig('prediction_evolution.png')
        plt.show()


# Model setup
latent_dim = 512
cond_dim = 4
model_weights_path = r'C:\Users\Fabio Ventura\Desktop\WORK_STUFF\HM_and_AI_models\VAE_Model\model_weights.weights.h5' 

# Load and preprocess data
F = np.load(r'C:\Users\Fabio Ventura\Desktop\WORK_STUFF\HM_and_AI_models\VAE_Model\x_stoch.npy')
psi = F[3500:, 1, :]

# Normalize data
mean_psi = np.mean(psi, axis=0, keepdims=True)
std_psi = np.std(psi, axis=0, keepdims=True)
psi = (psi - mean_psi) / std_psi

# Data preparation #State 0 is used to train, state 1 is used to make inferences
train_size = 5000
val_size = 1000  # Dedicated validation set size
test_time = 1000  # Test set size
lead = 1

# Define indices for splitting
train_end = train_size
val_start = train_end
val_end = val_start + val_size

# Training data
psi_input_Tr = psi[:train_end, :].reshape(-1, 75, 1)
psi_label_Tr = psi[:train_end, :].reshape(-1, 75, 1)

# Validation data
psi_input_val = psi[val_start:val_end, :].reshape(-1, 75, 1)
psi_label_val = psi[val_start + lead:val_end + lead, :].reshape(-1, 75, 1)

# Test set size (10% of the dataset)
test_size = int(psi.shape[0])

# Define test inputs and labels
psi_test_input = psi[val_end:val_end + test_size, :].reshape(-1, 75, 1)
psi_test_label = psi[val_end + lead:val_end + lead + test_size, :].reshape(-1, 75, 1)

# Define initial point for inference
initial_point = psi[val_end, :].reshape(1, 75, 1)
# Actual values corresponding to the test set for plotting
actual_values = (psi_test_label[:test_time, :, :].squeeze() * std_psi + mean_psi)
print(f"Actual values shape: {actual_values.shape}")

# Print shapes for debugging
print(f"Shape of psi: {psi.shape}")
print(f"Train input shape: {psi_input_Tr.shape}")
print(f"Train label shape: {psi_label_Tr.shape}")
print(f"Validation input shape: {psi_input_val.shape}")
print(f"Validation label shape: {psi_label_val.shape}")
print(f"Test input shape: {psi_test_input.shape}")
print(f"Test label shape: {psi_test_label.shape}")
print(f"Initial point shape: {initial_point.shape}")

#Actually Define the Model
# Encoder
input_data = Input(shape=(75, 1))  # Input shape (time_steps, channels)

x = Convolution1D(64, 3, activation='relu', padding='same')(input_data)
x = MaxPooling1D(2)(x)
x = Convolution1D(64, 3, activation='relu', padding='same')(x)
x = MaxPooling1D(2)(x)
x = Convolution1D(64, 3, activation='relu', padding='same')(x)
x = MaxPooling1D(2)(x)
x = Flatten()(x)  # Suppose this ends up being shape=(None, 10*64) = (None, 640)

# Latent space
distribution_mean = Dense(latent_dim, name='mean')(x)
distribution_logvar = Dense(latent_dim, name='log_variance')(x)

def sample_latent_features(args):
    mean, logvar = args
    epsilon = tf.random.normal(shape=tf.shape(mean))
    return mean + tf.exp(0.5 * logvar) * epsilon

latent_encoding = Lambda(sample_latent_features, name="latent_sampling")(
    [distribution_mean, distribution_logvar]
)

flattened_input_75 = Flatten()(input_data)  # shape = (None, 75)


# 6) Concatenate latent with 75-dim
latent_encoding_conditioned = concatenate(
    [latent_encoding, flattened_input_75], axis=-1
)

# Build encoder model (for clarity/debug)
encoder_model = Model(inputs=input_data,
                      outputs=[latent_encoding, distribution_mean, distribution_logvar],
                      name="Encoder")


# Decoder
decoder_input = Input(shape=(latent_dim + 75,), name="decoder_input")

# 2) Mirror the old shape from the encoder:
#    E.g. if your flattened shape was 10*64, do the same Dense:
encoder = Dense(10 * 64, activation='relu')(decoder_input)  # shape = (None, 640)
encoder = Reshape((10, 64))(encoder)  # (None, 10, 64)

encoder = Conv1DTranspose(64, 3, activation='relu', padding='same')(encoder)
encoder = UpSampling1D(2)(encoder)  # 10 -> 20
encoder = Conv1DTranspose(64, 3, activation='relu', padding='same')(encoder)
encoder = UpSampling1D(2)(encoder)  # 20 -> 40
encoder = Conv1DTranspose(64, 3, activation='relu', padding='same')(encoder)
encoder = UpSampling1D(2)(encoder)  # 40 -> 80

# Crop to match 75
encoder = Lambda(lambda x: x[:, :75, :])(encoder)

# Final output
decoder_output = Conv1DTranspose(1, 3, activation='linear', padding='same')(encoder)

decoder_model = Model(decoder_input, decoder_output, name="Decoder")

# Final output
z, mean, logvar = encoder_model(input_data)
flattened_75 = Flatten()(input_data)
z_conditioned = concatenate([z, flattened_75], axis=-1)  # same logic as above
decoded = decoder_model(z_conditioned)

vae = Model(inputs=input_data, outputs=decoded, name="VAE_with_75_condition")

vae.compile(optimizer="adam", loss="mse")

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
total_loss_layer = TotalLossLayer(beta=0.5)([y_true, decoded, mean, logvar])
vae_with_loss = Model(inputs=[input_data, y_true], outputs=total_loss_layer)
vae_with_loss.compile(optimizer='adam', loss=lambda y_true, y_pred: y_pred, run_eagerly=True)

# Set up callbacks
pred_callback = PredictionCallback(encoder_model, decoder_model, initial_point, test_time, mean_psi, std_psi, actual_values)

# Training and prediction evolution
if os.path.exists(model_weights_path):
    vae_with_loss.load_weights(model_weights_path)
    print(f"Model weights loaded from {model_weights_path}.")
else:
    print(f"No pre-trained weights found. Training model...")
    
    # Train model
    history = vae_with_loss.fit(
    [psi_input_Tr, psi_label_Tr],
    np.zeros((psi_input_Tr.shape[0], 1)),  # Dummy target for custom loss
    validation_data=([psi_input_val, psi_label_val], np.zeros((psi_input_val.shape[0], 1))),
    batch_size=100,
    epochs=10,
    shuffle=True,
    callbacks=[pred_callback, EarlyStopping(monitor='val_loss', patience=5)],
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
    
# Plot prediction evolution
plot_prediction_evolution(pred_callback.predictions_history, actual_values, 63)

# Inference
print("\nStarting inference...")
pred_mean = np.zeros((test_time, 75, 1))
ens = 1
sig_m = 0

# Ensure correct shape of std_psi and mean_psi
std_psi = std_psi.reshape(1, 75)  # (1, 75)
mean_psi = mean_psi.reshape(1, 75)  # (1, 75)

cond_data = initial_point.reshape(1, 75)     # also (1, 75)

for k in tqdm(range(test_time), desc="Inference Progress"):
    latent_encoding, _, _ = encoder_model.predict_step(initial_point)
    noise = np.random.normal(0, sig_m, (ens, latent_dim))
    latent_encoding_ens = np.repeat(latent_encoding, ens, axis=0)  # (ens, 512)
    z_batch = latent_encoding_ens + noise  # shape -> (ens, 512)
    cond_data_single = initial_point.squeeze(axis=-1)  # shape -> (1, 75)
    cond_data_ens = np.repeat(cond_data_single, ens, axis=0)  # shape -> (ens, 75)
    decoder_input = np.concatenate([z_batch, cond_data_ens], axis=-1)  
    pred_ens = decoder_model.predict_step(decoder_input)  
    pred_step = np.mean(pred_ens, axis=0)  # (75, 1) after mean over 'ens'
    pred_mean[k, :, :] = pred_step         # store it
    pred_denorm = (pred_step.squeeze() * std_psi + mean_psi).reshape(1, 75, 1)
    initial_point = (pred_denorm - mean_psi.reshape(1, -1, 1)) / std_psi.reshape(1, -1, 1)

# Denormalize final predictions
pred_mean = pred_mean.squeeze() * std_psi.reshape(1, -1) + mean_psi.reshape(1, -1)
pred_mean = pred_mean.reshape(test_time, 75, 1)

# Slice and denormalize test data
actual_values = psi_test_label[:test_time, :, :].squeeze()  # Shape (500, 75)
actual_values = actual_values * std_psi + mean_psi  # Shape (500, 75)
print(f"Shape of actual_values after denormalization: {actual_values.shape}")

# Calculate Mean Squared Error
from sklearn.metrics import mean_squared_error
pred_flat = pred_mean.reshape(test_time, 75)
actual_flat = actual_values
mse_value = mean_squared_error(actual_flat, pred_flat)
print(f"\nMean Squared Error: {mse_value}")

zonal_wind_idx = 63
# Plot predictions vs actual
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



# Save results
np.savez(r'C:\Users\Fabio Ventura\Desktop\WORK_STUFF\HM_and_AI_models\VAE_Model\predictions.npz',
         predictions=pred_mean, mean_psi=mean_psi, std_psi=std_psi, actual_values=actual_values)
# Plot training history if available
try:
    train_loss = np.load('train_loss.npy')
    val_loss = np.load('val_loss.npy')
    plt.figure(figsize=(10, 6))
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