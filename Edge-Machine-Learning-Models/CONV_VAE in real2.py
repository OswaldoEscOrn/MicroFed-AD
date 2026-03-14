import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, UpSampling1D,
                                     Dense, Flatten, Reshape, BatchNormalization,
                                     Dropout, Lambda, Layer)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import os
import warnings
import json

warnings.filterwarnings('ignore')

# ====================== 1. Configuration Parameters ======================
DATA_DIR = r"D:\Oswaldo's surf project\DR O's database\preprocessed_data"
X_PATH = os.path.join(DATA_DIR, "X_windows_100k.npy")
SCALED_DF_PATH = os.path.join(DATA_DIR, "normalized_hourly_data.csv")
MODEL_PATH = r"D:\Oswaldo's surf project\DR O's database\models\conv_vae_multi_modal100k_data.h5"
VISUALIZATION_PATH = r"D:\Oswaldo's surf project\DR O's database\visualizations"

TIMESTEPS = 24
FILTERS = [32, 16]
KERNEL_SIZE = 3
POOL_SIZE = 2
LATENT_DIM = 16
EPOCHS = 80
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.15
PATIENCE = 12
KL_WEIGHT = 0.001
ANOMALY_THRESHOLD_PCT = 95
PLOT_DAYS = 7

FEATURE_NAMES = [
    'avg_PM2.5_normalized_scaled',
    'total_noise_duration_scaled',
    'noise_event_count_scaled',
    'avg_salience_scaled'
]

os.makedirs(VISUALIZATION_PATH, exist_ok=True)


# ====================== 2. Custom Layers ======================
class Sampling(Layer):
    """Reparameterization trick layer"""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class VAELossLayer(Layer):
    """VAE loss calculation layer"""

    def __init__(self, kl_weight=0.001, **kwargs):
        super(VAELossLayer, self).__init__(**kwargs)
        self.kl_weight = kl_weight
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.recon_loss_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    def call(self, inputs):
        x_true, x_pred, z_mean, z_log_var = inputs

        # Calculate reconstruction loss - fix: remove unnecessary scaling
        reconstruction_loss = K.mean(K.abs(x_true - x_pred))  # directly take mean

        # Calculate KL divergence
        kl_loss = -0.5 * K.sum(
            1 + z_log_var - K.square(z_mean) - K.exp(z_log_var),
            axis=1
        )
        kl_loss = K.mean(kl_loss) * self.kl_weight

        total_loss = reconstruction_loss + kl_loss

        self.add_loss(total_loss)
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return x_pred

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.recon_loss_tracker, self.kl_loss_tracker]

    def compute_output_shape(self, input_shape):
        return input_shape[1]


# ====================== 3. Data Loading ======================

import time  # Ensure time module is imported
# Record start time
start_time = time.time()

print("Loading preprocessed sliding windows...")
X = np.load(X_PATH)
print(f"Loaded X shape: {X.shape}")

n_samples, timesteps, n_features = X.shape
assert timesteps == TIMESTEPS, f"Timesteps mismatch: expected {TIMESTEPS}, got {timesteps}"
assert n_features == len(FEATURE_NAMES), f"Feature count mismatch: expected {len(FEATURE_NAMES)}, got {n_features}"

df_hourly = pd.read_csv(SCALED_DF_PATH, index_col=0, parse_dates=True)
print(f"Hourly data shape: {df_hourly.shape}")

# Check data range
print(f"Data statistics:")
print(f"  Min: {X.min():.4f}")
print(f"  Max: {X.max():.4f}")
print(f"  Mean: {X.mean():.4f}")
print(f"  Std: {X.std():.4f}")

# Temporal split
split_idx = int(0.85 * n_samples)
X_train = X[:split_idx]
X_val = X[split_idx:]

print(f"Training windows   : {X_train.shape}")
print(f"Validation windows : {X_val.shape}")


# ====================== 4. Build Conv-VAE model ======================
def build_conv_vae():
    """Build a 1D convolutional variational autoencoder"""
    print("\nBuilding Conv-VAE model...")

    encoder_inputs = Input(shape=(TIMESTEPS, n_features), name='encoder_input')

    # Encoder
    x = Conv1D(32, kernel_size=5, activation='relu', padding='same')(encoder_inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2, padding='same')(x)
    x = Dropout(0.2)(x)

    x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2, padding='same')(x)
    x = Dropout(0.2)(x)

    x = Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2, padding='same')(x)

    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)

    z_mean = Dense(LATENT_DIM, name='z_mean')(x)
    z_log_var = Dense(LATENT_DIM, name='z_log_var')(x)
    z = Sampling()([z_mean, z_log_var])

    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')

    # Decoder
    latent_inputs = Input(shape=(LATENT_DIM,), name='z_sampling')
    conv_shape = (TIMESTEPS // 8, 128)

    x = Dense(int(np.prod(conv_shape)), activation='relu')(latent_inputs)
    x = Reshape(conv_shape)(x)

    x = Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling1D(size=2)(x)

    x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling1D(size=2)(x)

    x = Conv1D(32, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling1D(size=2)(x)

    # Use sigmoid activation, assuming data is in [0,1] range
    decoder_outputs = Conv1D(n_features, kernel_size=5, activation='sigmoid', padding='same')(x)

    decoder = Model(latent_inputs, decoder_outputs, name='decoder')

    # Complete VAE model
    z_mean, z_log_var, z = encoder(encoder_inputs)
    outputs = decoder(z)
    final_outputs = VAELossLayer(kl_weight=KL_WEIGHT)([encoder_inputs, outputs, z_mean, z_log_var])

    vae = Model(encoder_inputs, final_outputs, name='conv_vae')
    vae.compile(optimizer=Adam(learning_rate=0.001))

    total_params = vae.count_params()
    trainable_params = np.sum([np.prod(v.shape) for v in vae.trainable_weights])
    non_trainable_params = total_params - trainable_params

    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Non-trainable params: {non_trainable_params:,}")

    return vae, encoder, decoder, total_params


# ====================== 5. Train model ======================
print("\nBuilding / Loading Conv-VAE...")
history = None

if os.path.exists(MODEL_PATH):
    print(f"Loading saved model from {MODEL_PATH}")
    try:
        vae = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={'Sampling': Sampling, 'VAELossLayer': VAELossLayer},
            compile=False  # Important: load without compiling
        )
        print("Model loaded successfully!")

        # Recompile model
        vae.compile(optimizer=Adam(learning_rate=0.001))
        print("Model recompiled successfully!")

        total_params = vae.count_params()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Building new model instead...")
        vae, encoder, decoder, total_params = build_conv_vae()
else:
    print("No saved model → building new Conv-VAE")
    vae, encoder, decoder, total_params = build_conv_vae()

    print(f"\nTraining for max {EPOCHS} epochs...")
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1
    )

    history = vae.fit(
        X_train, X_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, X_val),
        callbacks=[early_stop],
        verbose=1
    )

    vae.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

# ====================== 6. Calculate Validation Loss ======================
print("\n" + "=" * 60)
print("VALIDATION LOSS INFORMATION")
print("=" * 60)

if history is not None:
    # Get validation loss from training history
    best_val_loss = min(history.history['val_loss'])
    best_epoch = np.argmin(history.history['val_loss']) + 1
    final_val_loss = history.history['val_loss'][-1]
    final_train_loss = history.history['loss'][-1]

    print(f"Training completed:")
    print(f"  - Best validation loss: {best_val_loss:.6f} (epoch {best_epoch})")
    print(f"  - Final training loss: {final_train_loss:.6f}")
    print(f"  - Final validation loss: {final_val_loss:.6f}")
    print(f"  - Training epochs: {len(history.history['loss'])}")
else:
    # For loaded model, re-evaluate validation loss
    print("Evaluating loaded model on validation set...")
    # Note: VAE model's evaluate may return multiple values (due to multiple metrics)
    eval_results = vae.evaluate(X_val, X_val, batch_size=BATCH_SIZE, verbose=0)

    # Determine return type
    if isinstance(eval_results, list):
        # If multiple metrics, the first is total loss
        val_loss = eval_results[0]
        print(f"  - Validation loss: {val_loss:.6f}")
        if len(eval_results) > 1:
            print(f"  - Reconstruction loss: {eval_results[1]:.6f}")
            if len(eval_results) > 2:
                print(f"  - KL loss: {eval_results[2]:.6f}")
    else:
        # If only one value
        val_loss = eval_results
        print(f"  - Validation loss: {val_loss:.6f}")

print("=" * 60)

# ====================== 7. Reconstruction and Anomaly Detection ======================
print("\nComputing reconstruction errors...")
recon_val = vae.predict(X_val, batch_size=BATCH_SIZE, verbose=0)
mae_val = np.mean(np.abs(X_val - recon_val), axis=(1, 2))

threshold = np.percentile(mae_val, ANOMALY_THRESHOLD_PCT)
print(f"Validation MAE {ANOMALY_THRESHOLD_PCT}th percentile threshold: {threshold:.6f}")

recon_full = vae.predict(X, batch_size=BATCH_SIZE, verbose=0)
mae_full = np.mean(np.abs(X - recon_full), axis=(1, 2))

anomaly_flags = (mae_full > threshold).astype(int)
anomaly_count = np.sum(anomaly_flags)
anomaly_ratio = np.mean(anomaly_flags) * 100

# Diagnose data length mismatch
print(f"\nDiagnosing data length mismatch:")
print(f"  - mae_full length: {len(mae_full)}")
print(f"  - X shape: {X.shape}")
print(f"  - df_hourly length: {len(df_hourly)}")
print(f"  - Expected windows from hourly data: {len(df_hourly) - TIMESTEPS + 1}")

# Take the minimum length to ensure matching
n_windows = min(len(mae_full), len(df_hourly) - TIMESTEPS + 1)
print(f"  - Using {n_windows} windows for alignment")

# Truncate data to match
mae_full = mae_full[:n_windows]
anomaly_flags = anomaly_flags[:n_windows]

window_end_indices = df_hourly.index[TIMESTEPS - 1: TIMESTEPS - 1 + n_windows]
df_anomalies = pd.DataFrame({
    'reconstruction_mae': mae_full,
    'is_detected_anomaly': anomaly_flags
}, index=window_end_indices)

print(f"Detected {anomaly_count} anomalous windows out of {len(anomaly_flags)} ({anomaly_ratio:.2f}%)")

# ====================== 8. Create Visualizations ======================
print("\nGenerating visualizations...")

# 1. Training history plot (if available)
if history is not None:
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)

    best_epoch = np.argmin(history.history['val_loss'])
    best_val_loss = history.history['val_loss'][best_epoch]
    plt.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.5)
    plt.scatter(best_epoch, best_val_loss, color='red', s=100, zorder=5)

    plt.title('Conv-VAE Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_PATH, 'conv_vae_training_history100k_data.png'), dpi=150, bbox_inches='tight')
    plt.close()

# 2. Reconstruction error plot
plt.figure(figsize=(14, 6))
plt.plot(df_anomalies.index, df_anomalies['reconstruction_mae'],
         label='Reconstruction MAE', color='purple', alpha=0.7)
plt.axhline(threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.5f})')
plt.scatter(df_anomalies[df_anomalies['is_detected_anomaly'] == 1].index,
            df_anomalies[df_anomalies['is_detected_anomaly'] == 1]['reconstruction_mae'],
            color='red', marker='o', s=60, label='Detected Anomaly')
plt.title('Conv-VAE Reconstruction Error & Detected Anomalies')
plt.xlabel('Time')
plt.ylabel('Mean Absolute Error (per window)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(VISUALIZATION_PATH, 'conv_vae_reconstruction_errors100k_data.png'), dpi=150, bbox_inches='tight')
plt.close()


# ====================== 8.1 Comprehensive Visualization (optional, similar to CONV_AE) ======================
def create_comprehensive_visualization():
    """Create comprehensive visualization (similar to CONV_AE)"""
    print("\nCreating comprehensive visualization...")

    plt.figure(figsize=(20, 12))

    # 1. Training history (if available)
    if history is not None:
        ax1 = plt.subplot(2, 3, 1)
        epochs = range(1, len(history.history['loss']) + 1)
        ax1.plot(epochs, history.history['loss'], label='Training Loss', linewidth=2)
        ax1.plot(epochs, history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax1.set_title('Conv-VAE Training History', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # 2. Reconstruction error distribution
    ax2 = plt.subplot(2, 3, 2 if history is None else 2)
    ax2.hist(mae_full, bins=50, alpha=0.7, color='skyblue', edgecolor='black', density=True)
    ax2.axvline(x=threshold, color='red', linestyle='--', linewidth=2,
                label=f'Threshold ({threshold:.4f})')
    ax2.set_title('Reconstruction Error Distribution')
    ax2.set_xlabel('Reconstruction Error (MAE)')
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Reconstruction error time series
    ax3 = plt.subplot(2, 3, 3 if history is None else 3)
    ax3.plot(df_anomalies.index, df_anomalies['reconstruction_mae'],
             label='Reconstruction MAE', color='purple', alpha=0.7, linewidth=1)
    ax3.axhline(threshold, color='red', linestyle='--', linewidth=2,
                label=f'Threshold ({threshold:.5f})')

    anomaly_indices = df_anomalies[df_anomalies['is_detected_anomaly'] == 1].index
    if len(anomaly_indices) > 0:
        ax3.scatter(anomaly_indices,
                    df_anomalies.loc[anomaly_indices, 'reconstruction_mae'],
                    color='red', marker='o', s=30, alpha=0.7, label='Detected Anomaly')

    ax3.set_title('Reconstruction Error Time Series')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Mean Absolute Error')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)

    # 4. Model parameter comparison
    ax4 = plt.subplot(2, 3, 4 if history is None else 4)
    models = ['LSTM-AE', 'Conv-AE', 'Conv-VAE']
    params = [29124, 7892, total_params]
    colors = ['lightgreen', 'lightcoral', 'gold']

    bars = ax4.bar(models, params, color=colors, alpha=0.8)
    ax4.set_title('Model Parameter Comparison')
    ax4.set_ylabel('Number of Parameters')
    ax4.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, param in zip(bars, params):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2, height + max(params) * 0.05,
                 f'{param:,}', ha='center', va='bottom', fontsize=9)

    # 5. Feature-level plot (first 7 days) - fix index issue
    ax5 = plt.subplot(2, 3, 5 if history is None else 5)
    n_plot = PLOT_DAYS * 24

    if len(df_hourly) >= n_plot:
        plot_df = df_hourly.iloc[:n_plot].copy()

        # Get time range of plot_df
        plot_start = plot_df.index[0]
        plot_end = plot_df.index[-1]

        # Filter anomalies within time range
        time_mask = (df_anomalies.index >= plot_start) & (df_anomalies.index <= plot_end)
        plot_anom = df_anomalies[time_mask].copy()

        if len(FEATURE_NAMES) > 0:
            feat = FEATURE_NAMES[0]
            ax5.plot(plot_df.index, plot_df[feat], label=feat, color='teal', lw=1.2)

            # Ensure anomaly timestamps exist in plot_df
            if len(plot_anom) > 0:
                valid_anom = plot_anom[plot_anom['is_detected_anomaly'] == 1]
                valid_anom_idx = valid_anom.index[valid_anom.index.isin(plot_df.index)]

                if len(valid_anom_idx) > 0:
                    ax5.scatter(valid_anom_idx, plot_df.loc[valid_anom_idx, feat],
                                color='red', marker='o', s=40, label='Anomaly')

            ax5.set_ylabel(feat)
            ax5.legend(loc='upper right')
            ax5.grid(True, alpha=0.3)

    # 6. Performance summary text
    ax6 = plt.subplot(2, 3, 6 if history is None else 6)
    ax6.axis('off')

    if history is not None:
        best_val_loss = min(history.history['val_loss'])
        final_val_loss = history.history['val_loss'][-1]
        val_loss_text = f"Best Val Loss: {best_val_loss:.4f}\nFinal Val Loss: {final_val_loss:.4f}"
    else:
        val_loss = vae.evaluate(X_val, X_val, batch_size=BATCH_SIZE, verbose=0)
        val_loss_text = f"Validation Loss: {val_loss:.4f}"

    summary_text = f"""Conv-VAE Performance Summary

Total Parameters: {total_params:,}
Training Samples: {len(X_train):,}
Validation Samples: {len(X_val):,}

Anomaly Threshold: {threshold:.4f}
Detected Anomalies: {anomaly_count:,}
Anomaly Ratio: {anomaly_ratio:.2f}%

{val_loss_text}

Latent Dimension: {LATENT_DIM}
KL Weight: {KL_WEIGHT}
"""

    ax6.text(0.1, 0.5, summary_text, fontsize=11,
             verticalalignment='center', family='monospace')

    plt.tight_layout()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(VISUALIZATION_PATH, f"conv_vae_comprehensive_100k_data{timestamp}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Comprehensive visualization saved to: {save_path}")
    return save_path


# Call the comprehensive visualization function
comprehensive_path = create_comprehensive_visualization()

# ====================== 9. Output Summary Report ======================
print("\n" + "=" * 80)
print("CONVOLUTIONAL VARIATIONAL AUTOENCODER - PERFORMANCE SUMMARY")
print("=" * 80)

# Get validation loss information
if history is not None:
    best_val_loss = min(history.history['val_loss'])
    final_val_loss = history.history['val_loss'][-1]
    val_loss_info = f"""
Training History:
  - Best validation loss: {best_val_loss:.6f}
  - Final validation loss: {final_val_loss:.6f}"""
else:
    val_loss = vae.evaluate(X_val, X_val, batch_size=BATCH_SIZE, verbose=0)
    val_loss_info = f"""
Validation Loss:
  - Validation loss: {val_loss:.6f}"""

summary_text = f"""
Model Name: Conv-VAE
Performance Summary:

Total Samples: {len(X):,}
Training Samples: {len(X_train):,}
Validation Samples: {len(X_val):,}

Total Parameters: {total_params:,}

Anomaly threshold (95th percentile): {threshold:.6f}
Anomalies detected: {anomaly_count:,} ({anomaly_ratio:.2f}%)
{val_loss_info}

Sequence Length: {TIMESTEPS} hours
Features: {n_features}

Model saved to: {MODEL_PATH}
Visualizations saved to: {VISUALIZATION_PATH}
Timestamp: {datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}
"""
print(summary_text)
print("=" * 80)

# Calculate total execution time
end_time = time.time()
total_time_seconds = end_time - start_time
total_time_minutes = total_time_seconds / 60
total_time_hours = total_time_minutes / 60

# Output total execution time
print(f"\n⏱️  Total Execution Time:")
print(f"  Total time: {total_time_seconds:.2f} seconds")
print(f"            : {total_time_minutes:.2f} minutes")
print(f"            : {total_time_hours:.2f} hours")


# ====================== 10. Save Results as JSON File ======================
def save_results_vae(results_dict, save_dir):
    """Save VAE training results to JSON file"""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(save_dir, f"conv_vae_results100k_data{timestamp}.json")

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)

    print(f"💾 Results saved to: {results_path}")
    return results_path


# Build results dictionary
results_dict = {
    'model_name': 'Conv-VAE',
    'timestamp': datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
    'data_info': {
        'total_samples': X.shape[0],
        'timesteps': TIMESTEPS,
        'n_features': n_features,
        'training_samples': X_train.shape[0],
        'validation_samples': X_val.shape[0]
    },
    'model_info': {
        'total_params': int(total_params),
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'latent_dim': LATENT_DIM,
        'kl_weight': KL_WEIGHT,
        'architecture': 'Convolutional Variational Autoencoder (VAE)'
    },
    'anomaly_detection': {
        'threshold_percentile': ANOMALY_THRESHOLD_PCT,
        'threshold_value': float(threshold),
        'anomalies_detected': int(anomaly_count),
        'total_windows': len(mae_full),
        'anomaly_ratio': float(anomaly_ratio)
    },
    'training_info': {
        'model_path': MODEL_PATH,
        'visualization_dir': VISUALIZATION_PATH,
        'data_path': DATA_DIR
    },
    'feature_names': FEATURE_NAMES,
    # Add total execution time to results
    'total_execution_time_seconds': float(total_time_seconds),
    'total_execution_time_minutes': float(total_time_minutes),
    'total_execution_time_hours': float(total_time_hours)
}

# Add validation loss information
if history is not None:
    results_dict['validation_results'] = {
        'best_val_loss': float(min(history.history['val_loss'])),
        'final_val_loss': float(history.history['val_loss'][-1]),
        'final_train_loss': float(history.history['loss'][-1])
}
else:
    val_loss = vae.evaluate(X_val, X_val, batch_size=BATCH_SIZE, verbose=0)
    results_dict['validation_results'] = {
        'validation_loss': float(val_loss)
    }

# Save JSON file
models_dir = r"D:\Oswaldo's surf project\DR O's database\models"
results_path = save_results_vae(results_dict, models_dir)

# ====================== 11. Final Output ======================
print(f"\n📊 Output Files:")
print(f"  - Model: {MODEL_PATH}")
print(f"  - Visualizations directory: {VISUALIZATION_PATH}")
print(f"  - Results (JSON): {results_path}")

print(f"\n✅ Conv-VAE pipeline completed successfully!")
print(f"   - Model combines probabilistic latent space with convolutional efficiency")
print(f"   - Validation loss properly calculated and reported")
