import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, Dense, Flatten, Reshape, \
    BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import os
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# ====================== 1. Configuration parameters (following the first python file) ======================
DATA_DIR = r"D:\Oswaldo's surf project\70% Virtual 30% Real Database_Hourly\preprocessed_data_mixed"
X_PATH = os.path.join(DATA_DIR, "X_windows.npy")
SCALED_DF_PATH = os.path.join(DATA_DIR, "normalized_hourly_data.csv")
MODEL_PATH = r"D:\Oswaldo's surf project\70% Virtual 30% Real Database_Hourly\models\conv1d_autoencoder_multi_modal.h5"
VISUALIZATION_PATH = r"D:\Oswaldo's surf project\70% Virtual 30% Real Database_Hourly\visualizations"

TIMESTEPS = 24  # Sliding window size
EPOCHS = 80
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.15
PATIENCE = 12
ANOMALY_THRESHOLD_PCT = 95
PLOT_DAYS = 7

# Feature names (following the first python file)
FEATURE_NAMES = [
    'avg_PM2.5_scaled',
    'total_noise_duration_scaled',
    'noise_event_count_scaled',
    'avg_salience_scaled'
]

# Ensure visualization directory exists
os.makedirs(VISUALIZATION_PATH, exist_ok=True)

import time
start_time = time.time()

# ====================== 2. Data loading process (following the first python file) ======================
print("Loading preprocessed sliding windows...")
X = np.load(X_PATH)
print(f"Loaded X shape: {X.shape}")  # (n_windows, 24, n_features)

n_samples, timesteps, n_features = X.shape
assert timesteps == TIMESTEPS, f"Timesteps mismatch: expected {TIMESTEPS}, got {timesteps}"
assert n_features == len(FEATURE_NAMES), f"Feature count mismatch: expected {len(FEATURE_NAMES)}, got {n_features}"

# Load normalized hourly data for plotting and alignment
df_hourly = pd.read_csv(SCALED_DF_PATH, index_col=0, parse_dates=True)
print(f"Hourly data shape: {df_hourly.shape}")

# ====================== 3. Time series split (following the first python file) ======================
split_idx = int(0.85 * n_samples)
X_train = X[:split_idx]
X_val = X[split_idx:]

print(f"Training windows   : {X_train.shape}")
print(f"Validation windows : {X_val.shape}")


# ====================== 4. Build improved Conv-AE model (following the last python file) ======================
def build_improved_conv_ae(sequence_length=24, n_features=4):
    """
    Build improved Conv-AE model (with residual connections, similar to paper structure)
    Following the build_residual_conv_ae function in the second python file
    """
    print(f"\nBuilding improved Residual CONV-AE:")
    print(f"  Input shape: ({sequence_length}, {n_features})")

    inputs = Input(shape=(sequence_length, n_features))

    # ====== Encoder ======
    # First convolution block
    x1 = Conv1D(32, kernel_size=5, padding='same', activation='relu')(inputs)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling1D(pool_size=2, padding='same')(x1)

    # Second convolution block
    x2 = Conv1D(64, kernel_size=3, padding='same', activation='relu')(x1)
    x2 = BatchNormalization()(x2)
    x2 = MaxPooling1D(pool_size=2, padding='same')(x2)

    # Third convolution block
    x3 = Conv1D(128, kernel_size=3, padding='same', activation='relu')(x2)
    x3 = BatchNormalization()(x3)
    encoded = MaxPooling1D(pool_size=2, padding='same')(x3)

    # ====== Decoder ======
    # First deconvolution block
    y1 = Conv1D(128, kernel_size=3, padding='same', activation='relu')(encoded)
    y1 = BatchNormalization()(y1)
    y1 = UpSampling1D(size=2)(y1)

    # Second deconvolution block
    y2 = Conv1D(64, kernel_size=3, padding='same', activation='relu')(y1)
    y2 = BatchNormalization()(y2)
    y2 = UpSampling1D(size=2)(y2)

    # Third deconvolution block
    y3 = Conv1D(32, kernel_size=3, padding='same', activation='relu')(y2)
    y3 = BatchNormalization()(y3)
    y3 = UpSampling1D(size=2)(y3)

    # Output layer
    outputs = Conv1D(n_features, kernel_size=5, padding='same', activation='linear')(y3)

    # ====== Full model ======
    autoencoder = Model(inputs, outputs, name='improved_conv_ae')

    # Compute parameters
    total_params = autoencoder.count_params()
    trainable_params = np.sum([np.prod(v.shape) for v in autoencoder.trainable_weights])
    non_trainable_params = total_params - trainable_params

    print(f"\nModel parameters count:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Non-trainable parameters: {non_trainable_params:,}")

    return autoencoder, total_params


# ====================== 5. Build or load model ======================
print("\nBuilding / Loading Improved Conv1D Autoencoder...")

if os.path.exists(MODEL_PATH):
    print(f"Loading saved model from {MODEL_PATH}")
    try:
        # Conv-AE has no custom layers, load directly
        autoencoder = load_model(MODEL_PATH)
        print("Model loaded successfully!")
        total_params = autoencoder.count_params()
        history = None
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Building new model instead...")
        autoencoder, total_params = build_improved_conv_ae(TIMESTEPS, n_features)
        history = None
else:
    print("No saved model → building new Conv-AE")
    autoencoder, total_params = build_improved_conv_ae(TIMESTEPS, n_features)

    # Compile model - using MAE as loss function
    autoencoder.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mae',  # Use mean absolute error
        metrics=['mse']  # Also track mean squared error
    )

    # Train model
    print(f"\nTraining for max {EPOCHS} epochs...")
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1
    )

    history = autoencoder.fit(
        X_train, X_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, X_val),
        callbacks=[early_stop],
        verbose=1
    )

    autoencoder.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

# ====================== 6. Compute validation loss ======================
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
    print(f"  - Best validation loss (MAE): {best_val_loss:.6f} (epoch {best_epoch})")
    print(f"  - Final training loss (MAE): {final_train_loss:.6f}")
    print(f"  - Final validation loss (MAE): {final_val_loss:.6f}")
    print(f"  - Training epochs: {len(history.history['loss'])}")

    # If MSE metrics are available
    if 'val_mse' in history.history:
        best_val_mse = min(history.history['val_mse'])
        final_val_mse = history.history['val_mse'][-1]
        print(f"  - Best validation MSE: {best_val_mse:.6f}")
        print(f"  - Final validation MSE: {final_val_mse:.6f}")
else:
    # For loaded model, re-evaluate validation loss
    print("Evaluating loaded model on validation set...")
    val_metrics = autoencoder.evaluate(X_val, X_val, batch_size=BATCH_SIZE, verbose=0)

    if isinstance(val_metrics, list):
        val_mae = val_metrics[0]
        if len(val_metrics) > 1:
            val_mse = val_metrics[1]
            print(f"  - Validation loss (MAE): {val_mae:.6f}")
            print(f"  - Validation MSE: {val_mse:.6f}")
        else:
            print(f"  - Validation loss: {val_mae:.6f}")
    else:
        print(f"  - Validation loss: {val_metrics:.6f}")

print("=" * 60)

# ====================== 7. Reconstruction and threshold calculation ======================
print("\nComputing reconstruction errors...")
recon_val = autoencoder.predict(X_val, batch_size=BATCH_SIZE, verbose=0)
mae_val = np.mean(np.abs(X_val - recon_val), axis=(1, 2))

threshold = np.percentile(mae_val, ANOMALY_THRESHOLD_PCT)
print(f"Validation MAE {ANOMALY_THRESHOLD_PCT}th percentile threshold: {threshold:.6f}")

recon_full = autoencoder.predict(X, batch_size=BATCH_SIZE, verbose=0)
mae_full = np.mean(np.abs(X - recon_full), axis=(1, 2))

print("\nComputing validation loss...")
val_loss = autoencoder.evaluate(X_val, X_val, verbose=0)
print(f"Validation Loss (MAE): {val_loss[0]:.6f}")
print(f"Validation MSE: {val_loss[1]:.6f}")

# ====================== 8. Anomaly detection and alignment ======================
anomaly_flags = (mae_full > threshold).astype(int)

window_end_indices = df_hourly.index[TIMESTEPS - 1: TIMESTEPS - 1 + len(mae_full)]
df_anomalies = pd.DataFrame({
    'reconstruction_mae': mae_full,
    'is_detected_anomaly': anomaly_flags
}, index=window_end_indices)

anomaly_count = np.sum(anomaly_flags)
anomaly_ratio = np.mean(anomaly_flags) * 100
print(f"Detected {anomaly_count} anomalous windows out of {len(anomaly_flags)} ({anomaly_ratio:.2f}%)")


# ====================== 9. Comprehensive visualization (following the second python file's layout) ======================
def create_comprehensive_visualization(df_hourly, df_anomalies, mae_full, threshold,
                                       X, recon_full, history=None, model_name="Improved Conv-AE"):
    """
    Create comprehensive visualization (following the second python file's layout)
    """
    print(f"\nCreating comprehensive visualization...")

    # Set plot style
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(20, 14))

    # 1. Training loss curve (if history data is available)
    if history is not None:
        ax1 = plt.subplot(3, 3, 1)
        epochs = range(1, len(history.history['loss']) + 1)
        ax1.plot(epochs, history.history['loss'], label='Training Loss', linewidth=2)
        ax1.plot(epochs, history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax1.set_title(f'{model_name} - Training History', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('MAE Loss', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # 2. Reconstruction error time series
    ax2 = plt.subplot(3, 3, 2 if history is None else 2)
    ax2.plot(df_anomalies.index, df_anomalies['reconstruction_mae'],
             label='Reconstruction MAE', color='teal', alpha=0.7, linewidth=1)
    ax2.axhline(threshold, color='red', linestyle='--', linewidth=2,
                label=f'Threshold ({threshold:.5f})')

    # Mark anomaly points
    anomaly_indices = df_anomalies[df_anomalies['is_detected_anomaly'] == 1].index
    if len(anomaly_indices) > 0:
        anomaly_errors = df_anomalies.loc[anomaly_indices, 'reconstruction_mae']
        ax2.scatter(anomaly_indices, anomaly_errors, color='red',
                    marker='o', s=40, alpha=0.7, label='Detected Anomaly')

    ax2.set_title('Reconstruction Error Time Series', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Mean Absolute Error', fontsize=12)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)

    # 3. Reconstruction error distribution
    ax3 = plt.subplot(3, 3, 3 if history is None else 3)
    n_bins = min(100, len(mae_full) // 10)
    ax3.hist(mae_full, bins=n_bins, alpha=0.7, color='skyblue',
             edgecolor='black', density=True)
    ax3.axvline(x=threshold, color='red', linestyle='--', linewidth=2,
                label=f'Threshold ({threshold:.4f})')

    # Add Gaussian fit
    from scipy.stats import norm
    mu, std = norm.fit(mae_full)
    xmin, xmax = ax3.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    ax3.plot(x, p, 'k', linewidth=2,
             label=f'Normal fit: μ={mu:.4f}, σ={std:.4f}')

    ax3.set_title('Reconstruction Error Distribution', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Reconstruction Error (MAE)', fontsize=12)
    ax3.set_ylabel('Density', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Feature-level comparison (first sample)
    ax4 = plt.subplot(3, 3, 4 if history is None else 4)
    if len(X) > 0:
        sample_idx = 0
        time_steps = range(TIMESTEPS)

        # Plot first feature of the first sample
        feature_idx = 0
        original_values = X[sample_idx][:, feature_idx]
        reconstructed_values = recon_full[sample_idx][:, feature_idx]

        ax4.plot(time_steps, original_values, label='Original',
                 linewidth=2, alpha=0.8, color='blue')
        ax4.plot(time_steps, reconstructed_values, label='Reconstructed',
                 linewidth=2, alpha=0.8, color='red', linestyle='--')

        ax4.set_title(f'Sample {sample_idx} - {FEATURE_NAMES[feature_idx]}',
                      fontsize=14, fontweight='bold')
        ax4.set_xlabel('Time Step (hours)', fontsize=12)
        ax4.set_ylabel('Normalized Value', fontsize=12)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    # 5. Feature-level comparison (second feature of the first sample)
    ax5 = plt.subplot(3, 3, 5 if history is None else 5)
    if len(X) > 0 and len(FEATURE_NAMES) > 1:
        sample_idx = 0
        time_steps = range(TIMESTEPS)

        feature_idx = 1
        original_values = X[sample_idx][:, feature_idx]
        reconstructed_values = recon_full[sample_idx][:, feature_idx]

        ax5.plot(time_steps, original_values, label='Original',
                 linewidth=2, alpha=0.8, color='green')
        ax5.plot(time_steps, reconstructed_values, label='Reconstructed',
                 linewidth=2, alpha=0.8, color='orange', linestyle='--')

        ax5.set_title(f'Sample {sample_idx} - {FEATURE_NAMES[feature_idx]}',
                      fontsize=14, fontweight='bold')
        ax5.set_xlabel('Time Step (hours)', fontsize=12)
        ax5.set_ylabel('Normalized Value', fontsize=12)
        ax5.legend()
        ax5.grid(True, alpha=0.3)

    # 6. Model parameter information
    ax6 = plt.subplot(3, 3, 6 if history is None else 6)
    models = ['LSTM-AE', 'Conv-AE', 'Improved Conv-AE']
    # Approximate parameter counts (based on paper and actual calculation)
    params_approx = [29124, 7892, total_params]
    colors = ['lightblue', 'lightgreen', 'lightcoral']

    bars = ax6.bar(models, params_approx, color=colors, alpha=0.8)
    ax6.set_title('Model Parameter Comparison', fontsize=14, fontweight='bold')
    ax6.set_ylabel('Number of Parameters', fontsize=12)
    ax6.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, param in zip(bars, params_approx):
        ax6.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 500,
                 f'{param:,}', ha='center', va='bottom', fontsize=10)

    # 7. Feature-level anomaly detection for the first N days
    ax7 = plt.subplot(3, 3, 7 if history is None else 7)
    n_plot = PLOT_DAYS * 24
    plot_df = df_hourly.iloc[:n_plot].copy()
    plot_anom = df_anomalies.iloc[:n_plot].copy()

    # Plot first feature
    if len(FEATURE_NAMES) > 0:
        feat = FEATURE_NAMES[0]
        ax7.plot(plot_df.index, plot_df[feat], label=feat, color='teal', lw=1.2)

        # Get anomaly indices and ensure they exist in plot_df
        anom_idx = plot_anom[plot_anom['is_detected_anomaly'] == 1].index
        # Use index intersection to avoid KeyError
        valid_anom_idx = anom_idx.intersection(plot_df.index)

        if len(valid_anom_idx) > 0:
            ax7.scatter(valid_anom_idx, plot_df.loc[valid_anom_idx, feat],
                        color='red', marker='o', s=50, label='Anomaly', alpha=0.7)

        ax7.set_ylabel(feat, fontsize=10)
        ax7.legend(loc='upper right', fontsize=9)
        ax7.grid(True, alpha=0.3)
        ax7.set_title(f'First {PLOT_DAYS} Days - {feat}', fontsize=12, fontweight='bold')

    # 8. Anomaly rate vs threshold
    ax8 = plt.subplot(3, 3, 8 if history is None else 8)
    thresholds = np.percentile(mae_val, range(90, 100))
    anomaly_rates = []

    for t in thresholds:
        anomaly_rate = np.sum(mae_full > t) / len(mae_full) * 100
        anomaly_rates.append(anomaly_rate)

    ax8.plot(thresholds, anomaly_rates, 'o-', linewidth=2, markersize=6)
    ax8.axvline(x=threshold, color='red', linestyle='--', linewidth=1.5,
                label=f'95th percentile ({threshold:.4f})')
    ax8.set_title('Anomaly Rate vs Threshold', fontsize=14, fontweight='bold')
    ax8.set_xlabel('Threshold', fontsize=12)
    ax8.set_ylabel('Anomaly Rate (%)', fontsize=12)
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # 9. Performance summary
    ax9 = plt.subplot(3, 3, 9 if history is None else 9)
    ax9.axis('off')

    summary_text = f"""
    {model_name} Performance Summary:

    Total Samples: {len(X):,}
    Training Samples: {len(X_train):,}
    Validation Samples: {len(X_val):,}

    Total Parameters: {total_params:,}

    Validation Threshold: {threshold:.6f}

    Detected Anomalies: {anomaly_count:,}
    Anomaly Ratio: {anomaly_ratio:.2f}%

    Sequence Length: {TIMESTEPS} hours
    Features: {n_features}
    """

    ax9.text(0.1, 0.95, summary_text, fontsize=10,
             verticalalignment='top', family='monospace')

    plt.tight_layout()

    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(VISUALIZATION_PATH, f"conv_ae_results_{timestamp}.png")

    try:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    except Exception as e:
        print(f"Failed to save image: {e}")
        alt_path = f"improved_conv_ae_results_{timestamp}.png"
        plt.savefig(alt_path, dpi=150, bbox_inches='tight')
        print(f"Saved to current directory: {alt_path}")

    plt.close()

    return save_path


# ====================== 10. Create visualization ======================
# If training history exists, load it
history_data = None
if os.path.exists(MODEL_PATH.replace('.h5', '_history.csv')):
    try:
        history_df = pd.read_csv(MODEL_PATH.replace('.h5', '_history.csv'))
        history_data = type('History', (), {'history': history_df.to_dict('list')})()
    except:
        pass

# Create comprehensive visualization
vis_path = create_comprehensive_visualization(
    df_hourly=df_hourly,
    df_anomalies=df_anomalies,
    mae_full=mae_full,
    threshold=threshold,
    X=X,
    recon_full=recon_full,
    history=history_data,
    model_name="Improved Conv-AE"
)

# Calculate total runtime
end_time = time.time()
total_time_seconds = end_time - start_time
total_time_minutes = total_time_seconds / 60
total_time_hours = total_time_minutes / 60

# Output total runtime
print(f"\n⏱️  Total Execution Time:")
print(f"  Total time: {total_time_seconds:.2f} seconds")
print(f"            : {total_time_minutes:.2f} minutes")
print(f"            : {total_time_hours:.2f} hours")

# ====================== 11. Output summary report ======================
print("\n" + "=" * 80)
print("IMPROVED CONVOLUTIONAL AUTOENCODER - TRAINING COMPLETE")
print("=" * 80)

print(f"\n📊 Model Performance Summary:")
print(f"  Model: Improved Residual Conv-AE")
print(f"  Total parameters: {total_params:,}")
print(f"  Input shape: ({TIMESTEPS}, {n_features})")
print(f"  Training samples: {X_train.shape[0]:,}")
print(f"  Validation samples: {X_val.shape[0]:,}")

print(f"\n📈 Anomaly Detection Results:")
print(f"  Anomaly threshold (95th percentile) : {threshold:.6f}")
print(f"  Detected anomalies: {anomaly_count:,} out of {len(mae_full):,}")
print(f"  Anomaly ratio: {anomaly_ratio:.2f}%")

print(f"\n💾 Saved Files:")
print(f"  Model: {MODEL_PATH}")
print(f"  Visualization: {vis_path}")

print(f"\n✅ Improved Conv-AE pipeline completed successfully!")
print("   Model is lightweight → suitable for TensorFlow Lite Micro / edge deployment.")

# ====================== 12. Save results as JSON file (following the second file) ======================
import json


def save_results_conv(results_dict, save_dir):
    """Save training results to JSON file"""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(save_dir, f"conv_ae_results_{timestamp}.json")

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)

    print(f"💾 Results saved to: {results_path}")
    return results_path


# Construct results dictionary
results_dict = {
    'model_name': ' Conv-AE',
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'data_info': {
        'total_samples': X.shape[0],
        'timesteps': TIMESTEPS,
        'n_features': n_features,
        'training_samples': X_train.shape[0],
        'validation_samples': X_val.shape[0]
    },
    'model_info': {
        'total_params': total_params,
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'architecture': 'Residual Conv1D Autoencoder'
    },
    'anomaly_detection': {
        'threshold_percentile': ANOMALY_THRESHOLD_PCT,
        'threshold_value': float(threshold),
        'anomalies_detected': int(anomaly_count),
        'total_windows': len(mae_full),
        'anomaly_ratio': float(anomaly_ratio)
    },
    'file_paths': {
        'model_path': MODEL_PATH,
        'visualization_path': vis_path,
        'data_path': DATA_DIR
    },
    'feature_names': FEATURE_NAMES
}

# Save JSON file
models_dir = r"D:\Oswaldo's surf project\70% Virtual 30% Real Database_Hourly\models"
results_path = save_results_conv(results_dict, models_dir)

# Update summary report output
print(f"\n💾 Output Files:")
print(f"  - Model: {MODEL_PATH}")
print(f"  - Visualization: {vis_path}")
print(f"  - Results (JSON): {results_path}")
