import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import warnings

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ====================== Configuration ======================
# Data paths - following the data loading procedure from the first Python file
DATA_DIR = r"D:\Oswaldo's surf project\DR O's database\preprocessed_data"
X_PATH = os.path.join(DATA_DIR, "X_windows.npy")
SCALED_DF_PATH = os.path.join(DATA_DIR, "normalized_hourly_data.csv")

# Model save path
MODEL_SAVE_DIR = r"D:\Oswaldo's surf project\DR O's database\models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "lstm_autoencoder_hybrid.h5")

# Visualization save path
VISUALIZATION_SAVE_DIR = r"D:\Oswaldo's surf project\DR O's database\visualizations"
os.makedirs(VISUALIZATION_SAVE_DIR, exist_ok=True)

# Model parameters - following the model structure from the second Python file
TIMESTEPS = 24
LSTM_UNITS = 128
ENCODING_DIM = 32
EPOCHS = 80
BATCH_SIZE = 64  # optimization from the second Python file
VALIDATION_SPLIT = 0.15
PATIENCE = 12

# Anomaly detection parameters
ANOMALY_THRESHOLD_PCT = 95
PLOT_DAYS = 7

# Feature names - as described in the PDF
FEATURE_NAMES = [
    'avg_PM2.5_normalized_scaled',
    'total_noise_duration_scaled',
    'noise_event_count_scaled',
    'avg_salience_scaled'
]


# ====================== 1. Data Loading ======================
def load_data():
    """Following the data loading procedure from the first Python file"""
    print("=" * 80)
    print("LSTM-AE HYBRID IMPLEMENTATION")
    print("=" * 80)

    print("\n📂 Loading preprocessed sliding windows...")

    # Load window data
    if not os.path.exists(X_PATH):
        raise FileNotFoundError(f"X_windows.npy not found at {X_PATH}")

    X = np.load(X_PATH)
    print(f"Loaded X shape: {X.shape}")  # Expected: (n_windows, 24, n_features)

    n_samples, timesteps, n_features = X.shape
    assert timesteps == TIMESTEPS, f"Timesteps mismatch: expected {TIMESTEPS}, got {timesteps}"
    assert n_features == len(FEATURE_NAMES), f"Feature count mismatch: expected {len(FEATURE_NAMES)}, got {n_features}"

    # Load hourly data for subsequent analysis
    print("\n📂 Loading normalized hourly data...")
    if os.path.exists(SCALED_DF_PATH):
        df_hourly = pd.read_csv(SCALED_DF_PATH, index_col=0, parse_dates=True)
        print(f"Hourly data shape: {df_hourly.shape}")
    else:
        print(f"Warning: {SCALED_DF_PATH} not found, creating placeholder hourly data")
        # Create placeholder data
        n_hours = n_samples + TIMESTEPS - 1
        date_range = pd.date_range(start='2013-03-01', periods=n_hours, freq='H')
        df_hourly = pd.DataFrame(
            np.random.randn(n_hours, len(FEATURE_NAMES)),
            index=date_range,
            columns=FEATURE_NAMES
        )

    return X, df_hourly


# ====================== 2. Data Splitting ======================
def split_data(X, split_ratio=0.85):
    """Following the data splitting method from the first Python file"""
    n_samples = X.shape[0]
    split_idx = int(split_ratio * n_samples)

    X_train = X[:split_idx]
    X_val = X[split_idx:]

    print(f"\n📊 Dataset splitting:")
    print(f"  Training windows   : {X_train.shape} ({split_ratio * 100:.1f}%)")
    print(f"  Validation windows : {X_val.shape} ({100 - split_ratio * 100:.1f}%)")
    print(f"  Total windows      : {n_samples}")

    return X_train, X_val


# ====================== 3. Model Building ======================
def build_lstm_autoencoder(sequence_length, n_features):
    """Building LSTM Autoencoder following the second Python file's model structure"""
    print(f"\n🔧 Building LSTM-AutoEncoder model:")
    print(f"  Input shape: ({sequence_length}, {n_features})")
    print(f"  LSTM units: {LSTM_UNITS}")
    print(f"  Encoding dimension: {ENCODING_DIM}")

    # ====== Encoder ======
    inputs = Input(shape=(sequence_length, n_features), name='lstm_input')

    # First LSTM layer
    encoded = LSTM(LSTM_UNITS, activation='tanh', return_sequences=True,
                   name='encoder_lstm1')(inputs)
    encoded = BatchNormalization(name='encoder_bn1')(encoded)
    encoded = Dropout(0.2, name='encoder_dropout1')(encoded)

    # Second LSTM layer
    encoded = LSTM(LSTM_UNITS // 2, activation='tanh', return_sequences=True,
                   name='encoder_lstm2')(encoded)
    encoded = BatchNormalization(name='encoder_bn2')(encoded)
    encoded = Dropout(0.2, name='encoder_dropout2')(encoded)

    # Third LSTM layer (does not return sequences)
    encoded = LSTM(LSTM_UNITS // 4, activation='tanh', return_sequences=False,
                   name='encoder_lstm3')(encoded)

    # Bottleneck layer
    encoded = Dense(ENCODING_DIM, activation='tanh', name='bottleneck')(encoded)

    # ====== Decoder ======
    # Repeat vector
    decoded = RepeatVector(sequence_length, name='repeat_vector')(encoded)

    # First LSTM layer
    decoded = LSTM(LSTM_UNITS // 4, activation='tanh', return_sequences=True,
                   name='decoder_lstm1')(decoded)
    decoded = BatchNormalization(name='decoder_bn1')(decoded)
    decoded = Dropout(0.2, name='decoder_dropout1')(decoded)

    # Second LSTM layer
    decoded = LSTM(LSTM_UNITS // 2, activation='tanh', return_sequences=True,
                   name='decoder_lstm2')(decoded)
    decoded = BatchNormalization(name='decoder_bn2')(decoded)
    decoded = Dropout(0.2, name='decoder_dropout2')(decoded)

    # Third LSTM layer
    decoded = LSTM(LSTM_UNITS, activation='tanh', return_sequences=True,
                   name='decoder_lstm3')(decoded)
    decoded = BatchNormalization(name='decoder_bn3')(decoded)

    # Output layer
    outputs = TimeDistributed(Dense(n_features, activation='linear'),
                              name='output')(decoded)

    # ====== Full model ======
    autoencoder = Model(inputs, outputs, name='lstm_autoencoder')

    # Encoder model (for feature extraction)
    encoder = Model(inputs, encoded, name='lstm_encoder')

    return autoencoder, encoder


# ====================== 4. Model Training ======================
def train_model(autoencoder, X_train, X_val):
    """Train the model, combining advantages from both Python files"""
    print(f"\n🚀 Starting training...")

    # Show model summary
    autoencoder.summary()

    # Calculate parameter count
    total_params = autoencoder.count_params()
    print(f"\n📊 Model parameters count:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Parameters size: {total_params * 4 / 1024:.2f} KB (float32)")

    # Compile the model - using MAE loss
    autoencoder.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mae',
        metrics=['mse']
    )

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

    print(f"\n📈 Training configuration:")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Max Epochs: {EPOCHS}")
    print(f"  Validation split: {VALIDATION_SPLIT}")

    # Train the model
    history = autoencoder.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    return history, total_params


# ====================== 5. Anomaly Detection ======================
def detect_anomalies(autoencoder, X, X_val, threshold_percentile=95):
    """Following the anomaly detection procedure from the first Python file"""
    print("\n🔍 Computing reconstruction errors...")

    # Validation set reconstruction error
    reconstructions_val = autoencoder.predict(X_val, batch_size=BATCH_SIZE, verbose=0)
    mae_val = np.mean(np.abs(X_val - reconstructions_val), axis=(1, 2))

    # Calculate threshold
    threshold = np.percentile(mae_val, threshold_percentile)
    print(f"Validation (assumed normal) MAE {threshold_percentile}th percentile threshold: {threshold:.6f}")

    # Full dataset reconstruction error
    reconstructions_full = autoencoder.predict(X, batch_size=BATCH_SIZE, verbose=0)
    mae_full = np.mean(np.abs(X - reconstructions_full), axis=(1, 2))

    # Detect anomalies
    anomaly_flags = (mae_full > threshold).astype(int)
    anomaly_count = np.sum(anomaly_flags)
    anomaly_ratio = anomaly_count / len(mae_full) * 100

    print(f"\nDetected {anomaly_count} anomalous windows out of {len(mae_full)} ({anomaly_ratio:.2f}%)")

    return mae_full, anomaly_flags, threshold, reconstructions_full


# ====================== 6. Visualization ======================
def visualize_results(df_hourly, mae_full, anomaly_flags, threshold,
                      X_original=None, X_reconstructed=None, history=None):
    """Following the visualization style from the second Python file, but only saving, not displaying"""
    print("\n📊 Generating visualizations...")

    # Set plotting style
    plt.style.use('seaborn-v0_8-darkgrid')

    # Create main figure - following the layout of the second Python file
    fig = plt.figure(figsize=(20, 15))

    # 1. Training history (if history provided)
    if history is not None:
        ax1 = plt.subplot(3, 3, 1)
        epochs = range(1, len(history.history['loss']) + 1)
        ax1.plot(epochs, history.history['loss'], label='Training Loss', linewidth=2)
        ax1.plot(epochs, history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax1.set_title('Training History', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('MAE Loss', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # 2. Reconstruction error time series (style from the first Python file)
    ax2 = plt.subplot(3, 3, 2)

    # Create time indices
    if len(df_hourly) >= len(mae_full):
        time_indices = df_hourly.index[TIMESTEPS - 1: TIMESTEPS - 1 + len(mae_full)]
    else:
        time_indices = range(len(mae_full))

    ax2.plot(time_indices, mae_full, label='Reconstruction MAE',
             color='steelblue', alpha=0.7, linewidth=1)
    ax2.axhline(threshold, color='red', linestyle='--',
                label=f'Threshold ({threshold:.5f})', linewidth=2)

    # Mark anomaly points
    anomaly_indices = np.where(anomaly_flags == 1)[0]
    if len(time_indices) == len(mae_full):
        ax2.scatter(time_indices[anomaly_indices], mae_full[anomaly_indices],
                    color='red', marker='o', s=30, label='Detected Anomaly', alpha=0.6)

    ax2.set_title('LSTM-Autoencoder Reconstruction Error & Detected Anomalies',
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Mean Absolute Error (per window)', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # 3. Reconstruction error distribution
    ax3 = plt.subplot(3, 3, 3)
    n_bins = min(50, len(mae_full) // 20)
    ax3.hist(mae_full, bins=n_bins, alpha=0.7, color='skyblue',
             edgecolor='black', density=True)
    ax3.axvline(x=threshold, color='red', linestyle='--', linewidth=2,
                label=f'Threshold ({threshold:.4f})')
    ax3.set_title('Reconstruction Error Distribution', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Reconstruction MAE', fontsize=12)
    ax3.set_ylabel('Density', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Original feature visualization (first 7 days)
    ax4 = plt.subplot(3, 3, 4)
    n_plot_hours = PLOT_DAYS * 24
    if len(df_hourly) >= n_plot_hours:
        plot_data = df_hourly.iloc[:n_plot_hours]
        ax4.plot(plot_data.index, plot_data[FEATURE_NAMES[0]],
                 label=FEATURE_NAMES[0], color='teal', linewidth=1.5)
        ax4.set_title(f'{FEATURE_NAMES[0]} (First {PLOT_DAYS} Days)',
                      fontsize=14, fontweight='bold')
        ax4.set_xlabel('Time', fontsize=12)
        ax4.set_ylabel('Normalized Value', fontsize=12)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    # 5. Multi-feature comparison plot
    ax5 = plt.subplot(3, 3, 5)
    if len(df_hourly) >= n_plot_hours:
        for i, feat in enumerate(FEATURE_NAMES):
            ax5.plot(plot_data.index, plot_data[feat], label=feat,
                     linewidth=1, alpha=0.8)
        ax5.set_title(f'All Features (First {PLOT_DAYS} Days)',
                      fontsize=14, fontweight='bold')
        ax5.set_xlabel('Time', fontsize=12)
        ax5.set_ylabel('Normalized Value', fontsize=12)
        ax5.legend(loc='upper right', fontsize=9)
        ax5.grid(True, alpha=0.3)

    # 6. Anomaly detection performance analysis
    ax6 = plt.subplot(3, 3, 6)
    thresholds = np.percentile(mae_full, range(90, 100))
    anomaly_rates = []

    for t in thresholds:
        anomaly_rate = np.sum(mae_full > t) / len(mae_full) * 100
        anomaly_rates.append(anomaly_rate)

    ax6.plot(thresholds, anomaly_rates, 'o-', linewidth=2, markersize=6)
    ax6.axvline(x=threshold, color='red', linestyle='--', linewidth=1.5,
                label=f'{ANOMALY_THRESHOLD_PCT}th percentile ({threshold:.4f})')
    ax6.set_title('Anomaly Rate vs Threshold', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Threshold', fontsize=12)
    ax6.set_ylabel('Anomaly Rate (%)', fontsize=12)
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # 7. Original vs Reconstruction comparison (first sample)
    ax7 = plt.subplot(3, 3, 7)
    if X_original is not None and X_reconstructed is not None and len(X_original) > 0:
        sample_idx = 0
        time_steps = range(TIMESTEPS)

        # Plot first feature
        ax7.plot(time_steps, X_original[sample_idx, :, 0],
                 label='Original', linewidth=2, alpha=0.8)
        ax7.plot(time_steps, X_reconstructed[sample_idx, :, 0],
                 label='Reconstructed', linewidth=2, alpha=0.8,
                 color='red', linestyle='--')

        ax7.set_title(f'Sample {sample_idx} - {FEATURE_NAMES[0]}',
                      fontsize=14, fontweight='bold')
        ax7.set_xlabel('Time Step (hours)', fontsize=12)
        ax7.set_ylabel('Feature Value', fontsize=12)
        ax7.legend()
        ax7.grid(True, alpha=0.3)

    # 8. Error boxplot
    ax8 = plt.subplot(3, 3, 8)
    if len(anomaly_indices) > 0:
        normal_errors = mae_full[anomaly_flags == 0]
        anomaly_errors = mae_full[anomaly_flags == 1]

        box_data = [normal_errors, anomaly_errors]
        box_labels = ['Normal', 'Anomaly']

        bp = ax8.boxplot(box_data, labels=box_labels, patch_artist=True)

        colors = ['lightblue', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax8.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5)
        ax8.set_title('Error Distribution by Category', fontsize=14, fontweight='bold')
        ax8.set_ylabel('Reconstruction MAE', fontsize=12)
        ax8.grid(True, alpha=0.3, axis='y')

    # 9. Feature heatmap (reconstruction error)
    ax9 = plt.subplot(3, 3, 9)
    if X_original is not None and X_reconstructed is not None and len(X_original) > 0:
        # Calculate feature-level error for the first sample
        sample_idx = 0
        feature_errors = np.abs(X_original[sample_idx] - X_reconstructed[sample_idx])

        im = ax9.imshow(feature_errors.T, aspect='auto', cmap='YlOrRd')
        ax9.set_xlabel('Time Step', fontsize=12)
        ax9.set_ylabel('Feature Index', fontsize=12)
        ax9.set_title(f'Sample {sample_idx} Feature-wise MAE',
                      fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax9, label='MAE')

        # Set feature labels
        ax9.set_yticks(range(len(FEATURE_NAMES)))
        ax9.set_yticklabels([f'F{i + 1}' for i in range(len(FEATURE_NAMES))])

    # Adjust layout
    plt.tight_layout()

    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(VISUALIZATION_SAVE_DIR, f"lstm_hybrid_visualization_{timestamp}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📸 Visualization saved to: {save_path}")

    plt.close(fig)

    return save_path


# ====================== 7. Save Results ======================
def save_results(results_dict, save_dir):
    """Save training results to a JSON file"""
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(save_dir, f"lstm_hybrid_results_{timestamp}.json")

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)

    print(f"💾 Results saved to: {results_path}")
    return results_path


# ====================== 8. Main Workflow ======================
def main():
    """Main function"""
    # Set TensorFlow optimizations
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    print("=" * 80)
    print("LSTM-AE HYBRID IMPLEMENTATION")
    print("Combining data loading from PDF/LSTM_AE.py with model from LSTM_realize_original.py")
    print("=" * 80)

    try:
        # 1. Load data (following the first Python file)
        X, df_hourly = load_data()

        # 2. Split data (following the first Python file)
        X_train, X_val = split_data(X, split_ratio=0.85)

        # 3. Build model (following the second Python file)
        print("\n" + "=" * 80)
        print("MODEL CONSTRUCTION")
        print("=" * 80)

        autoencoder, encoder = build_lstm_autoencoder(TIMESTEPS, len(FEATURE_NAMES))

        # 4. Train model (combining advantages of both files)
        print("\n" + "=" * 80)
        print("MODEL TRAINING")
        print("=" * 80)
        history, total_params = train_model(autoencoder, X_train, X_val)

        # Calculate best validation loss
        if history is not None and 'val_loss' in history.history:
            best_val_loss = min(history.history['val_loss'])
            best_epoch = np.argmin(history.history['val_loss']) + 1
            print(f"\n✨ Best Validation Loss: {best_val_loss:.6f} (epoch {best_epoch})")

            # If MSE metric exists
            if 'val_mse' in history.history:
                best_val_mse = min(history.history['val_mse'])
                print(f"✨ Best Validation MSE: {best_val_mse:.6f}")
        else:
            best_val_loss = history.history['val_loss'][-1] if history else None
            best_epoch = None
            print(f"\n⚠️ Warning: Could not compute best validation loss")

        # 5. Save model
        autoencoder.save(MODEL_PATH)
        print(f"\n💾 Model saved to: {MODEL_PATH}")

        # 6. Anomaly detection (following the first Python file)
        print("\n" + "=" * 80)
        print("ANOMALY DETECTION")
        print("=" * 80)
        mae_full, anomaly_flags, threshold, reconstructions_full = detect_anomalies(
            autoencoder, X, X_val, ANOMALY_THRESHOLD_PCT
        )

        # 7. Visualization (following the second Python file's layout, only saving)
        print("\n" + "=" * 80)
        print("VISUALIZATION")
        print("=" * 80)
        viz_path = visualize_results(
            df_hourly, mae_full, anomaly_flags, threshold,
            X_original=X, X_reconstructed=reconstructions_full, history=history
        )

        # 8. Prepare results dictionary
        results_dict = {
            'model_name': 'LSTM-AE_Hybrid',
            'data_loading_method': 'PDF/LSTM_AE.py',
            'model_architecture': 'LSTM_realize_original.py',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_info': {
                'total_samples': X.shape[0],
                'timesteps': TIMESTEPS,
                'n_features': len(FEATURE_NAMES),
                'training_samples': X_train.shape[0],
                'validation_samples': X_val.shape[0]
            },
            'model_info': {
                'total_params': total_params,
                'lstm_units': LSTM_UNITS,
                'encoding_dim': ENCODING_DIM,
                'batch_size': BATCH_SIZE,
                'epochs_trained': len(history.history['loss']),
                'final_train_loss': float(history.history['loss'][-1]),
                'final_val_loss': float(history.history['val_loss'][-1])
            },
            'anomaly_detection': {
                'threshold_percentile': ANOMALY_THRESHOLD_PCT,
                'threshold_value': float(threshold),
                'anomalies_detected': int(np.sum(anomaly_flags)),
                'total_windows': len(mae_full),
                'anomaly_ratio': float(np.mean(anomaly_flags) * 100)
            },
            'file_paths': {
                'model_path': MODEL_PATH,
                'visualization_path': viz_path,
                'data_path': DATA_DIR
            },
            'feature_names': FEATURE_NAMES
        }

        # 9. Save results
        results_path = save_results(results_dict, MODEL_SAVE_DIR)

        # 10. Print summary report (following the output format from the first Python file)
        print("\n" + "=" * 80)
        print("SUMMARY REPORT")
        print("=" * 80)
        print(f"📊 LSTM-Autoencoder Hybrid Implementation Complete")
        print(f"\n📈 Performance Summary:")
        print(f"  - Total samples processed: {X.shape[0]:,}")
        print(f"  - Training samples: {X_train.shape[0]:,}")
        print(f"  - Validation samples: {X_val.shape[0]:,}")
        print(f"  - Model parameters: {total_params:,}")
        print(f"  - Final training loss: {history.history['loss'][-1]:.6f}")
        print(f"  - Final validation loss: {history.history['val_loss'][-1]:.6f}")
        print(f"  - Anomaly threshold (95th percentile): {threshold:.6f}")
        print(
            f"  - Anomalies detected: {np.sum(anomaly_flags):,} out of {len(mae_full):,} ({np.mean(anomaly_flags) * 100:.2f}%)")

        print(f"\n📁 Output Files:")
        print(f"  - Model: {MODEL_PATH}")
        print(f"  - Visualization: {viz_path}")
        print(f"  - Results: {results_path}")

        print("\n✅ Process completed successfully!")

    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

def print_best_validation_loss(history):
    """Print the best validation loss"""
    if history and 'val_loss' in history.history:
        best_val_loss = min(history.history['val_loss'])
        best_epoch = np.argmin(history.history['val_loss']) + 1
        print(f"\n✨ Best Validation Loss: {best_val_loss:.6f} (epoch {best_epoch})")
        return best_val_loss, best_epoch
    return None, None


# ====================== Execute Main Function ======================
if __name__ == "__main__":
    # Set matplotlib
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 100

    # Run main function
    exit_code = main()
    exit(exit_code)
