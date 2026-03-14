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
import json

warnings.filterwarnings('ignore')


# ====================== 1. Data loading function ======================
def load_data():
    """
    Load data using new file paths
    """
    print("=" * 80)
    print("CONVOLUTIONAL AUTOENCODER TRAINING PIPELINE")
    print("=" * 80)

    print("\n📂 Loading preprocessed sliding windows...")

    # New configuration parameters
    DATA_DIR = r"D:\Oswaldo's surf project\DR O's database\final_preprocessed_data_70%_anomaly_complete"
    X_PATH = os.path.join(DATA_DIR, "X_windows.npy")
    SCALED_DF_PATH = os.path.join(DATA_DIR, "preprocessed_time_series_augmented.csv")

    # Load window data
    if not os.path.exists(X_PATH):
        raise FileNotFoundError(f"X_windows.npy not found at {X_PATH}")

    X = np.load(X_PATH)
    print(f"Loaded X shape: {X.shape}")  # (n_windows, 24, n_features)

    n_samples, timesteps, n_features = X.shape
    print(f"Data dimensions: {n_samples} samples, {timesteps} timesteps, {n_features} features")

    # Load hourly data for subsequent analysis
    print("\n📂 Loading preprocessed time series data...")
    if os.path.exists(SCALED_DF_PATH):
        df_hourly = pd.read_csv(SCALED_DF_PATH, index_col=0, parse_dates=True)
        print(f"Hourly data shape: {df_hourly.shape}")

        # Get actual feature names
        actual_features = df_hourly.columns.tolist()
        print(f"Actual features in data: {actual_features}")
    else:
        print(f"Warning: {SCALED_DF_PATH} not found, creating placeholder hourly data")
        # Create placeholder data
        n_hours = n_samples + timesteps - 1
        date_range = pd.date_range(start='2013-03-01', periods=n_hours, freq='H')
        df_hourly = pd.DataFrame(
            np.random.randn(n_hours, n_features),
            index=date_range,
            columns=[f'Feature_{i}' for i in range(n_features)]
        )
        actual_features = df_hourly.columns.tolist()

    return X, df_hourly, actual_features


# ====================== 2. Data splitting function ======================
def split_data(X, split_ratio=0.85):
    """Data splitting"""
    n_samples = X.shape[0]
    split_idx = int(split_ratio * n_samples)

    X_train = X[:split_idx]
    X_val = X[split_idx:]

    print(f"\n📊 Dataset splitting:")
    print(f"  Training windows   : {X_train.shape} ({split_ratio * 100:.1f}%)")
    print(f"  Validation windows : {X_val.shape} ({100 - split_ratio * 100:.1f}%)")
    print(f"  Total windows      : {n_samples}")

    return X_train, X_val


# ====================== 3. Build Conv-AE model ======================
def build_conv_autoencoder(sequence_length=24, n_features=4):
    """
    Build an improved Conv-AE model (with residual connections)
    """
    print(f"\n🔧 Building Conv-AutoEncoder model:")
    print(f"  Input shape: ({sequence_length}, {n_features})")

    inputs = Input(shape=(sequence_length, n_features))

    # ====== Encoder ======
    # First convolutional block
    x1 = Conv1D(32, kernel_size=5, padding='same', activation='relu')(inputs)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling1D(pool_size=2, padding='same')(x1)

    # Second convolutional block
    x2 = Conv1D(64, kernel_size=3, padding='same', activation='relu')(x1)
    x2 = BatchNormalization()(x2)
    x2 = MaxPooling1D(pool_size=2, padding='same')(x2)

    # Third convolutional block
    x3 = Conv1D(128, kernel_size=3, padding='same', activation='relu')(x2)
    x3 = BatchNormalization()(x3)
    encoded = MaxPooling1D(pool_size=2, padding='same')(x3)

    # ====== Decoder ======
    # First deconvolutional block
    y1 = Conv1D(128, kernel_size=3, padding='same', activation='relu')(encoded)
    y1 = BatchNormalization()(y1)
    y1 = UpSampling1D(size=2)(y1)

    # Second deconvolutional block
    y2 = Conv1D(64, kernel_size=3, padding='same', activation='relu')(y1)
    y2 = BatchNormalization()(y2)
    y2 = UpSampling1D(size=2)(y2)

    # Third deconvolutional block
    y3 = Conv1D(32, kernel_size=3, padding='same', activation='relu')(y2)
    y3 = BatchNormalization()(y3)
    y3 = UpSampling1D(size=2)(y3)

    # Output layer
    outputs = Conv1D(n_features, kernel_size=5, padding='same', activation='linear')(y3)

    # ====== Full model ======
    autoencoder = Model(inputs, outputs, name='conv_autoencoder')

    # Count parameters
    total_params = autoencoder.count_params()
    trainable_params = np.sum([np.prod(v.shape) for v in autoencoder.trainable_weights])
    non_trainable_params = total_params - trainable_params

    print(f"\n📊 Model parameters count:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Non-trainable parameters: {non_trainable_params:,}")

    return autoencoder, total_params


# ====================== 4. Model training ======================
def train_model(autoencoder, X_train, X_val, epochs=80, batch_size=128, patience=12):
    """Train the model"""
    print(f"\n🚀 Starting training...")

    # Show model summary
    autoencoder.summary()

    # Compile the model - using MAE as loss function
    autoencoder.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mae',
        metrics=['mse']
    )

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
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
    print(f"  Batch Size: {batch_size}")
    print(f"  Max Epochs: {epochs}")

    # Train the model
    history = autoencoder.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    return history


# ====================== 5. Anomaly detection ======================
def detect_anomalies(autoencoder, X, X_val, threshold_percentile=95, batch_size=128):
    """Anomaly detection"""
    print("\n🔍 Computing reconstruction errors...")

    # Validation set reconstruction error
    reconstructions_val = autoencoder.predict(X_val, batch_size=batch_size, verbose=0)
    mae_val = np.mean(np.abs(X_val - reconstructions_val), axis=(1, 2))

    # Calculate threshold
    threshold = np.percentile(mae_val, threshold_percentile)
    print(f"Validation (assumed normal) MAE {threshold_percentile}th percentile threshold: {threshold:.6f}")

    # Full dataset reconstruction error
    reconstructions_full = autoencoder.predict(X, batch_size=batch_size, verbose=0)
    mae_full = np.mean(np.abs(X - reconstructions_full), axis=(1, 2))

    # Detect anomalies
    anomaly_flags = (mae_full > threshold).astype(int)
    anomaly_count = np.sum(anomaly_flags)
    anomaly_ratio = anomaly_count / len(mae_full) * 100

    print(f"\nDetected {anomaly_count} anomalous windows out of {len(mae_full)} ({anomaly_ratio:.2f}%)")

    return mae_full, anomaly_flags, threshold, reconstructions_full


# ====================== 6. Visualization function ======================
def visualize_results(df_hourly, mae_full, anomaly_flags, threshold,
                      X_original=None, X_reconstructed=None, history=None,
                      actual_features=None, model_name="Conv AutoEncoder"):
    """Visualize results"""
    print("\n📊 Generating visualizations...")

    # Set save path
    VISUALIZATION_SAVE_DIR = r"D:\Oswaldo's surf project\DR O's database\final_preprocessed_data_70%_anomaly_complete\visualizations"
    os.makedirs(VISUALIZATION_SAVE_DIR, exist_ok=True)

    # Set plotting style
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(20, 15))

    # 1. Training history
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

    # Create time indices
    timesteps = X_original.shape[1] if X_original is not None else 24
    if len(df_hourly) >= len(mae_full):
        time_indices = df_hourly.index[timesteps - 1: timesteps - 1 + len(mae_full)]
    else:
        time_indices = range(len(mae_full))

    ax2.plot(time_indices, mae_full, label='Reconstruction MAE',
             color='teal', alpha=0.7, linewidth=1)
    ax2.axhline(threshold, color='red', linestyle='--', linewidth=2,
                label=f'Threshold ({threshold:.5f})')

    # Mark anomaly points
    anomaly_indices = np.where(anomaly_flags == 1)[0]
    if len(time_indices) == len(mae_full):
        ax2.scatter(time_indices[anomaly_indices], mae_full[anomaly_indices],
                    color='red', marker='o', s=30, label='Detected Anomaly', alpha=0.6)

    ax2.set_title(f'{model_name} - Reconstruction Error & Detected Anomalies',
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Mean Absolute Error (per window)', fontsize=12)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)

    # 3. Reconstruction error distribution
    ax3 = plt.subplot(3, 3, 3 if history is None else 3)
    n_bins = min(100, len(mae_full) // 10)
    ax3.hist(mae_full, bins=n_bins, alpha=0.7, color='skyblue',
             edgecolor='black', density=True)
    ax3.axvline(x=threshold, color='red', linestyle='--', linewidth=2,
                label=f'Threshold ({threshold:.4f})')

    # Add Gaussian distribution fit
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

    # 4. Original vs reconstruction comparison (first sample)
    ax4 = plt.subplot(3, 3, 4 if history is None else 4)
    if X_original is not None and X_reconstructed is not None and len(X_original) > 0:
        sample_idx = 0
        time_steps = range(timesteps)

        # Plot first feature
        if actual_features and len(actual_features) > 0:
            feature_name = actual_features[0]
        else:
            feature_name = 'Feature_0'

        ax4.plot(time_steps, X_original[sample_idx, :, 0],
                 label='Original', linewidth=2, alpha=0.8)
        ax4.plot(time_steps, X_reconstructed[sample_idx, :, 0],
                 label='Reconstructed', linewidth=2, alpha=0.8,
                 color='red', linestyle='--')

        ax4.set_title(f'Sample {sample_idx} - {feature_name}',
                      fontsize=14, fontweight='bold')
        ax4.set_xlabel('Time Step (hours)', fontsize=12)
        ax4.set_ylabel('Feature Value', fontsize=12)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    # 5. Multi-feature comparison
    ax5 = plt.subplot(3, 3, 5 if history is None else 5)
    if len(df_hourly) >= 7 * 24 and actual_features:
        plot_data = df_hourly.iloc[:7 * 24]
        for i, feat in enumerate(actual_features[:4]):  # Show at most 4 features
            ax5.plot(plot_data.index, plot_data[feat], label=feat,
                     linewidth=1, alpha=0.8)
        ax5.set_title(f'Feature Comparison (First 7 Days)',
                      fontsize=14, fontweight='bold')
        ax5.set_xlabel('Time', fontsize=12)
        ax5.set_ylabel('Normalized Value', fontsize=12)
        ax5.legend(loc='upper right', fontsize=9)
        ax5.grid(True, alpha=0.3)

    # 6. Anomaly detection performance analysis
    ax6 = plt.subplot(3, 3, 6 if history is None else 6)
    thresholds = np.percentile(mae_full, range(90, 100))
    anomaly_rates = []

    for t in thresholds:
        anomaly_rate = np.sum(mae_full > t) / len(mae_full) * 100
        anomaly_rates.append(anomaly_rate)

    ax6.plot(thresholds, anomaly_rates, 'o-', linewidth=2, markersize=6)
    ax6.axvline(x=threshold, color='red', linestyle='--', linewidth=1.5,
                label=f'95th percentile ({threshold:.4f})')
    ax6.set_title('Anomaly Rate vs Threshold', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Threshold', fontsize=12)
    ax6.set_ylabel('Anomaly Rate (%)', fontsize=12)
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # 7. Error boxplot
    ax7 = plt.subplot(3, 3, 7 if history is None else 7)
    if len(anomaly_indices) > 0:
        normal_errors = mae_full[anomaly_flags == 0]
        anomaly_errors = mae_full[anomaly_flags == 1]

        box_data = [normal_errors, anomaly_errors]
        box_labels = ['Normal', 'Anomaly']

        bp = ax7.boxplot(box_data, labels=box_labels, patch_artist=True)

        colors = ['lightblue', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax7.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5)
        ax7.set_title('Error Distribution by Category', fontsize=14, fontweight='bold')
        ax7.set_ylabel('Reconstruction MAE', fontsize=12)
        ax7.grid(True, alpha=0.3, axis='y')

    # 8. Feature heatmap (reconstruction error)
    ax8 = plt.subplot(3, 3, 8 if history is None else 8)
    if X_original is not None and X_reconstructed is not None and len(X_original) > 0:
        # Calculate feature-level error for the first sample
        sample_idx = 0
        feature_errors = np.abs(X_original[sample_idx] - X_reconstructed[sample_idx])

        im = ax8.imshow(feature_errors.T, aspect='auto', cmap='YlOrRd')
        ax8.set_xlabel('Time Step', fontsize=12)
        ax8.set_ylabel('Feature Index', fontsize=12)
        ax8.set_title(f'Sample {sample_idx} Feature-wise MAE',
                      fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax8, label='MAE')

        # Set feature labels
        n_features = feature_errors.shape[1]
        if actual_features and len(actual_features) >= n_features:
            feature_labels = [f'F{i + 1}:{actual_features[i][:10]}...' if len(
                actual_features[i]) > 10 else f'F{i + 1}:{actual_features[i]}'
                              for i in range(n_features)]
        else:
            feature_labels = [f'F{i + 1}' for i in range(n_features)]

        ax8.set_yticks(range(n_features))
        ax8.set_yticklabels(feature_labels)

    # 9. Model comparison
    ax9 = plt.subplot(3, 3, 9 if history is None else 9)

    # If X_original has shape information, show model compression ratio
    if X_original is not None:
        original_size = X_original.shape[1] * X_original.shape[2]

        # Compression in Conv-AE occurs in Conv layers, simplified here to calculate encoded size
        # Actually Conv-AE compresses through pooling, final bottleneck size is 3*128=384 (assuming input 24*4=96)
        encoded_size = 3 * 128  # Estimated size after 3 pooling layers

        compression_ratio = original_size / encoded_size

        summary_text = f"""
        {model_name} Summary:

        Input Shape: {X_original.shape[1]}×{X_original.shape[2]}
        Original Size: {original_size}
        Encoded Size: ~{encoded_size}
        Compression Ratio: {compression_ratio:.1f}x

        Total Parameters: {history.model.count_params() if history else 'N/A':,}

        Anomaly Threshold: {threshold:.6f}
        Anomalies Detected: {np.sum(anomaly_flags):,}
        Anomaly Ratio: {np.mean(anomaly_flags) * 100:.2f}%
        """

        ax9.text(0.1, 0.95, summary_text, fontsize=10,
                 verticalalignment='top', family='monospace')
        ax9.axis('off')

    # Adjust layout
    plt.tight_layout()

    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(VISUALIZATION_SAVE_DIR, f"conv_ae_visualization_{timestamp}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📸 Visualization saved to: {save_path}")

    plt.close(fig)

    return save_path



# ====================== 7.1 Helper function for type conversion ======================
def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (bool, str, int, float)):
        return obj
    elif obj is None:
        return None
    else:
        # For other types, try to convert to string
        try:
            return str(obj)
        except:
            return f"<{type(obj).__name__}>"



# ====================== 7. Save results ======================
def save_results(results_dict, save_dir):
    """Save training results to a JSON file"""
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(save_dir, f"conv_ae_results_{timestamp}.json")

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)

    print(f"💾 Results saved to: {results_path}")
    return results_path


# ====================== 8. Main workflow ======================
def main():
    """Main function"""
    # Set TensorFlow optimizations
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    print("=" * 80)
    print("CONVOLUTIONAL AUTOENCODER TRAINING PIPELINE")
    print("=" * 80)

    import time
    start_time = time.time()

    try:
        # 1. Load data
        X, df_hourly, actual_features = load_data()

        # Use actual number of features
        n_features = X.shape[2]
        timesteps = X.shape[1]
        print(f"\n📊 Data Information:")
        print(f"  Number of features: {n_features}")
        print(f"  Timesteps per window: {timesteps}")
        print(f"  Feature names: {actual_features if 'actual_features' in locals() else 'Not available'}")

        # 2. Split data
        X_train, X_val = split_data(X, split_ratio=0.85)

        # 3. Build model
        print("\n" + "=" * 80)
        print("MODEL CONSTRUCTION")
        print("=" * 80)

        autoencoder, total_params = build_conv_autoencoder(timesteps, n_features)

        # 4. Train model
        print("\n" + "=" * 80)
        print("MODEL TRAINING")
        print("=" * 80)

        history = train_model(
            autoencoder, X_train, X_val,
            epochs=80,
            batch_size=128,
            patience=12
        )

        # Calculate best validation loss
        if history is not None and 'val_loss' in history.history:
            best_val_loss = min(history.history['val_loss'])
            best_epoch = np.argmin(history.history['val_loss']) + 1
            final_train_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1]

            print(f"\n" + "=" * 60)
            print("VALIDATION LOSS SUMMARY")
            print("=" * 60)
            print(f"Training completed:")
            print(f"  - Best validation loss: {best_val_loss:.6f} (epoch {best_epoch})")
            print(f"  - Final training loss: {final_train_loss:.6f}")
            print(f"  - Final validation loss: {final_val_loss:.6f}")
            print(f"  - Total training epochs: {len(history.history['loss'])}")
            print("=" * 60)
        else:
            best_val_loss = history.history['val_loss'][-1] if history else None
            best_epoch = None

        # 5. Save model
        model_save_dir = r"D:\Oswaldo's surf project\DR O's database\final_preprocessed_data_70%_anomaly_complete\models"
        os.makedirs(model_save_dir, exist_ok=True)
        model_path = os.path.join(model_save_dir, "conv_ae_autoencoder.h5")

        autoencoder.save(model_path)
        print(f"\n💾 Model saved to: {model_path}")

        # 6. Anomaly detection
        print("\n" + "=" * 80)
        print("ANOMALY DETECTION")
        print("=" * 80)

        mae_full, anomaly_flags, threshold, reconstructions_full = detect_anomalies(
            autoencoder, X, X_val, threshold_percentile=95, batch_size=128
        )

        # 7. Visualization
        print("\n" + "=" * 80)
        print("VISUALIZATION")
        print("=" * 80)

        viz_path = visualize_results(
            df_hourly, mae_full, anomaly_flags, threshold,
            X_original=X, X_reconstructed=reconstructions_full, history=history,
            actual_features=actual_features, model_name="Conv AutoEncoder"
        )

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


        # 8. Prepare results dictionary
        results_dict = {
            'model_name': 'Conv_AutoEncoder',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_info': {
                'total_samples': X.shape[0],
                'timesteps': timesteps,
                'n_features': n_features,
                'training_samples': X_train.shape[0],
                'validation_samples': X_val.shape[0],
                'actual_features': actual_features
            },
            'model_info': {
                'total_params': total_params,
                'batch_size': 128,
                'epochs_trained': len(history.history['loss']),
                'final_train_loss': float(history.history['loss'][-1]),
                'final_val_loss': float(history.history['val_loss'][-1]),
                'best_val_loss': float(best_val_loss) if best_val_loss else None,
                'best_epoch': best_epoch
            },
            'anomaly_detection': {
                'threshold_percentile': 95,
                'threshold_value': float(threshold),
                'anomalies_detected': int(np.sum(anomaly_flags)),
                'total_windows': len(mae_full),
                'anomaly_ratio': float(np.mean(anomaly_flags) * 100)
            },
            'file_paths': {
                'model_path': model_path,
                'visualization_path': viz_path,
                'data_path': r"D:\Oswaldo's surf project\DR O's database\final_preprocessed_data_70%_anomaly_complete"
            }
        }

        # Convert numpy types to Python native types
        results_dict = convert_numpy_types(results_dict)

        # 9. Save results
        results_path = save_results(results_dict, model_save_dir)

        # 10. Print summary report
        print("\n" + "=" * 80)
        print("SUMMARY REPORT")
        print("=" * 80)
        print(f"📊 Conv AutoEncoder Training Complete")
        print(f"\n📈 Performance Summary:")
        print(f"  - Total samples processed: {X.shape[0]:,}")
        print(f"  - Training samples: {X_train.shape[0]:,}")
        print(f"  - Validation samples: {X_val.shape[0]:,}")
        print(f"  - Model parameters: {total_params:,}")
        print(f"  - Final training loss: {history.history['loss'][-1]:.6f}")
        print(f"  - Final validation loss: {history.history['val_loss'][-1]:.6f}")
        print(f"  - Best validation loss: {best_val_loss:.6f} (epoch {best_epoch})")
        print(f"  - Anomaly threshold (95th percentile): {threshold:.6f}")
        print(
            f"  - Anomalies detected: {np.sum(anomaly_flags):,} out of {len(mae_full):,} ({np.mean(anomaly_flags) * 100:.2f}%)")

        print(f"\n📁 Output Files:")
        print(f"  - Model: {model_path}")
        print(f"  - Visualization: {viz_path}")
        print(f"  - Results: {results_path}")

        print("\n✅ Process completed successfully!")

    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


# ====================== Execute main function ======================
if __name__ == "__main__":
    # Set matplotlib
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 100

    # Run main function
    exit_code = main()
    exit(exit_code)
