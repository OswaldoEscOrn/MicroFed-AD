import json
import os
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings('ignore')


# ====================== 1. Data loading (modified paths) ======================
def load_real_data():
    """
    Load real data using new file paths
    """
    # New configuration parameters
    DATA_DIR = r"D:\Oswaldo's surf project\DR O's database\final_preprocessed_data_complete"
    X_PATH = os.path.join(DATA_DIR, "X_windows.npy")
    SCALED_DF_PATH = os.path.join(DATA_DIR, "preprocessed_time_series_augmented.csv")

    print("Loading preprocessed sliding windows...")

    # Load sliding window data
    X = np.load(X_PATH)
    print(f"Loaded X shape: {X.shape}")  # (n_windows, 24, n_features)

    # Load hourly data
    df_hourly = pd.read_csv(SCALED_DF_PATH, index_col=0, parse_dates=True)
    print(f"Hourly data shape: {df_hourly.shape}")

    # Time series split: 85% training, 15% validation
    n_samples = X.shape[0]
    split_idx = int(0.85 * n_samples)
    X_train = X[:split_idx]
    X_val = X[split_idx:]

    print(f"Training windows   : {X_train.shape}")
    print(f"Validation windows : {X_val.shape}")

    return X, X_train, X_val, df_hourly


# ====================== 2. Data preparation function ======================
def prepare_dae_input(X_windows, mode='flatten'):
    """
    Prepare input data for AutoEncoder

    Parameters:
    X_windows: original window data, shape (n_windows, 24, n_features)
    mode: 'flatten', 'hour_only', 'time_series'
    """
    n_samples = X_windows.shape[0]
    n_timesteps = X_windows.shape[1]
    n_features = X_windows.shape[2]

    if mode == 'flatten':
        # Flatten to (n_samples, 24*n_features)
        X = X_windows.reshape(n_samples, -1)
        print(f"📊 Using flattened features: {X.shape}")

    elif mode == 'hour_only':
        # Use only the first timestep features
        X = X_windows[:, 0, :]
        print(f"📊 Using first hour features: {X.shape}")

    elif mode == 'time_series':
        # Keep time series structure
        X = X_windows
        print(f"📊 Using time series features: {X.shape}")

    return X


# ====================== 3. Deep AutoEncoder construction ======================
def build_deep_autoencoder(input_dim, encoding_dim=32):
    """
    Build Deep AutoEncoder model
    """
    print(f"\n🔧 Building Deep AutoEncoder model:")
    print(f"  Input dimension: {input_dim}")
    print(f"  Encoding dimension: {encoding_dim}")

    # ====== Encoder ======
    encoder = Sequential([
        Input(shape=(input_dim,), name='input'),
        Dense(256, activation='relu', name='encoder_dense1'),
        BatchNormalization(name='encoder_bn1'),
        Dropout(0.3, name='encoder_dropout1'),

        Dense(128, activation='relu', name='encoder_dense2'),
        BatchNormalization(name='encoder_bn2'),
        Dropout(0.3, name='encoder_dropout2'),

        Dense(64, activation='relu', name='encoder_dense3'),
        BatchNormalization(name='encoder_bn3'),
        Dropout(0.2, name='encoder_dropout3'),

        Dense(encoding_dim, activation='relu', name='bottleneck')
    ], name='encoder')

    # ====== Decoder ======
    decoder = Sequential([
        Input(shape=(encoding_dim,), name='decoder_input'),
        Dense(64, activation='relu', name='decoder_dense1'),
        BatchNormalization(name='decoder_bn1'),
        Dropout(0.2, name='decoder_dropout1'),

        Dense(128, activation='relu', name='decoder_dense2'),
        BatchNormalization(name='decoder_bn2'),
        Dropout(0.3, name='decoder_dropout2'),

        Dense(256, activation='relu', name='decoder_dense3'),
        BatchNormalization(name='decoder_bn3'),
        Dropout(0.3, name='decoder_dropout3'),

        Dense(input_dim, activation='linear', name='output')
    ], name='decoder')

    # ====== Full AutoEncoder ======
    input_layer = Input(shape=(input_dim,), name='autoencoder_input')
    encoded = encoder(input_layer)
    decoded = decoder(encoded)
    autoencoder = Model(input_layer, decoded, name='autoencoder')

    return autoencoder, encoder, decoder


# ====================== 4. Model training ======================
def train_autoencoder(X_train, X_val, encoding_dim=32, epochs=80):
    """
    Train AutoEncoder
    """
    input_dim = X_train.shape[1]

    # Build model
    autoencoder, encoder, decoder = build_deep_autoencoder(input_dim, encoding_dim)

    # Compile model
    autoencoder.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    print("\n📋 Model Architecture Summary:")
    autoencoder.summary()

    # Calculate parameter count
    total_params = autoencoder.count_params()
    trainable_params = np.sum([np.prod(v.shape) for v in autoencoder.trainable_weights])
    non_trainable_params = total_params - trainable_params

    print(f"\n📊 Model Parameter Statistics:")
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  Non-trainable params: {non_trainable_params:,}")
    print(f"  Memory usage: {total_params * 4 / 1024:.2f} KB")

    # Callbacks
    callbacks = [
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # Train model
    print("\n🚀 Start training Deep AutoEncoder...")
    print(f"  Training set size: {X_train.shape}")
    print(f"  Validation set size: {X_val.shape}")
    print(f"  Batch size: 128")
    print(f"  Max epochs: {epochs}")

    history = autoencoder.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=epochs,
        batch_size=128,
        callbacks=callbacks,
        verbose=1
    )

    # Calculate best validation loss
    print("\n" + "=" * 60)
    print("VALIDATION LOSS SUMMARY")
    print("=" * 60)

    if history is not None and 'val_loss' in history.history:
        best_val_loss = min(history.history['val_loss'])
        best_epoch = np.argmin(history.history['val_loss']) + 1
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]

        print(f"Training completed:")
        print(f"  - Best validation loss: {best_val_loss:.6f} (epoch {best_epoch})")
        print(f"  - Final training loss: {final_train_loss:.6f}")
        print(f"  - Final validation loss: {final_val_loss:.6f}")
        print(f"  - Total training epochs: {len(history.history['loss'])}")

        # If MSE metric exists
        if 'val_mse' in history.history:
            best_val_mse = min(history.history['val_mse'])
            final_val_mse = history.history['val_mse'][-1]
            print(f"  - Best validation MSE: {best_val_mse:.6f}")
            print(f"  - Final validation MSE: {final_val_mse:.6f}")
    else:
        print("Warning: No training history available")
        best_val_loss = None
        best_epoch = None

    print("=" * 60)

    return autoencoder, encoder, decoder, history, total_params, best_val_loss, best_epoch


# ====================== 5. Evaluation and anomaly detection ======================
def evaluate_and_detect_anomalies(autoencoder, X_val, X_test=None, percentile=95):
    """
    Evaluate model and detect anomalies
    """
    print("\n📈 Model Evaluation Results:")

    # Evaluate on validation set
    val_loss, val_mae = autoencoder.evaluate(X_val, X_val, verbose=0)
    print(f"  Validation loss (MSE): {val_loss:.6f}")
    print(f"  Validation MAE: {val_mae:.6f}")

    # Calculate validation set reconstruction errors
    val_reconstructed = autoencoder.predict(X_val, verbose=0)
    val_errors = np.mean((X_val - val_reconstructed) ** 2, axis=1)

    # Calculate threshold (95th percentile)
    threshold = np.percentile(val_errors, percentile)
    print(f"  Validation set reconstruction error {percentile}th percentile threshold: {threshold:.6f}")

    # If test set is provided, detect anomalies on test set
    if X_test is not None:
        test_reconstructed = autoencoder.predict(X_test, verbose=0)
        test_errors = np.mean((X_test - test_reconstructed) ** 2, axis=1)

        # Detect anomalies
        anomalies = test_errors > threshold
        anomalies_count = np.sum(anomalies)
        anomaly_ratio = anomalies_count / len(X_test) * 100

        print(f"  Detected {anomalies_count} anomalous samples out of {len(X_test)} total samples")
        print(f"  Anomaly ratio: {anomaly_ratio:.2f}%")

        return threshold, anomalies_count, anomaly_ratio, test_errors, test_reconstructed
    else:
        # Detect anomalies only on validation set
        anomalies = val_errors > threshold
        anomalies_count = np.sum(anomalies)
        anomaly_ratio = anomalies_count / len(X_val) * 100

        print(f"  Anomalies detected in validation set: {anomalies_count:,} / {len(X_val):,} ({anomaly_ratio:.2f}%)")

        return threshold, anomalies_count, anomaly_ratio, val_errors, val_reconstructed


# ====================== 6. Visualization function ======================
def visualize_results(history, X_original, X_reconstructed, errors, threshold,
                      df_hourly, model_name="Deep AutoEncoder"):
    """
    Visualize training results and anomaly detection
    """
    print(f"\n📊 Generating visualization results...")

    # Set save path
    SAVE_DIR = r"D:\Oswaldo's surf project\DR O's database\final_preprocessed_data_complete\visualizations"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Set plotting style
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(20, 15))

    # 1. Training loss history
    ax1 = plt.subplot(3, 3, 1)
    epochs = range(1, len(history.history['loss']) + 1)
    ax1.plot(epochs, history.history['loss'], label='Training Loss', linewidth=2)
    ax1.plot(epochs, history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_title(f'{model_name} - Training History', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('MSE Loss', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. MAE history
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(epochs, history.history['mae'], label='Training MAE', linewidth=2, color='orange')
    if 'val_mae' in history.history:
        ax2.plot(epochs, history.history['val_mae'], label='Validation MAE', linewidth=2, color='red')
    ax2.set_title('MAE History', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Mean Absolute Error', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Reconstruction error distribution
    ax3 = plt.subplot(3, 3, 3)
    n_bins = min(100, len(errors) // 10)
    ax3.hist(errors, bins=n_bins, alpha=0.7, color='skyblue', edgecolor='black', density=True)
    ax3.axvline(x=threshold, color='red', linestyle='--', linewidth=2,
                label=f'Threshold ({threshold:.4f})')

    # Add Gaussian distribution fit
    from scipy.stats import norm
    mu, std = norm.fit(errors)
    xmin, xmax = ax3.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    ax3.plot(x, p, 'k', linewidth=2, label=f'Normal fit: μ={mu:.4f}, σ={std:.4f}')

    ax3.set_title('Reconstruction Error Distribution', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Reconstruction Error (MSE)', fontsize=12)
    ax3.set_ylabel('Density', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Original vs reconstructed feature comparison (5 random samples)
    ax4 = plt.subplot(3, 3, 4)
    n_samples_to_show = min(5, len(X_original))

    # If flattened data, choose appropriate number of features to display
    if len(X_original.shape) == 2:
        n_features = min(100, X_original.shape[1])
    else:
        n_features = min(100, X_original.shape[1] * X_original.shape[2])

    for i in range(n_samples_to_show):
        sample_idx = np.random.randint(0, len(X_original))

        if len(X_original.shape) == 2:
            # Flattened data
            x_indices = range(n_features)
            if i == 0:
                ax4.scatter(x_indices, X_original[sample_idx][:n_features],
                            alpha=0.5, label='Original', s=20)
                ax4.scatter(x_indices, X_reconstructed[sample_idx][:n_features],
                            alpha=0.5, label='Reconstructed', s=20, color='red')
            else:
                ax4.scatter(x_indices, X_original[sample_idx][:n_features],
                            alpha=0.3, s=10)
                ax4.scatter(x_indices, X_reconstructed[sample_idx][:n_features],
                            alpha=0.3, s=10, color='red')
        else:
            # Time series data, display the first feature
            if i == 0:
                ax4.plot(X_original[sample_idx, :, 0], alpha=0.5, label='Original', linewidth=1)
                ax4.plot(X_reconstructed[sample_idx, :, 0], alpha=0.5, label='Reconstructed',
                         linewidth=1, color='red')
            else:
                ax4.plot(X_original[sample_idx, :, 0], alpha=0.3, linewidth=0.5)
                ax4.plot(X_reconstructed[sample_idx, :, 0], alpha=0.3, linewidth=0.5, color='red')

    ax4.set_title(f'Original vs Reconstructed (5 Random Samples)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Feature Index', fontsize=12)
    ax4.set_ylabel('Feature Value', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Reconstruction error time series
    ax5 = plt.subplot(3, 3, 5)
    sample_indices = range(len(errors))

    # Smooth the error
    window_size = min(100, len(errors) // 10)
    if window_size > 1:
        errors_smooth = np.convolve(errors, np.ones(window_size) / window_size, mode='valid')
        indices_smooth = sample_indices[:len(errors_smooth)]
        ax5.plot(indices_smooth, errors_smooth, linewidth=1, alpha=0.7, color='blue', label='Smoothed Error')

    ax5.plot(sample_indices, errors, linewidth=0.5, alpha=0.3, color='gray', label='Raw Error')
    ax5.axhline(y=threshold, color='red', linestyle='--', linewidth=2,
                label=f'Threshold ({threshold:.4f})')

    # Mark anomalies
    anomalies = errors > threshold
    anomaly_indices = np.where(anomalies)[0]
    ax5.scatter(anomaly_indices, errors[anomaly_indices],
                color='red', s=10, alpha=0.5, label='Anomalies')

    ax5.set_title('Reconstruction Error Time Series', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Sample Index', fontsize=12)
    ax5.set_ylabel('Reconstruction Error', fontsize=12)
    ax5.legend(loc='upper right', fontsize=10)
    ax5.grid(True, alpha=0.3)

    # 6. Error boxplot
    ax6 = plt.subplot(3, 3, 6)
    box_data = [errors[~anomalies], errors[anomalies]] if len(anomaly_indices) > 0 else [errors]
    box_labels = ['Normal', 'Anomaly'] if len(anomaly_indices) > 0 else ['All Samples']

    bp = ax6.boxplot(box_data, labels=box_labels, patch_artist=True)

    # Set colors
    colors = ['lightblue', 'lightcoral'] if len(anomaly_indices) > 0 else ['lightblue']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax6.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5)
    ax6.set_title('Error Distribution by Category', fontsize=14, fontweight='bold')
    ax6.set_ylabel('Reconstruction Error', fontsize=12)
    ax6.grid(True, alpha=0.3, axis='y')

    # 7. Learning rate changes
    ax7 = plt.subplot(3, 3, 7)
    if 'lr' in history.history:
        ax7.plot(epochs, history.history['lr'], linewidth=2, color='green')
        ax7.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax7.set_xlabel('Epoch', fontsize=12)
        ax7.set_ylabel('Learning Rate', fontsize=12)
        ax7.grid(True, alpha=0.3)
    else:
        ax7.text(0.5, 0.5, 'Learning Rate\nData Not Available',
                 ha='center', va='center', transform=ax7.transAxes, fontsize=12)
        ax7.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')

    # 8. Feature importance heatmap
    ax8 = plt.subplot(3, 3, 8)
    if len(X_original) > 0:
        # Calculate reconstruction error per feature
        if len(X_original.shape) == 2:
            # Flattened data
            feature_errors = np.mean((X_original - X_reconstructed) ** 2, axis=0)

            # Try to reshape to 24 hours * number of features structure
            n_features_total = X_original.shape[1]

            # First check if number of features is divisible by 24
            if n_features_total % 24 == 0:
                n_hourly_features = n_features_total // 24
                hour_errors = feature_errors.reshape(24, n_hourly_features)

                # Get feature names
                feature_names = []
                if df_hourly is not None:
                    # Get numeric column names
                    numeric_cols = df_hourly.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) >= n_hourly_features:
                        feature_names = list(numeric_cols[:n_hourly_features])
                    else:
                        feature_names = [f'Feat_{i}' for i in range(n_hourly_features)]
                else:
                    feature_names = [f'Feat_{i}' for i in range(n_hourly_features)]

                im = ax8.imshow(hour_errors, aspect='auto', cmap='YlOrRd')
                ax8.set_xlabel('Feature', fontsize=12)
                ax8.set_ylabel('Hour of Day', fontsize=12)
                ax8.set_title('Hourly Feature Reconstruction Error', fontsize=14, fontweight='bold')

                # Set x-axis labels
                ax8.set_xticks(range(n_hourly_features))
                ax8.set_xticklabels(feature_names, rotation=45, ha='right')

                plt.colorbar(im, ax=ax8, label='Reconstruction Error')

                # Set y-axis labels
                ax8.set_yticks(range(0, 24, 3))
                ax8.set_yticklabels([f'{h:02d}:00' for h in range(0, 24, 3)])
            else:
                # Display feature error bar chart
                n_top_features = min(20, len(feature_errors))
                top_indices = np.argsort(feature_errors)[-n_top_features:][::-1]
                top_errors = feature_errors[top_indices]

                ax8.barh(range(n_top_features), top_errors, color='skyblue')
                ax8.set_yticks(range(n_top_features))
                ax8.set_yticklabels([f'Feat_{i}' for i in top_indices])
                ax8.set_title(f'Top {n_top_features} Features by Reconstruction Error',
                              fontsize=14, fontweight='bold')
                ax8.set_xlabel('Average Reconstruction Error', fontsize=12)

    # 9. Anomaly detection performance
    ax9 = plt.subplot(3, 3, 9)
    if np.sum(errors > threshold) > 0:
        # Compute performance at different thresholds
        thresholds = np.percentile(errors, range(90, 100))
        anomaly_rates = []

        for t in thresholds:
            anomaly_rate = np.sum(errors > t) / len(errors) * 100
            anomaly_rates.append(anomaly_rate)

        ax9.plot(thresholds, anomaly_rates, 'o-', linewidth=2, markersize=6)
        ax9.axvline(x=threshold, color='red', linestyle='--', linewidth=1.5,
                    label=f'95th percentile ({threshold:.4f})')
        ax9.set_title('Anomaly Rate vs Threshold', fontsize=14, fontweight='bold')
        ax9.set_xlabel('Threshold', fontsize=12)
        ax9.set_ylabel('Anomaly Rate (%)', fontsize=12)
        ax9.legend()
        ax9.grid(True, alpha=0.3)
    else:
        ax9.text(0.5, 0.5, 'No Anomalies Detected\nwith Current Threshold',
                 ha='center', va='center', transform=ax9.transAxes, fontsize=12)
        ax9.set_title('Anomaly Detection Performance', fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(SAVE_DIR, f'deep_ae_visualization_{timestamp}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📸 Visualization results saved to: {save_path}")

    plt.close()

    return save_path


# ====================== 7. Main workflow ======================
def main():
    print("=" * 80)
    print("DEEP AUTOENCODER TRAINING PIPELINE")
    print("=" * 80)

    import time
    start_time = time.time()

    # 1. Load real data
    X_all, X_train, X_val, df_hourly = load_real_data()

    # 2. Prepare training data (flatten window data)
    print("\n📥 Preparing training data:")
    mode = 'flatten'  # Use flatten mode
    X_train_flat = prepare_dae_input(X_train, mode=mode)
    X_val_flat = prepare_dae_input(X_val, mode=mode)
    X_all_flat = prepare_dae_input(X_all, mode=mode)

    print(f"\n📊 Dataset splitting (85% training, 15% validation):")
    print(f"  Training Set: {X_train.shape[0]:,} samples ({X_train.shape[0] / X_all.shape[0] * 100:.0f}%)")
    print(f"  Validation Set: {X_val.shape[0]:,} samples ({X_val.shape[0] / X_all.shape[0] * 100:.0f}%)")
    print(f"  Total Samples: {X_all.shape[0]:,}")

    # 3. Train AutoEncoder
    autoencoder, encoder, decoder, history, total_params, best_val_loss, best_epoch = train_autoencoder(
        X_train_flat, X_val_flat,
        encoding_dim=32,  # Encoding dimension
        epochs=80
    )

    # Use best validation loss
    print(f"\n📊 Deep-AE best validation loss: {best_val_loss:.6f} (epoch {best_epoch})")

    # 4. Evaluate and detect anomalies on validation set
    print("\n🔍 Evaluating Deep_AE model on validation set:")
    threshold, val_anomalies_count, val_anomaly_ratio, val_errors, val_reconstructed = evaluate_and_detect_anomalies(
        autoencoder, X_val_flat, X_val_flat, percentile=95
    )

    # 5. Evaluate on the complete dataset
    print("\n🔍 Evaluating on the complete dataset:")
    all_reconstructed = autoencoder.predict(X_all_flat, verbose=0)
    all_errors = np.mean((X_all_flat - all_reconstructed) ** 2, axis=1)

    # Detect anomalies
    all_anomalies = all_errors > threshold
    all_anomalies_count = np.sum(all_anomalies)
    all_anomaly_ratio = all_anomalies_count / len(X_all_flat) * 100

    print(f"  Total samples: {len(X_all_flat):,}")
    print(f"  Anomalies detected: {all_anomalies_count:,}")
    print(f"  Anomaly proportion: {all_anomaly_ratio:.2f}%")

    # 6. Visualize results
    print("\n📊 Generating visualizations...")
    visualization_path = visualize_results(
        history, X_all_flat, all_reconstructed, all_errors, threshold,
        df_hourly, model_name="Deep AutoEncoder"
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

    # 7. Save models
    print("\n💾 Saving models...")
    model_save_path = r"D:\Oswaldo's surf project\DR O's database\final_preprocessed_data_complete\models"
    os.makedirs(model_save_path, exist_ok=True)

    # Save in HDF5 format
    autoencoder.save(os.path.join(model_save_path, "deep_ae_autoencoder.h5"))
    encoder.save(os.path.join(model_save_path, "deep_ae_encoder.h5"))
    decoder.save(os.path.join(model_save_path, "deep_ae_decoder.h5"))

    print(f"  Models saved to: {model_save_path}")

    # 8. Generate analysis report
    print("\n" + "=" * 80)
    print("📋 Deep_AE Training Completion Report")
    print("=" * 80)

    # Output format consistent with PDF
    print(f"\n📊 Deep_AE Model Performance Summary:")
    print(f"  Total samples: {X_all.shape[0]:,}")
    print(f"  Training samples: {X_train.shape[0]:,}")
    print(f"  Validation samples: {X_val.shape[0]:,}")
    print(f"  Total params: {total_params:,}")
    print(f"  Validation loss (MSE): {history.history['val_loss'][-1]:.6f}")
    print(f"  95th percentile threshold (validation): {threshold:.6f}")
    print(f"  Anomaly detection (all samples): {all_anomalies_count:,} / {X_all.shape[0]:,} ({all_anomaly_ratio:.2f}%)")
    print(f"  Anomaly detection (validation): {val_anomalies_count:,} / {X_val.shape[0]:,} ({val_anomaly_ratio:.2f}%)")

    # Feature information
    print(f"\n📊 Feature Information:")
    print(f"  Original window shape: {X_all.shape}")
    print(f"  Flattened input dimension: {X_all_flat.shape[1]}")
    print(f"  Compression ratio: {X_all_flat.shape[1]}:32 = {X_all_flat.shape[1] / 32:.1f}x")

    # Save detailed results
    results = {
        'model_name': 'Deep_AutoEncoder',
        'total_params': int(total_params),
        'n_all_samples': int(X_all.shape[0]),
        'n_train_samples': int(X_train.shape[0]),
        'n_val_samples': int(X_val.shape[0]),
        'input_dimension': int(X_all_flat.shape[1]),
        'latent_dimension': 32,
        'compression_ratio': float(X_all_flat.shape[1] / 32),
        'validation_threshold': float(threshold),
        'validation_anomalies': int(val_anomalies_count),
        'all_anomalies': int(all_anomalies_count),
        'anomaly_ratio_all': float(all_anomaly_ratio),
        'anomaly_ratio_val': float(val_anomaly_ratio),
        'final_train_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
        'final_train_mae': float(history.history['mae'][-1]),
        'final_val_mae': float(history.history.get('val_mae', [0])[-1]),
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'visualization_path': visualization_path,
        'model_save_path': model_save_path
    }

    results_path = os.path.join(model_save_path, 'deep_ae_training_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Training results saved to: {results_path}")
    print("\n🎉 Deep AutoEncoder training pipeline completed!")


# ====================== 8. Run main workflow ======================
if __name__ == "__main__":
    # Set TensorFlow log level
    tf.get_logger().setLevel('ERROR')

    # Set matplotlib
    plt.rcParams['axes.unicode_minus'] = False

    # Run main workflow
    main()
