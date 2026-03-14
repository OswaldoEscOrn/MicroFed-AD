import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

# ==============================================
# Configuration Parameters
# ==============================================
DATA_PATH = r"D:\Oswaldo's surf project\Mixed_Real_Virtual_Database_Hourly\preprocessed_data_mixed\normalized_hourly_data_augmented_100k.csv"
OUTPUT_DIR = r"D:\Oswaldo's surf project\Mixed_Real_Virtual_Database_Hourly\100k_windows_forming"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Window parameters (consistent with original code)
WINDOW_SIZE = 24  # 24 hours = 1 day (hourly data)
STRIDE = 1  # Stride between windows (1 = overlapping windows)
PREDICTION_HORIZON = 1  # For prediction; for anomaly detection usually 0 or 1

# Features for anomaly detection (using augmented normalized features)
FEATURES = [
    'avg_PM2.5_scaled',
    'total_noise_duration_scaled',
    'noise_event_count_scaled',
    'avg_salience_scaled'
    # 'avg_PM2.5' can be added if needed
]

# ==============================================
# 1. Load and inspect augmented dataset
# ==============================================
print("=" * 60)
print("Loading augmented time series data...")
print("=" * 60)

# Load data, first column is timestamp
df = pd.read_csv(DATA_PATH)

# Rename first column to timestamp
time_column = df.columns[0]
df = df.rename(columns={time_column: 'timestamp'})

# Convert to datetime and set as index
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df = df.set_index('timestamp')
df = df.sort_index()

print(f"Data shape: {df.shape}")
print(f"Time range: {df.index.min()} → {df.index.max()}")
print(f"Total samples: {len(df)}")
print("Column names:", df.columns.tolist())

# Check missing values
print("\nMissing value check:")
print(df.isna().sum())

# If there are missing values, fill them
if df.isna().sum().sum() > 0:
    print("Found missing values, filling...")
    df = df.fillna(method='ffill').fillna(method='bfill')
    print("Missing values after filling:", df.isna().sum().sum())

# ==============================================
# 2. Check if required features exist
# ==============================================
print("\n" + "=" * 60)
print("Checking feature columns...")
print("=" * 60)

available_features = []
missing_features = []

for feature in FEATURES:
    if feature in df.columns:
        available_features.append(feature)
        print(f"✓ {feature}: exists")
    else:
        missing_features.append(feature)
        print(f"✗ {feature}: missing")

# If there are missing features, try alternative names
if missing_features:
    print("\nAttempting to find alternative features...")
    for feature in missing_features[:]:  # iterate over a copy
        if '_scaled' in feature:
            # Try removing _scaled suffix
            alternative = feature.replace('_scaled', '')
            if alternative in df.columns:
                print(f"  Using {alternative} instead of {feature}")
                df[feature] = df[alternative]
                available_features.append(feature)
                missing_features.remove(feature)

if missing_features:
    print(f"Warning: The following features are still missing: {missing_features}")
    print("Proceeding with available features...")

if not available_features:
    print("Error: No available feature columns!")
    exit()

print(f"\nThe following features will be used to create windows: {available_features}")

# ==============================================
# 3. Verify statistical properties of the data
# ==============================================
print("\n" + "=" * 60)
print("Data statistical properties")
print("=" * 60)

for feature in available_features:
    if feature in df.columns:
        mean_val = df[feature].mean()
        std_val = df[feature].std()
        min_val = df[feature].min()
        max_val = df[feature].max()
        print(f"{feature}:")
        print(f"  Mean: {mean_val:.6f}, Std: {std_val:.6f}")
        print(f"  Range: [{min_val:.6f}, {max_val:.6f}]")

# ==============================================
# 4. Create sliding windows
# ==============================================
print("\n" + "=" * 60)
print(f"Creating sliding windows (size={WINDOW_SIZE}, stride={STRIDE}, horizon={PREDICTION_HORIZON})")
print("=" * 60)


def create_sliding_windows(data, window_size, stride=1, horizon=0):
    """
    Create 3D sliding window array: (n_samples, window_size, n_features)
    If horizon > 0 → also returns prediction target values
    """
    X = []
    y = [] if horizon > 0 else None

    n_samples = len(data)

    for i in range(0, n_samples - window_size - horizon + 1, stride):
        window = data[i: i + window_size]
        X.append(window)

        if horizon > 0:
            target = data[i + window_size: i + window_size + horizon]
            y.append(target)

    X = np.array(X)
    if y is not None:
        y = np.array(y)

    return X, y


# Create sliding windows using normalized features
data_array = df[available_features].values

X_windows, y_windows = create_sliding_windows(
    data_array,
    window_size=WINDOW_SIZE,
    stride=STRIDE,
    horizon=PREDICTION_HORIZON
)

print(f"Window shape: {X_windows.shape}")  # (n_windows, timesteps, n_features)
print(f"Number of windows: {X_windows.shape[0]:,}")
print(f"Timesteps per window: {X_windows.shape[1]}")
print(f"Features per timestep: {X_windows.shape[2]}")

if y_windows is not None:
    print(f"Target shape: {y_windows.shape}")

# ==============================================
# 5. Window data quality check
# ==============================================
print("\n" + "=" * 60)
print("Window data quality check")
print("=" * 60)

# Check for NaN values
nan_count = np.isnan(X_windows).sum()
print(f"Number of NaN values: {nan_count}")

if nan_count > 0:
    print("Found NaN values, filling...")
    # Use forward fill then backward fill
    for i in range(X_windows.shape[0]):
        for j in range(X_windows.shape[2]):
            col_data = X_windows[i, :, j]
            if np.isnan(col_data).any():
                df_series = pd.Series(col_data)
                df_series = df_series.fillna(method='ffill').fillna(method='bfill')
                X_windows[i, :, j] = df_series.values

# Check for infinite values
inf_count = np.isinf(X_windows).sum()
print(f"Number of infinite values: {inf_count}")

# Window statistics
print(f"\nWindow data statistics:")
print(f"  Mean: {X_windows.mean():.6f}")
print(f"  Std: {X_windows.std():.6f}")
print(f"  Min: {X_windows.min():.6f}")
print(f"  Max: {X_windows.max():.6f}")

# ==============================================
# 6. Save processed data
# ==============================================
print("\n" + "=" * 60)
print("Saving processed data")
print("=" * 60)

# Save sliding windows
window_path = os.path.join(OUTPUT_DIR, "x_windows_100k.npy")
np.save(window_path, X_windows)
print(f"✓ Sliding windows saved to: {window_path}")
print(f"  File size: {os.path.getsize(window_path) / (1024 * 1024):.2f} MB")

# Save target values (if any)
if y_windows is not None:
    y_path = os.path.join(OUTPUT_DIR, "y_windows_100k.npy")
    np.save(y_path, y_windows)
    print(f"✓ Target values saved to: {y_path}")

# Save processed DataFrame (including timestamps)
df_processed = df.copy()
processed_csv_path = os.path.join(OUTPUT_DIR, "processed_time_series_augmented_100k.csv")
df_processed.reset_index().to_csv(processed_csv_path, index=False)
print(f"✓ Processed time series saved to: {processed_csv_path}")

# ==============================================
# 7. Generate data report
# ==============================================
print("\n" + "=" * 60)
print("Generating data report")
print("=" * 60)

# Create report
report = {
    'processing_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'input_file': DATA_PATH,
    'output_files': {
        'x_windows': window_path,
        'processed_time_series': processed_csv_path
    },
    'data_statistics': {
        'original_samples': len(df),
        'window_count': X_windows.shape[0],
        'window_size': X_windows.shape[1],
        'features_per_timestep': X_windows.shape[2],
        'time_range': [str(df.index.min()), str(df.index.max())]
    },
    'window_settings': {
        'window_size': WINDOW_SIZE,
        'stride': STRIDE,
        'prediction_horizon': PREDICTION_HORIZON
    },
    'features_used': available_features,
    'data_quality': {
        'nan_values_before_filling': int(nan_count),
        'inf_values': int(inf_count),
        'window_data_mean': float(X_windows.mean()),
        'window_data_std': float(X_windows.std())
    }
}

# Add feature statistics
feature_stats = {}
for feature in available_features:
    if feature in df.columns:
        feature_stats[feature] = {
            'mean': float(df[feature].mean()),
            'std': float(df[feature].std()),
            'min': float(df[feature].min()),
            'max': float(df[feature].max())
        }
report['feature_statistics'] = feature_stats

# Save report
report_path = os.path.join(OUTPUT_DIR, "100k_window_processing_report.json")
import json

with open(report_path, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(f"✓ Processing report saved to: {report_path}")

# ==============================================
# 8. Visualization (optional)
# ==============================================
print("\n" + "=" * 60)
print("Generating visualization")
print("=" * 60)

try:
    # Create visualization
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))

    # 1. Time series plot (first 500 points)
    for i, feature in enumerate(available_features[:4]):
        ax = axes[i // 2, i % 2]
        plot_data = df[feature].iloc[:500]
        ax.plot(plot_data.index, plot_data.values, 'b-', alpha=0.7, linewidth=1)
        ax.set_title(f'{feature} (first 500 points)')
        ax.set_xlabel('Time')
        ax.set_ylabel('Normalized value')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)

    # 2. First window visualization
    if X_windows.shape[0] > 0:
        ax = axes[2, 0]
        for i in range(min(4, X_windows.shape[2])):
            ax.plot(X_windows[0, :, i], label=f'Feature {i + 1}')
        ax.set_title(f'First sliding window (size={WINDOW_SIZE})')
        ax.set_xlabel('Time step')
        ax.set_ylabel('Normalized value')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 3. Window count statistics
    ax = axes[2, 1]
    n_windows = X_windows.shape[0]
    ax.bar(['Total windows'], [n_windows], color='skyblue')
    ax.set_title(f'Total number of sliding windows: {n_windows:,}')
    ax.set_ylabel('Count')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    viz_path = os.path.join(OUTPUT_DIR, "100k_window_visualization.png")
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Visualization saved to: {viz_path}")

except Exception as e:
    print(f"⚠ Visualization generation failed: {e}")

# ==============================================
# 9. Verify saved data
# ==============================================
print("\n" + "=" * 60)
print("Verifying saved data")
print("=" * 60)

# Reload saved window data
try:
    X_loaded = np.load(window_path)
    print(f"✓ Successfully loaded x_windows_100k.npy")
    print(f"  Shape: {X_loaded.shape}")
    print(f"  Data type: {X_loaded.dtype}")

    # Verify data integrity
    loaded_nan = np.isnan(X_loaded).sum()
    loaded_inf = np.isinf(X_loaded).sum()
    print(f"  NaN values: {loaded_nan}")
    print(f"  Infinite values: {loaded_inf}")

    if loaded_nan == 0 and loaded_inf == 0:
        print("  ✅ Data complete, no anomalies")
    else:
        print("  ⚠ Anomalies found, may need reprocessing")

except Exception as e:
    print(f"✗ Load verification failed: {e}")

# ==============================================
# 10. Final summary
# ==============================================
print("\n" + "=" * 60)
print("🎉 Data processing complete!")
print("=" * 60)

print(f"\n📊 Processing results:")
print(f"  Input data: {len(df)} samples")
print(f"  Windows created: {X_windows.shape[0]:,}")
print(f"  Each window: {X_windows.shape[1]} time steps, {X_windows.shape[2]} features")
print(f"  Time range: {df.index.min()} to {df.index.max()}")

print(f"\n📁 Generated files:")
print(f"  1. x_windows_100k.npy - Sliding window data ({X_windows.shape[0]:,} windows)")
print(f"  2. processed_time_series_augmented.csv - Processed time series")
print(f"  3. window_processing_report.json - Processing report")
print(f"  4. window_visualization.png - Visualization")

print(f"\n🔧 Next step:")
print(f"  Now you can use x_windows_100k.npy for model training!")
print(f"  Example loading code:")
print(f"  X_windows = np.load(r\"{window_path}\")")
print(f"  print(f\"Window shape: {{X_windows.shape}}\")")
