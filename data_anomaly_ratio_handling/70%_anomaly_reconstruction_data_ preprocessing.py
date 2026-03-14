import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import json
import joblib

# ==============================================
# Configuration Parameters
# ==============================================
DATA_PATH = r"D:\Oswaldo's surf project\DR O's database\data_anomaly_ratio_handling\reconstruction_data_70%_anomaly_fixed.csv"
OUTPUT_DIR = r"D:\Oswaldo's surf project\DR O's database\final_preprocessed_data_70%_anomaly_complete"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Window parameters
WINDOW_SIZE = 24
STRIDE = 1
PREDICTION_HORIZON = 1

# The 5 target columns to process
TARGET_COLUMNS = [
    'avg_PM2.5_normalized',
    'total_noise_duration',
    'noise_event_count',
    'avg_salience',
    'avg_PM2.5'
]

# ==============================================
# 1. Load data and ensure timestamp is the first column
# ==============================================
print("📥 Loading data...")
df = pd.read_csv(DATA_PATH)

# Check what the first column is
print(f"Original CSV columns: {list(df.columns)}")
print(f"First column name: '{df.columns[0]}'")

# Rename the first column
time_column = df.columns[0]
df.rename(columns={time_column: 'timestamp'}, inplace=True)

# Convert to datetime and set as index
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y/%m/%d %H:%M', errors='coerce')
df = df.set_index('timestamp')
df = df.sort_index()

print(f"✅ Data loading complete:")
print(f"   Time range: {df.index.min()} to {df.index.max()}")
print(f"   Total rows: {len(df)}")
print(f"   Columns: {list(df.columns)}")

# ==============================================
# 2. Check target columns
# ==============================================
print("\n🔍 Checking target columns...")
available_columns = []
for col in TARGET_COLUMNS:
    if col in df.columns:
        available_columns.append(col)
        print(f"   ✓ {col}: exists")
    else:
        print(f"   ✗ {col}: missing")

# Exit if no target columns are found
if not available_columns:
    print("❌ No target columns found!")
    exit()

# Create DataFrame containing target columns
processed_df = pd.DataFrame(index=df.index)
for col in available_columns:
    processed_df[col] = df[col]

print(f"Processed data shape: {processed_df.shape}")

# ==============================================
# 3. Standardization
# ==============================================
print("\n⚖️ Performing standardization...")

columns_to_scale = [col for col in available_columns if col != 'avg_PM2.5']
scalers = {}

for col in columns_to_scale:
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(processed_df[[col]])
    new_col_name = f"{col}_scaled"
    processed_df[new_col_name] = scaled_data
    scalers[col] = scaler
    print(f"   ✓ {col} → {new_col_name}")

# ==============================================
# 4. Create final DataFrame (ensure correct column order)
# ==============================================
print("\n🎯 Organizing final columns...")

# Define final columns
FINAL_COLUMNS = [
    'avg_PM2.5_normalized_scaled',
    'total_noise_duration_scaled',
    'noise_event_count_scaled',
    'avg_salience_scaled',
    'avg_PM2.5'
]

# Create final DataFrame (extract needed columns from processed_df)
final_df = pd.DataFrame(index=processed_df.index)

for col in FINAL_COLUMNS:
    if col in processed_df.columns:
        final_df[col] = processed_df[col]
    else:
        # Try to find the original column
        alt_col = col.replace('_scaled', '')
        if alt_col in processed_df.columns:
            final_df[col] = processed_df[alt_col]
            print(f"   ⚠ Using {alt_col} instead of {col}")
        else:
            print(f"   ⚠ Column {col} does not exist")

print(f"   Final data shape: {final_df.shape}")
print(f"   Columns: {list(final_df.columns)}")

# ==============================================
# 5. Create sliding windows
# ==============================================
print(f"\n🪟 Creating sliding windows...")

# Get timestamps and data
timestamps = final_df.index.values
data_array = final_df.values


def create_sliding_windows(data, timestamps, window_size, stride=1, horizon=0):
    """Create sliding windows, returning data, targets, and timestamps"""
    X, y, X_timestamps = [], [], []
    n_samples = len(data)

    for i in range(0, n_samples - window_size - horizon + 1, stride):
        # Data window
        window = data[i:i + window_size]
        X.append(window)

        # Start timestamp of the window
        X_timestamps.append(timestamps[i])

        if horizon > 0:
            target = data[i + window_size:i + window_size + horizon]
            y.append(target)

    X = np.array(X)
    if horizon > 0:
        y = np.array(y)
        return X, y, np.array(X_timestamps)
    return X, None, np.array(X_timestamps)


# Create sliding windows
X_windows, y_windows, window_timestamps = create_sliding_windows(
    data=data_array,
    timestamps=timestamps,
    window_size=WINDOW_SIZE,
    stride=STRIDE,
    horizon=PREDICTION_HORIZON
)

print(f"   Number of windows created: {X_windows.shape[0]:,}")
print(f"   Window shape: {X_windows.shape}")
print(f"   Number of timestamps: {len(window_timestamps)}")

# ==============================================
# 6. Save data (critical fix: ensure timestamp is first column)
# ==============================================
print("\n💾 Saving data...")

# 6.1 Save sliding window data (numpy format)
np.save(os.path.join(OUTPUT_DIR, "X_windows.npy"), X_windows)
np.save(os.path.join(OUTPUT_DIR, "window_timestamps.npy"), window_timestamps)

if y_windows is not None:
    np.save(os.path.join(OUTPUT_DIR, "y_windows.npy"), y_windows)

print(f"   ✓ Sliding windows saved")

# 6.2 Save original time series data (ensure timestamp is first column)
print("\n📊 Saving time series data...")

# Method 1: directly use to_csv, ensuring time column is the first column
csv_path = os.path.join(OUTPUT_DIR, "preprocessed_time_series.csv")

# Reset index, timestamp will become the first column
final_df_with_time = final_df.reset_index()

# Ensure column name is correct
final_df_with_time = final_df_with_time.rename(columns={'timestamp': time_column})

# Save CSV
final_df_with_time.to_csv(csv_path, index=False)

print(f"   ✓ Time series CSV saved: {csv_path}")

# Verify the saved CSV
print(f"\n🔍 Verifying saved CSV file:")
test_df = pd.read_csv(csv_path, nrows=3)
print(f"   Column names: {list(test_df.columns)}")
print(f"   First column: '{test_df.columns[0]}'")
print(f"   First 3 rows of first column:")
for i in range(min(3, len(test_df))):
    print(f"     Row {i + 1}: {test_df.iloc[i, 0]}")

# 6.3 Create window data CSV (flatten windows, include timestamps)
print("\n📊 Creating window data CSV...")

n_windows = X_windows.shape[0]
n_timesteps = X_windows.shape[1]
n_features = X_windows.shape[2]

# Create column names
feature_names = final_df.columns.tolist()
column_names = [time_column]  # Use original time column name

# Create column names for each timestep and feature
for t in range(n_timesteps):
    for f_idx, feature in enumerate(feature_names):
        column_names.append(f"{feature}_t{t + 1}")

# Flatten window data
flattened_data = X_windows.reshape(n_windows, n_timesteps * n_features)

# Create DataFrame (critical: place timestamp in the first column)
window_df = pd.DataFrame(flattened_data, columns=column_names[1:])
window_df.insert(0, time_column, pd.to_datetime(window_timestamps))

# Save window data CSV
window_csv_path = os.path.join(OUTPUT_DIR, "window_data.csv")
window_df.to_csv(window_csv_path, index=False)
print(f"   ✓ Window data CSV saved: {window_csv_path}")

# Verify window CSV
print(f"\n🔍 Verifying window CSV:")
window_test = pd.read_csv(window_csv_path, nrows=2)
print(f"   Number of columns: {len(window_test.columns)}")
print(f"   First column: '{window_test.columns[0]}'")
print(f"   First row, first column value: {window_test.iloc[0, 0]}")

# 6.4 Save scalers
scaler_dir = os.path.join(OUTPUT_DIR, "scalers")
os.makedirs(scaler_dir, exist_ok=True)

for col_name, scaler in scalers.items():
    scaler_path = os.path.join(scaler_dir, f"scaler_{col_name}.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"   ✓ Scaler saved: scaler_{col_name}.pkl")

# 6.5 Save metadata
metadata = {
    'processing_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'original_data_path': DATA_PATH,
    'time_series_shape': list(final_df.shape),
    'time_series_columns': final_df.columns.tolist(),
    'window_stats': {
        'total_windows': int(X_windows.shape[0]),
        'window_shape': list(X_windows.shape),
        'window_size': WINDOW_SIZE,
        'stride': STRIDE,
        'horizon': PREDICTION_HORIZON
    },
    'timestamp_info': {
        'time_column_name': time_column,
        'start': str(final_df.index.min()),
        'end': str(final_df.index.max()),
        'total_samples': len(final_df)
    }
}

metadata_path = os.path.join(OUTPUT_DIR, "processing_metadata.json")
with open(metadata_path, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"   ✓ Metadata saved: {metadata_path}")

# ==============================================
# 7. Final verification
# ==============================================
print("\n" + "=" * 60)
print("🔍 Final Verification")
print("=" * 60)

# Reload all files for verification
print("1. Reloading time series CSV...")
ts_df = pd.read_csv(csv_path)
print(f"   Shape: {ts_df.shape}")
print(f"   First column name: '{ts_df.columns[0]}'")
print(f"   First column type: {type(ts_df.iloc[0, 0])}")

print("\n2. Reloading window CSV...")
win_df = pd.read_csv(window_csv_path)
print(f"   Shape: {win_df.shape}")
print(f"   First column name: '{win_df.columns[0]}'")

print("\n3. Loading sliding window data...")
X_loaded = np.load(os.path.join(OUTPUT_DIR, "X_windows.npy"))
timestamps_loaded = np.load(os.path.join(OUTPUT_DIR, "window_timestamps.npy"))
print(f"   X_windows shape: {X_loaded.shape}")
print(f"   Number of timestamps: {len(timestamps_loaded)}")

# ==============================================
# 8. Summary
# ==============================================
print("\n" + "=" * 60)
print("🎉 Preprocessing Complete!")
print("=" * 60)
print(f"Output directory: {OUTPUT_DIR}")
print(f"\nGenerated files:")
print(f"  1. preprocessed_time_series.csv - Time series data ({ts_df.shape[0]} rows, {ts_df.shape[1]} columns)")
print(f"     First column: '{ts_df.columns[0]}'")
print(f"  2. window_data.csv - Window data ({win_df.shape[0]} windows, {win_df.shape[1]} columns)")
print(f"     First column: '{win_df.columns[0]}'")
print(f"  3. X_windows.npy - Sliding window array (shape: {X_windows.shape})")
print(f"  4. window_timestamps.npy - Window start timestamps")
print(f"  5. scalers/ - Scaler folder")
print(f"  6. processing_metadata.json - Metadata")

print(f"\nData statistics:")
print(f"  Original samples: {len(final_df)}")
print(f"  Windows created: {X_windows.shape[0]}")
print(f"  Each window: {X_windows.shape[1]} hours, {X_windows.shape[2]} features")
print(f"  Time range: {final_df.index.min()} to {final_df.index.max()}")

# Display first few rows
print(f"\nTime series data preview (first 3 rows):")
print(ts_df.head(3).to_string(index=False))
