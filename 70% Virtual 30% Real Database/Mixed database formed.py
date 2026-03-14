# ====================== Complete Dataset Merging and Saving Process (Corrected Version) ======================
import json
import os
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

warnings.filterwarnings('ignore')


def load_real_data_hourly():
    """Load real dataset (hourly format)"""
    print("🔍 Loading real dataset (hourly format)...")

    # Load the CSV file you provided
    real_data_path = r"D:\Oswaldo's surf project\DR O's database\preprocessed_data\normalized_hourly_data.csv"

    if not os.path.exists(real_data_path):
        print(f"❌ Real data file does not exist: {real_data_path}")
        return None

    # Read CSV file
    df_real = pd.read_csv(real_data_path, index_col=0, parse_dates=True)
    print(f"  Real data shape: {df_real.shape}")
    print(f"  Real data column names: {df_real.columns.tolist()}")

    # Extract first 4 columns as features
    feature_columns = df_real.columns[:4].tolist()
    print(f"  Feature columns used (first 4 columns): {feature_columns}")

    # Extract feature data
    X_real_hourly = df_real[feature_columns].values
    print(f"  Feature data shape: {X_real_hourly.shape}")

    return X_real_hourly, df_real, feature_columns


def load_virtual_data_24hour():
    """Load virtual dataset in 24-hour window format"""
    print("🔍 Loading virtual dataset in 24-hour window format...")

    # Virtual data path
    virtual_samples_path = r"D:\Oswaldo's surf project\My Database\Merge_and_DAE_4_extracted_features\virtual_samples_4features_35040.npy"

    if not os.path.exists(virtual_samples_path):
        print(f"❌ Virtual data file does not exist: {virtual_samples_path}")
        return None

    # Load virtual data
    virtual_samples = np.load(virtual_samples_path)
    print(f"  Virtual data original shape: {virtual_samples.shape}")

    # Check dimensions, ensure it is in 24-hour window format
    if len(virtual_samples.shape) == 3:
        # Already 3D data
        if virtual_samples.shape[1] == 24 and virtual_samples.shape[2] == 4:
            print(f"  Virtual data is already in 24-hour window format: {virtual_samples.shape}")
            return virtual_samples
        else:
            print(f"⚠️  Virtual data dimensions do not match expected 24-hour window format: {virtual_samples.shape}")
            print(f"  Attempting to reshape...")
    elif len(virtual_samples.shape) == 2:
        # If 2D data, try to reshape to 24-hour windows
        print(f"  Virtual data is 2D, attempting to reshape to 24-hour windows...")
        total_samples = virtual_samples.shape[0]

        # Check if it can be reshaped to (n_samples, 24, 4)
        if total_samples % 24 == 0:
            n_samples = total_samples // 24
            virtual_samples = virtual_samples.reshape(n_samples, 24, 4)
            print(f"  Successfully reshaped to: {virtual_samples.shape}")
        else:
            print(f"❌ Cannot reshape virtual data to 24-hour windows: total samples {total_samples} not divisible by 24")
            return None
    else:
        print(f"❌ Virtual data dimension error: {virtual_samples.shape}")
        return None

    return virtual_samples


def create_hourly_data_from_windows(window_data):
    """Convert window data to hourly data"""
    print(f"  Converting window data to hourly data...")

    if len(window_data.shape) == 3:
        # Window data shape: (n_windows, 24, 4)
        n_windows = window_data.shape[0]
        n_hours = window_data.shape[1]
        n_features = window_data.shape[2]

        # Reshape to hourly data
        hourly_data = window_data.reshape(-1, n_features)
        print(f"  Hourly data shape after conversion: {hourly_data.shape}")
        return hourly_data
    else:
        print(f"❌ Window data format error: {window_data.shape}")
        return None


def create_mixed_dataset_hourly(X_real_hourly, X_virtual_hourly, target_hours=35064, real_ratio=0.3):
    """
    Create mixed dataset (hourly format) - Key modification: uniform mixing
    """
    print("\n🎯 Creating mixed dataset (3:7 ratio, hourly format)...")

    # Check input data
    if X_real_hourly is None or X_virtual_hourly is None:
        print("❌ Input data is empty")
        return None, None

    print(f"  Real hourly data shape: {X_real_hourly.shape}")
    print(f"  Virtual hourly data shape: {X_virtual_hourly.shape}")

    # Calculate required number of hours
    n_real_hours_needed = int(target_hours * real_ratio)
    n_virtual_hours_needed = target_hours - n_real_hours_needed

    print(f"  Total target hours: {target_hours:,}")
    print(f"  Real hours needed: {n_real_hours_needed:,} ({real_ratio * 100:.0f}%)")
    print(f"  Virtual hours needed: {n_virtual_hours_needed:,} ({(1 - real_ratio) * 100:.0f}%)")

    # ==================== Key modification: uniform mixing ====================
    print("\n🔀 Creating uniformly mixed dataset...")

    # Ensure we have enough real data
    if X_real_hourly.shape[0] < n_real_hours_needed:
        print(f"⚠️  Insufficient real data, using all real data")
        n_real_actual = X_real_hourly.shape[0]
        X_real_selected = X_real_hourly[:n_real_actual]  # Keep chronological order
    else:
        n_real_actual = n_real_hours_needed
        # Select uniformly from real data (keep chronological order)
        step = max(1, X_real_hourly.shape[0] // n_real_actual)
        indices = np.arange(0, X_real_hourly.shape[0], step)[:n_real_actual]
        X_real_selected = X_real_hourly[indices]

    # Ensure we have enough virtual data
    if X_virtual_hourly.shape[0] < n_virtual_hours_needed:
        print(f"⚠️  Insufficient virtual data, using all virtual data")
        n_virtual_actual = X_virtual_hourly.shape[0]
        X_virtual_selected = X_virtual_hourly[:n_virtual_actual]
    else:
        n_virtual_actual = n_virtual_hours_needed
        # Select uniformly from virtual data
        step = max(1, X_virtual_hourly.shape[0] // n_virtual_actual)
        indices = np.arange(0, X_virtual_hourly.shape[0], step)[:n_virtual_actual]
        X_virtual_selected = X_virtual_hourly[indices]

    # ==================== Uniform mixing: alternate insertion ====================
    print("\n🔀 Alternately inserting real and virtual data...")

    X_mixed = np.zeros((n_real_actual + n_virtual_actual, X_real_hourly.shape[1]))
    y_mixed = np.zeros(n_real_actual + n_virtual_actual)

    real_idx = 0
    virtual_idx = 0
    mixed_idx = 0

    # Alternate insertion of real and virtual data
    while real_idx < n_real_actual and virtual_idx < n_virtual_actual:
        # Insert real data
        X_mixed[mixed_idx] = X_real_selected[real_idx]
        y_mixed[mixed_idx] = 1
        real_idx += 1
        mixed_idx += 1

        # Insert virtual data
        X_mixed[mixed_idx] = X_virtual_selected[virtual_idx]
        y_mixed[mixed_idx] = 0
        virtual_idx += 1
        mixed_idx += 1

    # If any real data remaining
    while real_idx < n_real_actual:
        X_mixed[mixed_idx] = X_real_selected[real_idx]
        y_mixed[mixed_idx] = 1
        real_idx += 1
        mixed_idx += 1

    # If any virtual data remaining
    while virtual_idx < n_virtual_actual:
        X_mixed[mixed_idx] = X_virtual_selected[virtual_idx]
        y_mixed[mixed_idx] = 0
        virtual_idx += 1
        mixed_idx += 1

    print(f"✅ Mixed dataset creation completed:")
    print(f"  Actual total hours: {len(X_mixed):,}")
    print(f"  Actual real hours: {np.sum(y_mixed):,} ({np.sum(y_mixed) / len(y_mixed) * 100:.1f}%)")
    print(f"  Actual virtual hours: {len(y_mixed) - np.sum(y_mixed):,} ({100 - np.sum(y_mixed) / len(y_mixed) * 100:.1f}%)")
    print(f"  Mixed data shape: {X_mixed.shape}")

    return X_mixed, y_mixed


def save_mixed_dataset_hourly(X_mixed, y_mixed, feature_names):
    """Save mixed dataset (hourly format)"""
    print("\n💾 Saving mixed dataset (hourly format)...")

    # Create save directory
    save_dir = r"D:\Oswaldo's surf project\70% Virtual 30% Real Database_Hourly"
    os.makedirs(save_dir, exist_ok=True)

    # Save NumPy format (hourly)
    X_save_path = os.path.join(save_dir, "X_mixed_hourly_35040.npy")
    y_save_path = os.path.join(save_dir, "y_mixed_labels_hourly_35040.npy")

    np.save(X_save_path, X_mixed)
    np.save(y_save_path, y_mixed)

    print(f"  NumPy files saved:")
    print(f"    X_mixed_hourly_35040.npy: {X_mixed.shape}")
    print(f"    y_mixed_labels_hourly_35040.npy: {y_mixed.shape}")

    # ==================== Generate CSV file (35040 rows) ====================
    print("\n📊 Creating and saving CSV file (35064 rows, hourly format)...")

    n_samples, n_features = X_mixed.shape

    print(f"  Data shape: {X_mixed.shape}")
    print(f"  Each row represents one hour, total 35040 rows")

    # Simplify feature column names
    clean_feature_names = []
    for name in feature_names:
        clean_name = name.replace('_normalized_scaled', '').replace('_scaled', '')
        clean_feature_names.append(clean_name)

    print(f"  Simplified feature column names: {clean_feature_names}")

    # Generate timestamps (starting from 2013/3/1 1:00, hourly)
    print("\n📅 Generating timestamps...")
    start_date = datetime(2013, 3, 1, 1, 0, 0)
    timestamps = [start_date + timedelta(hours=i) for i in range(n_samples)]

    print(f"  First timestamp: {timestamps[0]}")
    print(f"  Last timestamp: {timestamps[-1]}")
    print(f"  Total hours: {len(timestamps)}")

    # Create DataFrame
    df_hourly = pd.DataFrame({
        'timestamp': timestamps
    })

    # Add feature columns
    for i in range(n_features):
        df_hourly[clean_feature_names[i]] = X_mixed[:, i]

    # Save CSV file
    csv_path = os.path.join(save_dir, "mixed_dataset_hourly_35040.csv")
    df_hourly.to_csv(csv_path, index=False, encoding='utf-8-sig')

    print(f"✅ CSV file saved: {csv_path}")
    print(f"  File size: {os.path.getsize(csv_path) / (1024 * 1024):.2f} MB")
    print(f"  Rows: {len(df_hourly):,}")
    print(f"  Columns: {len(df_hourly.columns)}")
    print(f"  Column names: {list(df_hourly.columns)}")

    # Display first few rows
    print(f"\n📋 Preview of first 10 rows of CSV file:")
    print(df_hourly.head(10).to_string(index=False))

    # Show mixing pattern (check alternating pattern)
    print(f"\n🔍 Mixing pattern check (labels of first 20 time points):")
    print(f"  Label sequence (1=real, 0=virtual): {y_mixed[:20]}")
    print(f"  Real data proportion: {np.sum(y_mixed) / len(y_mixed) * 100:.1f}%")

    # Save metadata
    metadata = {
        "dataset_name": "70% Virtual 30% Real Database_Hourly",
        "total_hours": int(n_samples),
        "real_hours": int(np.sum(y_mixed)),
        "virtual_hours": int(len(y_mixed) - np.sum(y_mixed)),
        "real_ratio": float(np.sum(y_mixed) / len(y_mixed)),
        "virtual_ratio": float(1 - np.sum(y_mixed) / len(y_mixed)),
        "data_shape": list(X_mixed.shape),
        "feature_names": clean_feature_names,
        "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "description": "Mixed dataset: 30% real data + 70% virtual data, alternating uniform mixing",
        "timestamp_start": timestamps[0].strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp_end": timestamps[-1].strftime("%Y-%m-%d %H:%M:%S"),
        "mixing_method": "Alternating mixing (real-virtual-real-virtual...)",
        "note": "CSV file contains 35064 rows, each row represents one hour, features mixed alternately"
    }

    metadata_path = os.path.join(save_dir, "mixed_dataset_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"  Metadata file saved: {metadata_path}")

    return save_dir, csv_path


def main():
    print("=" * 80)
    print("Real and Virtual Data Mixing Tool (Solving High Anomaly Rate Issue)")
    print("Goal: Create 35064-row mixed dataset, uniformly mixing real and virtual data")
    print("=" * 80)

    # 1. Load real data (hourly format)
    print("\n1. Loading real dataset...")
    result = load_real_data_hourly()
    if result is None:
        return

    X_real_hourly, df_real, real_feature_names = result
    print(f"    Real data hours available: {X_real_hourly.shape[0]:,}")

    # 2. Load virtual data (24-hour window format)
    print("\n2. Loading virtual dataset...")
    X_virtual_24h = load_virtual_data_24hour()
    if X_virtual_24h is None:
        print("❌ Failed to load virtual data, exiting")
        return

    # 3. Convert virtual data to hourly format
    print("\n3. Converting virtual data format...")
    X_virtual_hourly = create_hourly_data_from_windows(X_virtual_24h)
    if X_virtual_hourly is None:
        print("❌ Failed to convert virtual data format")
        return

    print(f"    Virtual data hours after conversion: {X_virtual_hourly.shape[0]:,}")

    # 4. Create mixed dataset (uniform mixing)
    print("\n4. Creating mixed dataset (uniform mixing)...")
    X_mixed, y_mixed = create_mixed_dataset_hourly(
        X_real_hourly,
        X_virtual_hourly,
        target_hours=35064,
        real_ratio=0.3
    )

    if X_mixed is None:
        print("❌ Failed to create mixed dataset")
        return

    # 5. Save dataset
    save_dir, csv_path = save_mixed_dataset_hourly(X_mixed, y_mixed, real_feature_names)

    # 6. Dataset statistics
    print("\n📊 Dataset statistics:")
    print(f"  Total hours: {len(X_mixed):,}")
    print(f"  Real hours: {np.sum(y_mixed):,} ({np.sum(y_mixed) / len(y_mixed) * 100:.1f}%)")
    print(f"  Virtual hours: {len(y_mixed) - np.sum(y_mixed):,} ({100 - np.sum(y_mixed) / len(y_mixed) * 100:.1f}%)")
    print(f"  Data shape: {X_mixed.shape}")
    print(f"  Mixing method: Alternating mixing (real-virtual-real-virtual...)")

    print(f"\n✅ Dataset creation completed!")
    print(f"🎯 Goal: 35064-row CSV file generated")
    print(f"📍 All files saved to: {save_dir}")
    print(f"\n📝 Next steps:")
    print(f"  1. Run the second code (data_preparation.py)")
    print(f"  2. It will convert 35064 hours of data into window data")
    print(f"  3. Modify the path in the third code to point to the new preprocessed data")


if __name__ == "__main__":
    main()
