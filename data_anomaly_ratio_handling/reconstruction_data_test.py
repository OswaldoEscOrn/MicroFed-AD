import pandas as pd
import os


def fix_duplicated_reconstruction_data():
    """Fix duplicated reconstruction_data.csv"""

    input_path = r"D:\Oswaldo's surf project\DR O's database\data_anomaly_ratio_handling\reconstruction_data.csv"
    output_path = r"D:\Oswaldo's surf project\DR O's database\data_anomaly_ratio_handling\reconstruction_data_fixed.csv"

    print("🔍 Detecting and fixing duplicate data...")

    # 1. Load data
    df = pd.read_csv(input_path)

    # Rename time column
    time_column = df.columns[0]
    df.rename(columns={time_column: 'timestamp'}, inplace=True)

    # Convert time format
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y/%m/%d %H:%M', errors='coerce')

    print(f"Original data shape: {df.shape}")
    print(f"Time column: 'timestamp'")

    # 2. Detect duplicate rows (based on all columns)
    duplicate_rows = df.duplicated().sum()
    print(f"Number of fully duplicate rows: {duplicate_rows} ({duplicate_rows / len(df) * 100:.1f}%)")

    # 3. Detect timestamp duplicates
    time_duplicates = df['timestamp'].duplicated().sum()
    print(f"Number of duplicate timestamps: {time_duplicates}")

    # 4. Group by timestamp for statistics
    print("\n📊 Group statistics by timestamp:")
    time_groups = df.groupby('timestamp').size().reset_index(name='count')

    # Distribution of duplicate counts
    duplicate_counts = time_groups['count'].value_counts().sort_index()
    for count, freq in duplicate_counts.items():
        print(f"  Timestamps with {count} occurrences: {freq}")

    # 5. Fix: remove duplicate rows, keep the first
    print("\n🧹 Fixing data...")

    # Method 1: deduplicate by timestamp, keep the first
    df_fixed = df.drop_duplicates(subset=['timestamp'], keep='first')

    print(f"Data shape after fix: {df_fixed.shape}")
    print(f"Removed {len(df) - len(df_fixed)} duplicate rows")

    # 6. Verify fix results
    print("\n✅ Verifying fix results:")
    remaining_duplicates = df_fixed['timestamp'].duplicated().sum()
    print(f"Remaining duplicate timestamps: {remaining_duplicates}")

    # 7. Save fixed data
    print(f"\n💾 Saving fixed data to: {output_path}")
    df_fixed.to_csv(output_path, index=False)

    # 8. Validate saved file
    print("\n🔍 Validating saved file...")
    df_loaded = pd.read_csv(output_path)
    print(f"Loaded data shape: {df_loaded.shape}")

    # Display first few rows
    print("\nFirst 5 rows of data:")
    print(df_loaded.head())

    # Check time series continuity
    times = pd.to_datetime(df_loaded['timestamp'])
    time_diffs = times.diff().dropna()

    print(f"\n📈 Time interval statistics:")
    print(f"  Minimum interval: {time_diffs.min()}")
    print(f"  Maximum interval: {time_diffs.max()}")
    print(f"  Average interval: {time_diffs.mean()}")

    # Check if all intervals are 1 hour
    one_hour = pd.Timedelta(hours=1)
    is_hourly = (time_diffs == one_hour).all()

    if is_hourly:
        print(f"  ✅ Data is a continuous hourly time series")
    else:
        print(f"  ⚠ Data is not strictly hourly")
        irregular = (time_diffs != one_hour).sum()
        print(f"    Number of irregular intervals: {irregular}")

    return df_fixed


# Run the fix function
fixed_df = fix_duplicated_reconstruction_data()
