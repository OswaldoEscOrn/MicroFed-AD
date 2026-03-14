import pandas as pd
import numpy as np
import os


def create_70percent_anomaly_dataset_time_series(input_path, output_path, target_ratio=0.70):
    """
    Create a dataset with 70% anomaly ratio, preserving time series structure
    Adjust the ratio by resampling anomaly and normal samples while maintaining time order
    """
    print("=" * 60)
    print("Creating dataset with 70% anomaly ratio (preserving time series)")
    print("=" * 60)

    # 1. Load original combined data
    print("Loading original dataset...")
    df = pd.read_csv(input_path)
    print(f"Original dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # 2. Determine the time column (first column)
    time_column = df.columns[0]
    print(f"Time column: '{time_column}'")

    # Ensure data is sorted by time
    df = df.sort_values(by=time_column).reset_index(drop=True)

    # 3. Calculate anomaly labels (not added to output)
    print("\nCalculating anomaly labels...")

    # Air anomaly: PM2.5 > 75
    df['_air_anomaly'] = df['avg_PM2.5'] > 75

    # Noise anomaly: based on aggregated data
    # Calculate statistics
    duration_mean = df['total_noise_duration'].mean()
    duration_std = df['total_noise_duration'].std()
    event_mean = df['noise_event_count'].mean()
    event_std = df['noise_event_count'].std()

    # Define noise anomaly rules
    high_event_threshold = event_mean + event_std
    high_salience_threshold = 1.5
    extreme_duration_threshold = duration_mean + 2 * duration_std
    medium_event_threshold = event_mean + 0.5 * event_std
    medium_duration_threshold = duration_mean + 0.5 * duration_std

    # Noise anomaly rules
    rule1 = (df['noise_event_count'] > high_event_threshold) & (df['avg_salience'] > high_salience_threshold)
    rule2 = df['total_noise_duration'] > extreme_duration_threshold
    rule3 = (df['noise_event_count'] > medium_event_threshold) & (
            df['total_noise_duration'] > medium_duration_threshold)
    rule4 = (df['avg_salience'] > high_salience_threshold) & (df['total_noise_duration'] > medium_duration_threshold)

    df['_noise_anomaly'] = rule1 | rule2 | rule3 | rule4

    # Overall anomaly: air anomaly OR noise anomaly
    df['_overall_anomaly'] = df['_air_anomaly'] | df['_noise_anomaly']

    # Original anomaly statistics
    anomaly_count = df['_overall_anomaly'].sum()
    normal_count = len(df) - anomaly_count
    original_ratio = anomaly_count / len(df)

    print(f"Original anomaly statistics:")
    print(f"  Total samples: {len(df):,}")
    print(f"  Anomaly samples: {anomaly_count:,}")
    print(f"  Normal samples: {normal_count:,}")
    print(f"  Anomaly ratio: {original_ratio:.2%}")
    print(f"  Target ratio: {target_ratio:.0%}")

    # 4. Calculate required sample counts
    total_samples = len(df)
    target_anomaly_count = int(total_samples * target_ratio)
    target_normal_count = total_samples - target_anomaly_count

    print(f"\nTarget sample counts:")
    print(f"  Total: {total_samples:,}")
    print(f"  Anomaly: {target_anomaly_count:,}")
    print(f"  Normal: {target_normal_count:,}")

    # 5. Resample to achieve target ratio - maintain time order
    print("\nResampling to achieve target ratio (preserving time order)...")

    # Separate anomaly and normal samples - maintain time order
    anomaly_df = df[df['_overall_anomaly'] == 1].copy()
    normal_df = df[df['_overall_anomaly'] == 0].copy()

    # Sort by time
    anomaly_df = anomaly_df.sort_values(by=time_column)
    normal_df = normal_df.sort_values(by=time_column)

    print(f"  Original anomaly samples: {len(anomaly_df):,}")
    print(f"  Original normal samples: {len(normal_df):,}")

    # Resample anomaly samples - maintain time order
    if target_anomaly_count > len(anomaly_df):
        # Need more anomaly samples - cycle through in time order
        print("  Oversampling anomaly samples (time-ordered)...")
        repetitions = target_anomaly_count // len(anomaly_df) + 1
        anomaly_resampled = pd.concat([anomaly_df] * repetitions, ignore_index=True)
        anomaly_resampled = anomaly_resampled.head(target_anomaly_count).sort_values(by=time_column)
    else:
        # Need fewer anomaly samples - sample evenly across time
        print("  Undersampling anomaly samples (time-ordered)...")
        step = len(anomaly_df) / target_anomaly_count
        indices = [int(i * step) for i in range(target_anomaly_count)]
        anomaly_resampled = anomaly_df.iloc[indices].copy()

    # Resample normal samples - maintain time order
    if target_normal_count > len(normal_df):
        # Need more normal samples - cycle through in time order
        print("  Oversampling normal samples (time-ordered)...")
        repetitions = target_normal_count // len(normal_df) + 1
        normal_resampled = pd.concat([normal_df] * repetitions, ignore_index=True)
        normal_resampled = normal_resampled.head(target_normal_count).sort_values(by=time_column)
    else:
        # Need fewer normal samples - sample evenly across time
        print("  Undersampling normal samples (time-ordered)...")
        step = len(normal_df) / target_normal_count
        indices = [int(i * step) for i in range(target_normal_count)]
        normal_resampled = normal_df.iloc[indices].copy()

    # 6. Combine resampled data - sort by time
    print("\nCombining resampled data and sorting by time...")
    resampled_df = pd.concat([anomaly_resampled, normal_resampled], ignore_index=True)
    resampled_df = resampled_df.sort_values(by=time_column).reset_index(drop=True)

    # 7. Remove internally used anomaly columns to keep original format
    print("\nRemoving internal anomaly columns...")
    columns_to_drop = ['_air_anomaly', '_noise_anomaly', '_overall_anomaly']
    for col in columns_to_drop:
        if col in resampled_df.columns:
            resampled_df = resampled_df.drop(columns=[col])

    # 8. Verify output format
    print("\nVerifying output format...")
    print(f"  Output shape: {resampled_df.shape}")
    print(f"  Output columns: {list(resampled_df.columns)}")

    # Check column order matches original
    original_columns = list(df.columns)
    output_columns = list(resampled_df.columns)

    # Original columns after removing internal ones
    original_columns_clean = [col for col in original_columns if not col.startswith('_')]

    if output_columns == original_columns_clean:
        print("  ✓ Column order matches original")
    else:
        print("  ⚠ Column order may differ from original")
        print(f"  Original columns (cleaned): {original_columns_clean}")
        print(f"  Output columns: {output_columns}")

    # Verify sample count
    if len(resampled_df) == total_samples:
        print(f"  ✓ Total sample count matches original: {len(resampled_df):,}")
    else:
        print(f"  ⚠ Total sample count differs: {len(resampled_df):,} (expected {total_samples:,})")

    # 9. Save result
    print(f"\nSaving to: {output_path}")
    resampled_df.to_csv(output_path, index=False)

    # 10. Verify time order
    print("\nVerifying time order...")
    times = pd.to_datetime(resampled_df[time_column])
    is_sorted = all(times[i] <= times[i + 1] for i in range(len(times) - 1))

    if is_sorted:
        print("  ✓ Data is correctly sorted by time")
    else:
        print("  ⚠ Data is NOT sorted by time - check for issues")

    # 11. Compute final statistics (need to recalculate anomaly labels)
    print("\nFinal dataset statistics:")
    print("Recalculating anomaly labels for verification...")

    # Recalculate anomaly labels to verify ratio
    final_air_anomaly = resampled_df['avg_PM2.5'] > 75

    # Recalculate noise anomaly (using resampled statistics)
    final_duration_mean = resampled_df['total_noise_duration'].mean()
    final_duration_std = resampled_df['total_noise_duration'].std()
    final_event_mean = resampled_df['noise_event_count'].mean()
    final_event_std = resampled_df['noise_event_count'].std()

    final_high_event = final_event_mean + final_event_std
    final_extreme_duration = final_duration_mean + 2 * final_duration_std
    final_medium_event = final_event_mean + 0.5 * final_event_std
    final_medium_duration = final_duration_mean + 0.5 * final_duration_std

    final_rule1 = (resampled_df['noise_event_count'] > final_high_event) & (
            resampled_df['avg_salience'] > high_salience_threshold)
    final_rule2 = resampled_df['total_noise_duration'] > final_extreme_duration
    final_rule3 = (resampled_df['noise_event_count'] > final_medium_event) & (
            resampled_df['total_noise_duration'] > final_medium_duration)
    final_rule4 = (resampled_df['avg_salience'] > high_salience_threshold) & (
            resampled_df['total_noise_duration'] > final_medium_duration)

    final_noise_anomaly = final_rule1 | final_rule2 | final_rule3 | final_rule4
    final_overall_anomaly = final_air_anomaly | final_noise_anomaly

    final_anomaly_count = final_overall_anomaly.sum()
    final_ratio = final_anomaly_count / len(resampled_df)

    print(f"  Total samples: {len(resampled_df):,}")
    print(f"  Anomaly samples: {final_anomaly_count:,}")
    print(f"  Normal samples: {len(resampled_df) - final_anomaly_count:,}")
    print(f"  Final anomaly ratio: {final_ratio:.2%}")
    print(f"  Target anomaly ratio: {target_ratio:.0%}")

    if abs(final_ratio - target_ratio) < 0.01:  # 1% tolerance
        print(f"  ✓ Successfully achieved target ratio (within 1% tolerance)")
    else:
        print(f"  ⚠ Final ratio ({final_ratio:.2%}) differs from target ({target_ratio:.0%})")

    print("\n" + "=" * 60)
    print("Dataset creation completed! Time series structure preserved.")
    print("=" * 60)

    return resampled_df


# Run the function
input_file = r"D:\Oswaldo's surf project\DR O's database\combined_environmental_data.csv"
output_file = r"D:\Oswaldo's surf project\DR O's database\data_anomaly_ratio_handling\reconstruction_data_70%_anomaly.csv"

# Create output directory if it does not exist
output_dir = os.path.dirname(output_file)
os.makedirs(output_dir, exist_ok=True)

# Generate the dataset
result_df = create_70percent_anomaly_dataset_time_series(
    input_path=input_file,
    output_path=output_file,
    target_ratio=0.70
)

# Display first few rows to verify format
print("\nFirst 5 rows of generated dataset:")
print(result_df.head())

print("\nDataset info:")
print(f"Saved to: {output_file}")
print(f"Shape: {result_df.shape}")
print(f"Columns: {list(result_df.columns)}")
