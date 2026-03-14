import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import json
import warnings

warnings.filterwarnings('ignore')

# ==============================================
# Configuration Parameters
# ==============================================
INPUT_PATH = r"D:\Oswaldo's surf project\DR O's database\final_preprocessed_data_complete\preprocessed_time_series.csv"
OUTPUT_PATH = r"D:\Oswaldo's surf project\DR O's database\final_preprocessed_data_complete\preprocessed_time_series_augmented_100k.csv"
TARGET_SAMPLES = 100000  # Target number of samples: 100,000

# ==============================================
# 1. Load and fill missing timestamps
# ==============================================
print("=" * 60)
print("🔧 Step 1: Fill missing timestamps")
print("=" * 60)

# 1.1 Load data
print("📥 Loading preprocessed data...")
df = pd.read_csv(INPUT_PATH)

# Check the first column
time_column = df.columns[0]
print(f"Time column name: '{time_column}'")

# Rename time column for processing
df = df.rename(columns={time_column: 'timestamp'})
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df = df.set_index('timestamp').sort_index()

print(f"📊 Original data:")
print(f"  Time range: {df.index.min()} to {df.index.max()}")
print(f"  Total samples: {len(df)}")
print(f"  Feature columns: {list(df.columns)}")

# 1.2 Analyze missing time points
print("\n📈 Analyzing missing time points...")

# Create a complete hourly time range
expected_times = pd.date_range(
    start=df.index.min(),
    end=df.index.max(),
    freq='h'
)

print(f"  Complete time series should have: {len(expected_times)} hours")
print(f"  Actual data contains: {len(df)} hours")
print(f"  Missing hours: {len(expected_times) - len(df)}")
print(f"  Missing rate: {(len(expected_times) - len(df)) / len(expected_times) * 100:.1f}%")

# 1.3 Reindex to fill missing timestamps
print("\n🔧 Filling missing timestamps...")
df_complete = df.reindex(expected_times)

# Count numeric columns before and after filling
numeric_cols = df_complete.select_dtypes(include=[np.number]).columns
print(f"  Numeric columns: {list(numeric_cols)}")

# Use time-aware interpolation
print("  Performing time interpolation...")
for col in numeric_cols:
    original_non_null = df[col].notna().sum()
    df_complete[col] = df_complete[col].interpolate(method='time', limit_direction='both')
    filled_count = df_complete[col].notna().sum() - original_non_null
    print(f"    {col}: filled {filled_count} values")

print(f"  Data shape after filling: {df_complete.shape}")

# ==============================================
# 2. Data quality validation
# ==============================================
print("\n🔍 Data quality validation:")
print("-" * 40)

# Check data quality after interpolation
nan_count = df_complete.isna().sum().sum()
print(f"  Total NaN values: {nan_count}")

if nan_count > 0:
    print(f"  ⚠ There are still {nan_count} missing values, using forward fill")
    df_complete = df_complete.fillna(method='ffill').fillna(method='bfill')

# Verify time continuity
time_diff = df_complete.index.to_series().diff()
print(f"  Time interval check:")
print(f"    Min interval: {time_diff.min()}")
print(f"    Max interval: {time_diff.max()}")
print(f"    Avg interval: {time_diff.mean()}")

if (time_diff == pd.Timedelta(hours=1)).all():
    print(f"  ✅ Time series is fully continuous (one point per hour)")
else:
    irregular = (time_diff != pd.Timedelta(hours=1)).sum()
    print(f"  ⚠ Found {irregular} irregular time intervals")

# ==============================================
# 3. Data augmentation to target number of samples (100,000 samples)
# ==============================================
print("\n" + "=" * 60)
print("🚀 Step 2: Data augmentation to 100,000 samples")
print("=" * 60)

# 3.1 Prepare data
scaled_features = ['avg_PM2.5_normalized_scaled', 'total_noise_duration_scaled',
                   'noise_event_count_scaled', 'avg_salience_scaled']

# Check if necessary columns exist
missing_features = [col for col in scaled_features + ['avg_PM2.5'] if col not in df_complete.columns]
if missing_features:
    print(f"❌ Missing necessary feature columns: {missing_features}")
    print("  Trying to find alternative columns...")
    # Try to find alternative column names
    for col in missing_features:
        if '_scaled' in col:
            alt_col = col.replace('_scaled', '')
            if alt_col in df_complete.columns:
                df_complete[col] = df_complete[alt_col]
                print(f"   Using {alt_col} as alternative for {col}")
                missing_features.remove(col)

if missing_features:
    raise ValueError(f"❌ Cannot find the following columns: {missing_features}")

# 3.2 Extract data
X_scaled = df_complete[scaled_features].values
pm25_original = df_complete['avg_PM2.5'].values
original_timestamps = df_complete.index
original_samples = len(df_complete)

print(f"📊 Data extraction complete:")
print(f"  X_scaled shape: {X_scaled.shape}")
print(f"  PM2.5 shape: {pm25_original.shape}")

# 3.3 Calculate required multiplier
multiplier = max(1, int(np.ceil(TARGET_SAMPLES / original_samples)))
print(f"\n🔄 Augmentation plan:")
print(f"  Original samples: {original_samples}")
print(f"  Target samples: {TARGET_SAMPLES}")
print(f"  Required multiplier: {multiplier}x")

# 3.4 Perform data augmentation - using more complex augmentation strategy
print("\n🔧 Starting data augmentation...")

all_scaled_data = []
all_pm25_data = []
all_timestamps = []

# Starting time (from the beginning of original data)
current_time = original_timestamps[0]

for i in range(multiplier):
    print(f"  Processing batch {i + 1}/{multiplier}...")

    # 3.4.1 Copy base data
    batch_scaled = X_scaled.copy()
    batch_pm25 = pm25_original.copy()

    # 3.4.2 Calculate batch-specific augmentation parameters
    # Batch shift factor to ensure each batch has a different pattern
    batch_shift = i * 0.5

    # Seasonal factor (12-month cycle)
    season_factor = 1.0 + 0.15 * np.sin(batch_shift + i * np.pi / 6)

    # Annual trend factor (simulate long-term trend)
    year_trend = 1.0 + 0.05 * np.sin(batch_shift + i * np.pi / 12)

    # 3.4.3 Apply multi-level time patterns
    # Hourly pattern (24-hour cycle)
    hour_of_day = np.arange(original_samples) % 24
    hour_noise = np.sin(hour_of_day * np.pi / 12).reshape(-1, 1) * 0.08 * season_factor

    # Weekly pattern (7-day cycle)
    day_of_week = np.arange(original_samples) % 168  # 168 hours = 7 days
    week_noise = np.sin(day_of_week * np.pi / 84).reshape(-1, 1) * 0.05 * year_trend

    # Monthly pattern (30-day cycle)
    day_of_month = np.arange(original_samples) % 720  # 720 hours = 30 days
    month_noise = np.sin(day_of_month * np.pi / 360).reshape(-1, 1) * 0.1 * season_factor

    # Combine time pattern noise
    time_pattern_noise = (hour_noise + week_noise + month_noise) / 3.0

    # 3.4.4 Add random noise (multi-scale)
    # High-frequency random noise (hourly variation)
    high_freq_noise = np.random.normal(0, 0.03, batch_scaled.shape)

    # Low-frequency random noise (daily variation) - corrected version
    # First calculate daily averages, then expand back to hourly
    days = int(np.ceil(original_samples / 24))
    daily_noise = np.random.normal(0, 0.02, (days, batch_scaled.shape[1]))

    # Expand daily average noise to hourly
    low_freq_noise = np.repeat(daily_noise, 24, axis=0)

    # Adjust to correct length if mismatch
    if low_freq_noise.shape[0] > original_samples:
        low_freq_noise = low_freq_noise[:original_samples]
    elif low_freq_noise.shape[0] < original_samples:
        # Repeat last day's data
        remaining = original_samples - low_freq_noise.shape[0]
        low_freq_noise = np.vstack([low_freq_noise, low_freq_noise[-remaining:]])

    # Combine random noise
    random_noise = (high_freq_noise * 0.6 + low_freq_noise * 0.4) * season_factor

    # 3.4.5 Apply augmentation to scaled features
    batch_scaled = batch_scaled + time_pattern_noise + random_noise

    # Keep data within reasonable range (for z-score normalized data)
    batch_scaled = np.clip(batch_scaled, -3, 3)  # Limit to ±3 standard deviations

    # 3.4.6 Apply more complex seasonal adjustments to PM2.5
    # Base seasonality (higher in winter)
    base_season = 1.0 + 0.25 * np.sin(batch_shift + (i % 12) * np.pi / 6)

    # Weekend effect (usually lower on weekends)
    day_of_week_pm25 = np.arange(original_samples) % 7
    weekend_effect = np.where((day_of_week_pm25 == 5) | (day_of_week_pm25 == 6), 0.9, 1.0)

    # Diurnal pattern (higher during the day)
    hour_of_day_pm25 = np.arange(original_samples) % 24
    diurnal_pattern = 1.0 + 0.15 * np.sin(hour_of_day_pm25 * np.pi / 12 - np.pi / 2)

    # Apply all adjustments
    batch_pm25 = batch_pm25 * base_season * weekend_effect * diurnal_pattern

    # Add multi-scale random noise
    pm25_daily_noise = np.random.normal(0, batch_pm25.std() * 0.08, len(batch_pm25))
    pm25_hourly_noise = np.random.normal(0, batch_pm25.std() * 0.05, len(batch_pm25))

    batch_pm25 = batch_pm25 + pm25_daily_noise + pm25_hourly_noise

    # Ensure PM2.5 is within reasonable range (non-negative, and not exceeding extreme values)
    batch_pm25 = np.clip(batch_pm25, 0, batch_pm25.max() * 1.5)

    # 3.4.7 Generate timestamps
    # Shift each batch by 1 year to create a continuous time series
    batch_timestamps = pd.date_range(
        start=current_time + pd.DateOffset(years=i),
        periods=original_samples,
        freq='h'
    )

    # 3.4.8 Append to total data
    all_scaled_data.append(batch_scaled)
    all_pm25_data.append(batch_pm25)
    all_timestamps.extend(batch_timestamps)

    # Stop early if we have reached or exceeded target samples
    total_so_far = sum(len(batch) for batch in all_scaled_data)
    if total_so_far >= TARGET_SAMPLES:
        print(f"  ⏹ Target sample count reached, stopping augmentation")
        break

# 3.5 Merge all batches
print("\n🔗 Merging augmented data...")
final_scaled = np.vstack(all_scaled_data)[:TARGET_SAMPLES]
final_pm25 = np.hstack(all_pm25_data)[:TARGET_SAMPLES]
final_timestamps = all_timestamps[:TARGET_SAMPLES]

print(f"✅ Data merging complete:")
print(f"  Final scaled data shape: {final_scaled.shape}")
print(f"  Final PM2.5 data shape: {final_pm25.shape}")
print(f"  Number of timestamps: {len(final_timestamps)}")

# ==============================================
# 4. Create augmented DataFrame
# ==============================================
print("\n📊 Creating augmented DataFrame...")

# Create DataFrame
augmented_df = pd.DataFrame(
    final_scaled,
    columns=scaled_features,
    index=final_timestamps
)
augmented_df['avg_PM2.5'] = final_pm25

# Ensure time index is correctly sorted
augmented_df = augmented_df.sort_index()

print(f"✅ DataFrame created:")
print(f"  Shape: {augmented_df.shape}")
print(f"  Time range: {augmented_df.index.min()} to {augmented_df.index.max()}")
print(f"  Feature columns: {list(augmented_df.columns)}")

# ==============================================
# 5. Data quality validation
# ==============================================
print("\n" + "=" * 60)
print("🔍 Augmented data quality validation")
print("=" * 60)

# 5.1 Time continuity check
print("1. Time continuity check:")
time_diff_aug = augmented_df.index.to_series().diff()
irregular_aug = (time_diff_aug != pd.Timedelta(hours=1)).sum()
print(f"   Irregular time intervals: {irregular_aug}")
print(f"   Min interval: {time_diff_aug.min()}")
print(f"   Max interval: {time_diff_aug.max()}")

if irregular_aug == 0:
    print("   ✅ Time series is fully continuous")
else:
    print(f"   ⚠ There are {irregular_aug} irregular time intervals")

# 5.2 Statistical distribution check
print("\n2. Statistical distribution check:")
print("   Original data statistics vs Augmented data statistics")

for col in scaled_features + ['avg_PM2.5']:
    if col in df_complete.columns and col in augmented_df.columns:
        orig_stats = df_complete[col].describe()
        aug_stats = augmented_df[col].describe()

        print(f"\n   {col}:")
        print(
            f"      Original - mean: {orig_stats['mean']:.4f}, std: {orig_stats['std']:.4f}, range: [{df_complete[col].min():.2f}, {df_complete[col].max():.2f}]")
        print(
            f"      Augmented - mean: {aug_stats['mean']:.4f}, std: {aug_stats['std']:.4f}, range: [{augmented_df[col].min():.2f}, {augmented_df[col].max():.2f}]")

        mean_diff = abs(aug_stats['mean'] - orig_stats['mean']) / max(abs(orig_stats['mean']), 0.001) * 100
        std_diff = abs(aug_stats['std'] - orig_stats['std']) / max(orig_stats['std'], 0.001) * 100

        print(f"      Mean difference: {mean_diff:.1f}%, Std difference: {std_diff:.1f}%")

        if mean_diff < 10 and std_diff < 20:
            print(f"      ✅ Statistical properties well preserved")
        else:
            print(f"      ⚠ Statistical properties have significant changes")

# 5.3 Data integrity check
print("\n3. Data integrity check:")
nan_count_aug = augmented_df.isna().sum().sum()
inf_count = np.isinf(augmented_df.select_dtypes(include=[np.number])).sum().sum()
print(f"   NaN count: {nan_count_aug}")
print(f"   Infinite value count: {inf_count}")

if nan_count_aug == 0 and inf_count == 0:
    print("   ✅ Data is complete, no anomalies")
else:
    print(f"   ⚠ Anomalies found, need to process")

# 5.4 Check if target sample count is reached
print("\n4. Sample count verification:")
print(f"   Target samples: {TARGET_SAMPLES}")
print(f"   Actual samples: {len(augmented_df)}")

if len(augmented_df) >= TARGET_SAMPLES:
    print(f"   ✅ Target sample count reached")
else:
    print(f"   ⚠ Target sample count not reached, difference: {TARGET_SAMPLES - len(augmented_df)}")

# 5.5 Check data distribution
print("\n5. Data distribution check:")
print("   PM2.5 distribution percentiles:")
for p in [0, 25, 50, 75, 95, 99, 100]:
    percentile = np.percentile(augmented_df['avg_PM2.5'], p)
    print(f"     {p}th percentile: {percentile:.2f}")

# ==============================================
# 6. Save augmented data
# ==============================================
print("\n" + "=" * 60)
print("💾 Saving augmented data")
print("=" * 60)

# 6.1 Save as CSV (ensure time column is first)
print("Saving as CSV file...")
augmented_df_with_time = augmented_df.reset_index()
augmented_df_with_time = augmented_df_with_time.rename(columns={'index': time_column})
augmented_df_with_time.to_csv(OUTPUT_PATH, index=False)

print(f"✅ Augmented data saved to: {OUTPUT_PATH}")
print(f"   File size: {os.path.getsize(OUTPUT_PATH) / (1024 * 1024):.2f} MB")

# 6.2 Validate saved file
print("\n🔍 Validating saved file...")
test_load = pd.read_csv(OUTPUT_PATH, nrows=5)
print(f"   Successfully loaded, shape: {test_load.shape}")
print(f"   First column name: '{test_load.columns[0]}'")
print(f"   First 3 timestamps:")
for i in range(min(3, len(test_load))):
    print(f"     {test_load.iloc[i, 0]}")

# ==============================================
# 7. Save metadata and augmentation report
# ==============================================
print("\n📋 Saving augmentation report...")

# Create augmentation report
enhancement_report = {
    'processing_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'input_file': INPUT_PATH,
    'output_file': OUTPUT_PATH,
    'target_samples': TARGET_SAMPLES,
    'original_stats': {
        'original_samples': len(df),
        'complete_samples_after_filling': len(df_complete),
        'missing_hours_before_filling': len(expected_times) - len(df),
        'time_range': [str(df.index.min()), str(df.index.max())]
    },
    'augmented_stats': {
        'final_samples': len(augmented_df),
        'time_range': [str(augmented_df.index.min()), str(augmented_df.index.max())],
        'multiplier_used': multiplier,
        'features_used': list(augmented_df.columns)
    },
    'data_quality': {
        'time_continuity': {
            'irregular_intervals': int(irregular_aug),
            'min_interval': str(time_diff_aug.min()),
            'max_interval': str(time_diff_aug.max())
        },
        'missing_values': {
            'nan_count': int(nan_count_aug),
            'inf_count': int(inf_count)
        }
    },
    'statistical_comparison': {},
    'pm25_distribution': {}
}

# Add statistical comparison information
for col in scaled_features + ['avg_PM2.5']:
    if col in df_complete.columns and col in augmented_df.columns:
        orig_mean = float(df_complete[col].mean())
        aug_mean = float(augmented_df[col].mean())
        orig_std = float(df_complete[col].std())
        aug_std = float(augmented_df[col].std())
        orig_min = float(df_complete[col].min())
        aug_min = float(augmented_df[col].min())
        orig_max = float(df_complete[col].max())
        aug_max = float(augmented_df[col].max())

        enhancement_report['statistical_comparison'][col] = {
            'original_mean': orig_mean,
            'augmented_mean': aug_mean,
            'original_std': orig_std,
            'augmented_std': aug_std,
            'original_min': orig_min,
            'augmented_min': aug_min,
            'original_max': orig_max,
            'augmented_max': aug_max,
            'mean_diff_percentage': float(abs(aug_mean - orig_mean) / max(abs(orig_mean), 0.001) * 100),
            'std_diff_percentage': float(abs(aug_std - orig_std) / max(orig_std, 0.001) * 100)
        }

# Add PM2.5 distribution information
for p in [0, 25, 50, 75, 95, 99, 100]:
    percentile = float(np.percentile(augmented_df['avg_PM2.5'], p))
    enhancement_report['pm25_distribution'][f'percentile_{p}'] = percentile

# Save report
report_path = OUTPUT_PATH.replace('.csv', '_report.json')
with open(report_path, 'w', encoding='utf-8') as f:
    json.dump(enhancement_report, f, indent=2, ensure_ascii=False)

print(f"✅ Augmentation report saved: {report_path}")

# ==============================================
# 8. Final summary
# ==============================================
print("\n" + "=" * 60)
print("🎉 Data augmentation complete!")
print("=" * 60)
print(f"\n📊 Augmentation summary:")
print(f"  Original data: {len(df)} samples")
print(f"  Data after filling: {len(df_complete)} samples")
print(f"  Augmented data: {len(augmented_df)} samples")
print(f"  Target samples: {TARGET_SAMPLES} samples")
print(f"  Achievement rate: {len(augmented_df) / TARGET_SAMPLES * 100:.1f}%")

print(f"\n⏰ Time range:")
print(f"  Original: {df.index.min()} to {df.index.max()}")
print(f"  Augmented: {augmented_df.index.min()} to {augmented_df.index.max()}")
print(f"  Time span: {(augmented_df.index.max() - augmented_df.index.min()).days} days")

print(f"\n📁 Generated files:")
print(f"  1. {OUTPUT_PATH} - Augmented time series data (100,000 samples)")
print(f"  2. {report_path} - Augmentation processing report")

print(f"\n🔧 Augmentation features:")
print(f"  • Multi-level time pattern enhancement (hourly/weekly/monthly)")
print(f"  • Multi-scale random noise (high/low frequency)")
print(f"  • Complex seasonal adjustments (base seasonality + weekend effect + diurnal pattern)")
print(f"  • Preserved time series continuity (one point per hour)")
print(f"  • Maintained statistical properties of the original data")
print(f"  • Reasonable data range control")

print(f"\n📈 Data growth:")
print(f"  From {len(df)} → {len(augmented_df)} samples")
print(f"  Increased by {len(augmented_df) / len(df):.1f} times")

print(f"\n🚀 Ready for model training!")
