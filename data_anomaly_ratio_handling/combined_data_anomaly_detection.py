import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the combined data
print("Loading combined environmental data...")
combined_df = pd.read_csv(r"D:\Oswaldo's surf project\DR O's database\combined_environmental_data.csv")
print(f"Combined dataset shape: {combined_df.shape}")

# View data structure and columns
print(f"\nData columns: {list(combined_df.columns)}")
print(f"First 5 rows:")
print(combined_df.head())

# Check index information
print(f"\nIndex information:")
print(f"Index name: {combined_df.index.name}")
print(f"Index type: {type(combined_df.index)}")

# Set the correct datetime index
# First column is the timestamp column without a name
# We need to use the first column as the index
print(f"\nSetting first column as datetime index...")

# Check if the first column is datetime-like
first_column_name = combined_df.columns[0]
print(f"First column name: '{first_column_name}'")

# Convert the first column to datetime and set as index
combined_df['datetime'] = pd.to_datetime(combined_df[first_column_name], errors='coerce')
combined_df = combined_df.set_index('datetime')
combined_df = combined_df.sort_index()

# Drop the original first column if it's still in the dataframe
if first_column_name in combined_df.columns:
    combined_df = combined_df.drop(columns=[first_column_name])

print(f"\nTime range: {combined_df.index.min()} to {combined_df.index.max()}")
print(f"Total hours: {len(combined_df)}")

# Check data quality
print(f"\nData statistics summary:")
print(combined_df[['avg_PM2.5', 'total_noise_duration', 'noise_event_count', 'avg_salience']].describe())


# Redesigned anomaly detection rules (based on aggregated data, maintaining original rule spirit)
def detect_anomalies_combined(combined_df):
    """
    Anomaly detection based on aggregated data (maintaining original rule spirit)
    """
    # Initialize anomaly flags
    combined_df['air_anomaly'] = False
    combined_df['noise_anomaly'] = False
    combined_df['overall_anomaly'] = False

    # 1. Air anomaly (unchanged): PM2.5 > 75
    air_anomaly_threshold = 75
    combined_df['air_anomaly'] = combined_df['avg_PM2.5'] > air_anomaly_threshold

    print(f"\nAir anomaly standard: PM2.5 > {air_anomaly_threshold}")
    print(f"Air anomaly hours detected: {combined_df['air_anomaly'].sum()}")

    # 2. Noise anomaly (redesigned, maintaining original rule spirit)
    # Original rule spirit: certain categories (gun_shot, siren) are absolute anomalies, others based on duration and salience
    # For aggregated data, we can only use total duration, event count, and average salience

    # Calculate statistics
    duration_mean = combined_df['total_noise_duration'].mean()
    duration_std = combined_df['total_noise_duration'].std()
    event_mean = combined_df['noise_event_count'].mean()
    event_std = combined_df['noise_event_count'].std()

    print(f"\nNoise statistics:")
    print(f"  Average hourly total duration: {duration_mean:.2f} seconds")
    print(f"  Average hourly event count: {event_mean:.2f}")

    # Rule 1: High event count + high average salience (similar to absolute anomaly categories in original rules)
    high_event_threshold = event_mean + event_std
    high_salience_threshold = 1.5  # average salience > 1.5

    rule1_mask = (combined_df['noise_event_count'] > high_event_threshold) & \
                 (combined_df['avg_salience'] > high_salience_threshold)

    print(f"\nRule 1 - High event count + high salience:")
    print(f"  Event count threshold: > {high_event_threshold:.2f}")
    print(f"  Salience threshold: > {high_salience_threshold}")
    print(f"  Anomaly hours detected: {rule1_mask.sum()}")

    # Rule 2: Extremely long total duration
    extreme_duration_threshold = duration_mean + 2 * duration_std
    rule2_mask = combined_df['total_noise_duration'] > extreme_duration_threshold

    # Exclude already marked
    rule2_mask = rule2_mask & ~rule1_mask

    print(f"\nRule 2 - Extremely long total duration:")
    print(f"  Threshold: > {extreme_duration_threshold:.2f} seconds")
    print(f"  Anomaly hours detected: {rule2_mask.sum()}")

    # Rule 3: Medium event count + medium duration
    medium_event_threshold = event_mean + 0.5 * event_std
    medium_duration_threshold = duration_mean + 0.5 * duration_std

    rule3_mask = (combined_df['noise_event_count'] > medium_event_threshold) & \
                 (combined_df['total_noise_duration'] > medium_duration_threshold)

    # Exclude already marked
    rule3_mask = rule3_mask & ~rule1_mask & ~rule2_mask

    print(f"\nRule 3 - Medium event count + medium duration:")
    print(f"  Event count threshold: > {medium_event_threshold:.2f}")
    print(f"  Duration threshold: > {medium_duration_threshold:.2f} seconds")
    print(f"  Anomaly hours detected: {rule3_mask.sum()}")

    # Rule 4: High average salience + medium duration
    rule4_mask = (combined_df['avg_salience'] > high_salience_threshold) & \
                 (combined_df['total_noise_duration'] > medium_duration_threshold)

    # Exclude already marked
    rule4_mask = rule4_mask & ~rule1_mask & ~rule2_mask & ~rule3_mask

    print(f"\nRule 4 - High salience + medium duration:")
    print(f"  Salience threshold: > {high_salience_threshold}")
    print(f"  Duration threshold: > {medium_duration_threshold:.2f} seconds")
    print(f"  Anomaly hours detected: {rule4_mask.sum()}")

    # Combined noise anomaly: meet any rule
    combined_df['noise_anomaly'] = rule1_mask | rule2_mask | rule3_mask | rule4_mask

    print(f"\nTotal noise anomaly hours: {combined_df['noise_anomaly'].sum()}")

    # 3. Overall anomaly (loose definition): air anomaly OR noise anomaly
    combined_df['overall_anomaly'] = combined_df['air_anomaly'] | combined_df['noise_anomaly']

    return combined_df


# Apply anomaly detection
print(f"\n{'=' * 60}")
print("Applying anomaly detection rules (based on aggregated data, maintaining original rule spirit)")
print(f"{'=' * 60}")

combined_df = detect_anomalies_combined(combined_df)

# Calculate anomaly ratios
total_hours = len(combined_df)
air_anomaly_count = combined_df['air_anomaly'].sum()
noise_anomaly_count = combined_df['noise_anomaly'].sum()
overall_anomaly_count = combined_df['overall_anomaly'].sum()

air_anomaly_ratio = air_anomaly_count / total_hours
noise_anomaly_ratio = noise_anomaly_count / total_hours
overall_anomaly_ratio = overall_anomaly_count / total_hours

print(f"\n{'=' * 60}")
print("Anomaly ratio calculation results")
print(f"{'=' * 60}")
print(f"Total hours: {total_hours}")
print(f"\n1. Air anomaly (PM2.5 > 75):")
print(f"   Anomaly hours: {air_anomaly_count}")
print(f"   Anomaly ratio: {air_anomaly_ratio:.4f} ({air_anomaly_ratio:.2%})")

print(f"\n2. Noise anomaly (based on aggregated data):")
print(f"   Anomaly hours: {noise_anomaly_count}")
print(f"   Anomaly ratio: {noise_anomaly_ratio:.4f} ({noise_anomaly_ratio:.2%})")

print(f"\n3. Overall anomaly (air OR noise - loose definition):")
print(f"   Anomaly hours: {overall_anomaly_count}")
print(f"   Anomaly ratio: {overall_anomaly_ratio:.4f} ({overall_anomaly_ratio:.2%})")

# Cross-analysis
print(f"\n{'=' * 60}")
print("Anomaly cross-analysis")
print(f"{'=' * 60}")

both_anomaly = ((combined_df['air_anomaly']) & (combined_df['noise_anomaly'])).sum()
only_air_anomaly = ((combined_df['air_anomaly']) & (~combined_df['noise_anomaly'])).sum()
only_noise_anomaly = ((~combined_df['air_anomaly']) & (combined_df['noise_anomaly'])).sum()
no_anomaly = ((~combined_df['air_anomaly']) & (~combined_df['noise_anomaly'])).sum()

print(f"Both air and noise anomaly: {both_anomaly} ({both_anomaly / total_hours:.2%})")
print(f"Only air anomaly: {only_air_anomaly} ({only_air_anomaly / total_hours:.2%})")
print(f"Only noise anomaly: {only_noise_anomaly} ({only_noise_anomaly / total_hours:.2%})")
print(f"Both normal: {no_anomaly} ({no_anomaly / total_hours:.2%})")

# Verify total
print(f"\nVerification total: {both_anomaly + only_air_anomaly + only_noise_anomaly + no_anomaly} = {total_hours}")

# Save results
output_path = r"D:\Oswaldo's surf project\DR O's database\data_anomaly_ratio_handling\combined_anomaly_analysis_refined.csv"
combined_df.to_csv(output_path, index=True, encoding='utf-8-sig')
print(f"\nAnalysis results saved to: {output_path}")

# Generate final report
print(f"\n{'=' * 60}")
print("Final anomaly analysis report")
print(f"{'=' * 60}")
print(f"Dataset: Combined environmental data (hourly aggregation)")
print(f"Total hours: {total_hours}")

# Safely display time range
try:
    # Try to get date part
    min_date = combined_df.index.min()
    max_date = combined_df.index.max()

    # Check if it's datetime type
    if hasattr(min_date, 'date') and hasattr(max_date, 'date'):
        print(f"Time range: {min_date.date()} to {max_date.date()}")
    else:
        print(f"Time range: {min_date} to {max_date}")
except:
    print("Cannot get time range information")

print(f"\nOriginal data anomaly ratios:")
print(f"  PRSA PM2.5 anomaly: 39.07%")
print(f"  UrbanSound8K noise anomaly: 17.69%")
print(f"\nCombined data anomaly ratios (maintaining original rule spirit):")
print(f"  Air anomaly (PM2.5>75): {air_anomaly_count} hours, {air_anomaly_ratio:.2%}")
print(f"  Noise anomaly (aggregated metrics): {noise_anomaly_count} hours, {noise_anomaly_ratio:.2%}")
print(f"  Overall anomaly (air OR noise): {overall_anomaly_count} hours, {overall_anomaly_ratio:.2%}")

# Simple visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 1. Anomaly type distribution
labels = ['Both air and noise anomaly', 'Only air anomaly', 'Only noise anomaly', 'Both normal']
sizes = [both_anomaly, only_air_anomaly, only_noise_anomaly, no_anomaly]
colors = ['red', 'orange', 'yellow', 'lightgreen']

axes[0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
axes[0].set_title('Anomaly type distribution')
axes[0].axis('equal')

# 2. Anomaly ratio comparison
categories = ['Air anomaly', 'Noise anomaly', 'Overall anomaly']
ratios = [air_anomaly_ratio * 100, noise_anomaly_ratio * 100, overall_anomaly_ratio * 100]

bars = axes[1].bar(categories, ratios, color=['orange', 'blue', 'green'], alpha=0.7)
axes[1].set_xlabel('Anomaly type')
axes[1].set_ylabel('Anomaly ratio (%)')
axes[1].set_title('Anomaly ratio comparison')
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].set_ylim(0, max(ratios) * 1.2)

for bar, ratio in zip(bars, ratios):
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width() / 2., height + 1,
                 f'{ratio:.1f}%', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(
    r"D:\Oswaldo's surf project\DR O's database\data_anomaly_ratio_handling\combined_anomaly_refined_analysis.png",
    dpi=300, bbox_inches='tight')
plt.close()

print(f"\n{'=' * 60}")
print("Analysis completed!")
print(f"{'=' * 60}")
