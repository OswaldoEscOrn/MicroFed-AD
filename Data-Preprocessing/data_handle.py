import pandas as pd
import random
import os
import numpy as np

# ==================== Configuration paths ====================
input_path = r"D:\Oswaldo's surf project\My Database\PM2.5\PM25_data.csv"
output_path = r"D:\Oswaldo's surf project\My Database\PM2.5\random_city3PM25_data.csv"

# Ensure output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# ==================== 1. Load original data ====================
print("🔍 Loading original data...")
df = pd.read_csv(input_path, encoding='gbk')  # Original file contains Chinese, use GBK encoding
print(f"Original data shape: {df.shape}")

# Get all column names
all_columns = df.columns.tolist()
date_column = all_columns[0]
hour_column = all_columns[1]
city_columns = all_columns[2:]  # Starting from third column are cities

print(f"Date column: {date_column}, Hour column: {hour_column}")
print(f"Total {len(city_columns)} city columns")

# ==================== 2. Randomly select 5 cities ====================
selected_cities = random.sample(city_columns, 5)
print(f"\n🎲 Randomly selected 5 cities: {selected_cities}")

# Create a new DataFrame with date, hour and the 5 selected cities
selected_columns = [date_column, hour_column] + selected_cities
df_selected = df[selected_columns].copy()
print(f"Data shape after selection: {df_selected.shape}")

# ==================== 3. Construct timestamp column ====================
print("\n⏰ Constructing timestamp column...")
# Convert date to string, zero-pad hour to two digits, combine and parse as datetime
df_selected['timestamp'] = pd.to_datetime(
    df_selected[date_column].astype(str) + df_selected[hour_column].astype(str).str.zfill(2),
    format='%Y%m%d%H'
)
# Format as readable string (optional, for easier viewing when saved)
df_selected['timestamp'] = df_selected['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

# Rearrange columns: put timestamp first
cols = ['timestamp'] + selected_cities
df_final = df_selected[cols].copy()
print(f"Data shape after adding timestamp: {df_final.shape}")

# ==================== 4. Handle missing values ====================
print("\n🧹 Handling missing values...")

# 4.1 Missing values in timestamp column (theoretically none, but keep forward-fill logic)
df_final['timestamp'] = pd.to_datetime(df_final['timestamp'])  # convert back to datetime for filling
df_final['timestamp'] = df_final['timestamp'].fillna(method='ffill')
df_final['timestamp'] = df_final['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')  # convert back to string

# 4.2 Missing values in city columns: fill with column mean
for city in selected_cities:
    city_mean = df_final[city].mean()
    df_final[city] = df_final[city].fillna(city_mean)
    print(f"  Column {city} mean: {city_mean:.2f}")

# Check missing values after processing
print("\nMissing value statistics after processing:")
print(df_final.isnull().sum())

# ==================== 5. Save final file ====================
df_final.to_csv(output_path, index=False, encoding='utf-8')
print(f"\n✅ Data saved to: {output_path}")
print(f"Final data shape: {df_final.shape}")

# ==================== 6. Preview data ====================
print("\n📋 Preview of first 5 rows:")
print(df_final.head())
