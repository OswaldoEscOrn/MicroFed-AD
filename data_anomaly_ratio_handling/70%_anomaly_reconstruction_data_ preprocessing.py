import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import json
import joblib

# ==============================================
# 配置参数
# ==============================================
DATA_PATH = r"D:\Oswaldo's surf project\DR O's database\data_anomaly_ratio_handling\reconstruction_data_70%_anomaly_fixed.csv"
OUTPUT_DIR = r"D:\Oswaldo's surf project\DR O's database\final_preprocessed_data_70%_anomaly_complete"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 窗口参数
WINDOW_SIZE = 24
STRIDE = 1
PREDICTION_HORIZON = 1

# 需要处理的5个目标列
TARGET_COLUMNS = [
    'avg_PM2.5_normalized',
    'total_noise_duration',
    'noise_event_count',
    'avg_salience',
    'avg_PM2.5'
]

# ==============================================
# 1. 加载数据并确保时间戳是第一列
# ==============================================
print("📥 加载数据...")
df = pd.read_csv(DATA_PATH)

# 检查第一列是什么
print(f"原始CSV列: {list(df.columns)}")
print(f"第一列名称: '{df.columns[0]}'")

# 重命名第一列
time_column = df.columns[0]
df.rename(columns={time_column: 'timestamp'}, inplace=True)

# 转换为datetime并设置为索引
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y/%m/%d %H:%M', errors='coerce')
df = df.set_index('timestamp')
df = df.sort_index()

print(f"✅ 数据加载完成:")
print(f"   时间范围: {df.index.min()} 到 {df.index.max()}")
print(f"   总行数: {len(df)}")
print(f"   列: {list(df.columns)}")

# ==============================================
# 2. 检查目标列
# ==============================================
print("\n🔍 检查目标列...")
available_columns = []
for col in TARGET_COLUMNS:
    if col in df.columns:
        available_columns.append(col)
        print(f"   ✓ {col}: 存在")
    else:
        print(f"   ✗ {col}: 缺失")

# 如果目标列不全，退出
if not available_columns:
    print("❌ 没有找到任何目标列！")
    exit()

# 创建包含目标列的DataFrame
processed_df = pd.DataFrame(index=df.index)
for col in available_columns:
    processed_df[col] = df[col]

print(f"\n处理后的数据形状: {processed_df.shape}")

# ==============================================
# 3. 标准化处理
# ==============================================
print("\n⚖️ 执行标准化...")

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
# 4. 创建最终DataFrame（确保列顺序正确）
# ==============================================
print("\n🎯 整理最终列...")

# 定义最终列
FINAL_COLUMNS = [
    'avg_PM2.5_normalized_scaled',
    'total_noise_duration_scaled',
    'noise_event_count_scaled',
    'avg_salience_scaled',
    'avg_PM2.5'
]

# 创建最终DataFrame（从processed_df中提取需要的列）
final_df = pd.DataFrame(index=processed_df.index)

for col in FINAL_COLUMNS:
    if col in processed_df.columns:
        final_df[col] = processed_df[col]
    else:
        # 尝试找原始列
        alt_col = col.replace('_scaled', '')
        if alt_col in processed_df.columns:
            final_df[col] = processed_df[alt_col]
            print(f"   ⚠ 使用 {alt_col} 替代 {col}")
        else:
            print(f"   ⚠ 列 {col} 不存在")

print(f"   最终数据形状: {final_df.shape}")
print(f"   列: {list(final_df.columns)}")

# ==============================================
# 5. 创建滑动窗口
# ==============================================
print(f"\n🪟 创建滑动窗口...")

# 获取时间戳和数据
timestamps = final_df.index.values
data_array = final_df.values


def create_sliding_windows(data, timestamps, window_size, stride=1, horizon=0):
    """创建滑动窗口，返回数据、目标和时间戳"""
    X, y, X_timestamps = [], [], []
    n_samples = len(data)

    for i in range(0, n_samples - window_size - horizon + 1, stride):
        # 数据窗口
        window = data[i:i + window_size]
        X.append(window)

        # 时间戳窗口的开始时间
        X_timestamps.append(timestamps[i])

        if horizon > 0:
            target = data[i + window_size:i + window_size + horizon]
            y.append(target)

    X = np.array(X)
    if horizon > 0:
        y = np.array(y)
        return X, y, np.array(X_timestamps)
    return X, None, np.array(X_timestamps)


# 创建滑动窗口
X_windows, y_windows, window_timestamps = create_sliding_windows(
    data=data_array,
    timestamps=timestamps,
    window_size=WINDOW_SIZE,
    stride=STRIDE,
    horizon=PREDICTION_HORIZON
)

print(f"   创建的窗口数: {X_windows.shape[0]:,}")
print(f"   窗口形状: {X_windows.shape}")
print(f"   时间戳数量: {len(window_timestamps)}")

# ==============================================
# 6. 保存数据（关键修复：确保时间戳是第一列）
# ==============================================
print("\n💾 保存数据...")

# 6.1 保存滑动窗口数据（numpy格式）
np.save(os.path.join(OUTPUT_DIR, "X_windows.npy"), X_windows)
np.save(os.path.join(OUTPUT_DIR, "window_timestamps.npy"), window_timestamps)

if y_windows is not None:
    np.save(os.path.join(OUTPUT_DIR, "y_windows.npy"), y_windows)

print(f"   ✓ 滑动窗口保存")

# 6.2 保存原始时间序列数据（确保时间戳是第一列）
print("\n📊 保存时间序列数据...")

# 方法1：直接使用to_csv，确保时间列是第一列
csv_path = os.path.join(OUTPUT_DIR, "preprocessed_time_series.csv")

# 重置索引，时间戳会成为第一列
final_df_with_time = final_df.reset_index()

# 确保列名正确
final_df_with_time = final_df_with_time.rename(columns={'timestamp': time_column})

# 保存CSV
final_df_with_time.to_csv(csv_path, index=False)

print(f"   ✓ 时间序列CSV保存: {csv_path}")

# 验证保存的CSV
print(f"\n🔍 验证保存的CSV文件:")
test_df = pd.read_csv(csv_path, nrows=3)
print(f"   列名: {list(test_df.columns)}")
print(f"   第一列: '{test_df.columns[0]}'")
print(f"   前3行第一列的值:")
for i in range(min(3, len(test_df))):
    print(f"     行{i + 1}: {test_df.iloc[i, 0]}")

# 6.3 创建窗口数据CSV（展平窗口，包含时间戳）
print("\n📊 创建窗口数据CSV...")

n_windows = X_windows.shape[0]
n_timesteps = X_windows.shape[1]
n_features = X_windows.shape[2]

# 创建列名
feature_names = final_df.columns.tolist()
column_names = [time_column]  # 使用原始时间列名

# 为每个时间步和特征创建列名
for t in range(n_timesteps):
    for f_idx, feature in enumerate(feature_names):
        column_names.append(f"{feature}_t{t + 1}")

# 展平窗口数据
flattened_data = X_windows.reshape(n_windows, n_timesteps * n_features)

# 创建DataFrame（关键：将时间戳放在第一列）
window_df = pd.DataFrame(flattened_data, columns=column_names[1:])
window_df.insert(0, time_column, pd.to_datetime(window_timestamps))

# 保存窗口数据CSV
window_csv_path = os.path.join(OUTPUT_DIR, "window_data.csv")
window_df.to_csv(window_csv_path, index=False)
print(f"   ✓ 窗口数据CSV保存: {window_csv_path}")

# 验证窗口CSV
print(f"\n🔍 验证窗口CSV:")
window_test = pd.read_csv(window_csv_path, nrows=2)
print(f"   列数: {len(window_test.columns)}")
print(f"   第一列: '{window_test.columns[0]}'")
print(f"   第一行第一列值: {window_test.iloc[0, 0]}")

# 6.4 保存标准化器
scaler_dir = os.path.join(OUTPUT_DIR, "scalers")
os.makedirs(scaler_dir, exist_ok=True)

for col_name, scaler in scalers.items():
    scaler_path = os.path.join(scaler_dir, f"scaler_{col_name}.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"   ✓ 标准化器保存: scaler_{col_name}.pkl")

# 6.5 保存元数据
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

print(f"   ✓ 元数据保存: {metadata_path}")

# ==============================================
# 7. 最终验证
# ==============================================
print("\n" + "=" * 60)
print("🔍 最终验证")
print("=" * 60)

# 重新加载所有文件验证
print("1. 重新加载时间序列CSV...")
ts_df = pd.read_csv(csv_path)
print(f"   形状: {ts_df.shape}")
print(f"   第一列名: '{ts_df.columns[0]}'")
print(f"   第一列类型: {type(ts_df.iloc[0, 0])}")

print("\n2. 重新加载窗口CSV...")
win_df = pd.read_csv(window_csv_path)
print(f"   形状: {win_df.shape}")
print(f"   第一列名: '{win_df.columns[0]}'")

print("\n3. 加载滑动窗口数据...")
X_loaded = np.load(os.path.join(OUTPUT_DIR, "X_windows.npy"))
timestamps_loaded = np.load(os.path.join(OUTPUT_DIR, "window_timestamps.npy"))
print(f"   X_windows形状: {X_loaded.shape}")
print(f"   时间戳数量: {len(timestamps_loaded)}")

# ==============================================
# 8. 总结
# ==============================================
print("\n" + "=" * 60)
print("🎉 预处理完成!")
print("=" * 60)
print(f"输出目录: {OUTPUT_DIR}")
print(f"\n生成的文件:")
print(f"  1. preprocessed_time_series.csv - 时间序列数据 ({ts_df.shape[0]}行, {ts_df.shape[1]}列)")
print(f"     第一列: '{ts_df.columns[0]}'")
print(f"  2. window_data.csv - 窗口数据 ({win_df.shape[0]}窗口, {win_df.shape[1]}列)")
print(f"     第一列: '{win_df.columns[0]}'")
print(f"  3. X_windows.npy - 滑动窗口数组 (形状: {X_windows.shape})")
print(f"  4. window_timestamps.npy - 窗口开始时间")
print(f"  5. scalers/ - 标准化器文件夹")
print(f"  6. processing_metadata.json - 元数据")

print(f"\n数据统计:")
print(f"  原始样本数: {len(final_df)}")
print(f"  创建窗口数: {X_windows.shape[0]}")
print(f"  每个窗口: {X_windows.shape[1]}小时, {X_windows.shape[2]}个特征")
print(f"  时间范围: {final_df.index.min()} 到 {final_df.index.max()}")

# 显示前几行数据
print(f"\n时间序列数据预览（前3行）:")
print(ts_df.head(3).to_string(index=False))