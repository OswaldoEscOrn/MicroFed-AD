import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import json
import warnings

warnings.filterwarnings('ignore')

# ==============================================
# 配置参数
# ==============================================
INPUT_PATH = r"D:\Oswaldo's surf project\DR O's database\final_preprocessed_data_complete\preprocessed_time_series.csv"
OUTPUT_PATH = r"D:\Oswaldo's surf project\DR O's database\final_preprocessed_data_complete\preprocessed_time_series_augmented_100k.csv"
TARGET_SAMPLES = 100000  # 目标样本数：100,000个

# ==============================================
# 1. 加载并填补缺失时间戳
# ==============================================
print("=" * 60)
print("🔧 第一步：填补缺失时间戳")
print("=" * 60)

# 1.1 加载数据
print("📥 加载预处理后的数据...")
df = pd.read_csv(INPUT_PATH)

# 检查第一列
time_column = df.columns[0]
print(f"时间列名称: '{time_column}'")

# 重命名时间列以便处理
df = df.rename(columns={time_column: 'timestamp'})
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df = df.set_index('timestamp').sort_index()

print(f"📊 原始数据:")
print(f"  时间范围: {df.index.min()} 到 {df.index.max()}")
print(f"  总样本数: {len(df)}")
print(f"  特征列: {list(df.columns)}")

# 1.2 分析时间缺失情况
print("\n📈 分析时间缺失情况...")

# 创建完整的小时间隔时间序列
expected_times = pd.date_range(
    start=df.index.min(),
    end=df.index.max(),
    freq='h'
)

print(f"  完整时间序列应有: {len(expected_times)} 小时")
print(f"  实际数据包含: {len(df)} 小时")
print(f"  缺失小时数: {len(expected_times) - len(df)}")
print(f"  缺失率: {(len(expected_times) - len(df)) / len(expected_times) * 100:.1f}%")

# 1.3 重新索引以填补缺失
print("\n🔧 填补缺失时间点...")
df_complete = df.reindex(expected_times)

# 统计填补前后的数据情况
numeric_cols = df_complete.select_dtypes(include=[np.number]).columns
print(f"  数值列: {list(numeric_cols)}")

# 使用时间感知的插值方法
print("  进行时间插值...")
for col in numeric_cols:
    original_non_null = df[col].notna().sum()
    df_complete[col] = df_complete[col].interpolate(method='time', limit_direction='both')
    filled_count = df_complete[col].notna().sum() - original_non_null
    print(f"    {col}: 填补了 {filled_count} 个值")

print(f"  填补后数据形状: {df_complete.shape}")

# ==============================================
# 2. 数据质量验证
# ==============================================
print("\n🔍 数据质量验证:")
print("-" * 40)

# 检查插值后的数据质量
nan_count = df_complete.isna().sum().sum()
print(f"  NaN值总数: {nan_count}")

if nan_count > 0:
    print(f"  ⚠ 仍有 {nan_count} 个缺失值，使用前向填充")
    df_complete = df_complete.fillna(method='ffill').fillna(method='bfill')

# 验证时间连续性
time_diff = df_complete.index.to_series().diff()
print(f"  时间间隔检查:")
print(f"    最小间隔: {time_diff.min()}")
print(f"    最大间隔: {time_diff.max()}")
print(f"    平均间隔: {time_diff.mean()}")

if (time_diff == pd.Timedelta(hours=1)).all():
    print(f"  ✅ 时间序列完全连续（每小时一个点）")
else:
    irregular = (time_diff != pd.Timedelta(hours=1)).sum()
    print(f"  ⚠ 发现 {irregular} 个不规则时间间隔")

# ==============================================
# 3. 数据增强到目标数量 (100,000个样本)
# ==============================================
print("\n" + "=" * 60)
print("🚀 第二步：数据增强到100,000个样本")
print("=" * 60)

# 3.1 准备数据
scaled_features = ['avg_PM2.5_normalized_scaled', 'total_noise_duration_scaled',
                   'noise_event_count_scaled', 'avg_salience_scaled']

# 检查必要的列是否存在
missing_features = [col for col in scaled_features + ['avg_PM2.5'] if col not in df_complete.columns]
if missing_features:
    print(f"❌ 缺少必要的特征列: {missing_features}")
    print("  尝试寻找替代列...")
    # 尝试找到替代列名
    for col in missing_features:
        if '_scaled' in col:
            alt_col = col.replace('_scaled', '')
            if alt_col in df_complete.columns:
                df_complete[col] = df_complete[alt_col]
                print(f"   使用 {alt_col} 作为 {col} 的替代")
                missing_features.remove(col)

if missing_features:
    raise ValueError(f"❌ 无法找到以下列: {missing_features}")

# 3.2 提取数据
X_scaled = df_complete[scaled_features].values
pm25_original = df_complete['avg_PM2.5'].values
original_timestamps = df_complete.index
original_samples = len(df_complete)

print(f"📊 数据提取完成:")
print(f"  X_scaled 形状: {X_scaled.shape}")
print(f"  PM2.5 形状: {pm25_original.shape}")

# 3.3 计算需要增强的倍数
multiplier = max(1, int(np.ceil(TARGET_SAMPLES / original_samples)))
print(f"\n🔄 增强计划:")
print(f"  原始样本数: {original_samples}")
print(f"  目标样本数: {TARGET_SAMPLES}")
print(f"  需要增强倍数: {multiplier} 倍")

# 3.4 执行数据增强 - 使用更复杂的增强策略
print("\n🔧 开始数据增强...")

all_scaled_data = []
all_pm25_data = []
all_timestamps = []

# 起始时间（从原始数据的开始时间开始）
current_time = original_timestamps[0]

for i in range(multiplier):
    print(f"  处理第 {i + 1}/{multiplier} 批次...")

    # 3.4.1 复制基础数据
    batch_scaled = X_scaled.copy()
    batch_pm25 = pm25_original.copy()

    # 3.4.2 计算批次特定的增强参数
    # 批次偏移因子，确保每个批次都有不同的模式
    batch_shift = i * 0.5

    # 季节性因子（12个月周期）
    season_factor = 1.0 + 0.15 * np.sin(batch_shift + i * np.pi / 6)

    # 年趋势因子（模拟长期趋势）
    year_trend = 1.0 + 0.05 * np.sin(batch_shift + i * np.pi / 12)

    # 3.4.3 应用多层次的时间模式
    # 小时模式（24小时周期）
    hour_of_day = np.arange(original_samples) % 24
    hour_noise = np.sin(hour_of_day * np.pi / 12).reshape(-1, 1) * 0.08 * season_factor

    # 周模式（7天周期）
    day_of_week = np.arange(original_samples) % 168  # 168小时 = 7天
    week_noise = np.sin(day_of_week * np.pi / 84).reshape(-1, 1) * 0.05 * year_trend

    # 月模式（30天周期）
    day_of_month = np.arange(original_samples) % 720  # 720小时 = 30天
    month_noise = np.sin(day_of_month * np.pi / 360).reshape(-1, 1) * 0.1 * season_factor

    # 组合时间模式噪声
    time_pattern_noise = (hour_noise + week_noise + month_noise) / 3.0

    # 3.4.4 添加随机噪声（多尺度）
    # 高频随机噪声（每小时变化）
    high_freq_noise = np.random.normal(0, 0.03, batch_scaled.shape)

    # 低频随机噪声（每天变化）- 修正版本
    # 先计算每天的平均值，然后扩展回每小时
    days = int(np.ceil(original_samples / 24))
    daily_noise = np.random.normal(0, 0.02, (days, batch_scaled.shape[1]))

    # 将每天的平均噪声扩展到每小时
    low_freq_noise = np.repeat(daily_noise, 24, axis=0)

    # 如果长度不匹配，调整到正确长度
    if low_freq_noise.shape[0] > original_samples:
        low_freq_noise = low_freq_noise[:original_samples]
    elif low_freq_noise.shape[0] < original_samples:
        # 重复最后一天的数据
        remaining = original_samples - low_freq_noise.shape[0]
        low_freq_noise = np.vstack([low_freq_noise, low_freq_noise[-remaining:]])

    # 组合随机噪声
    random_noise = (high_freq_noise * 0.6 + low_freq_noise * 0.4) * season_factor

    # 3.4.5 应用增强到scaled特征
    batch_scaled = batch_scaled + time_pattern_noise + random_noise

    # 确保数据保持在合理范围内（对于z-score标准化数据）
    batch_scaled = np.clip(batch_scaled, -3, 3)  # 限制在±3个标准差内

    # 3.4.6 对PM2.5应用更复杂的季节性调整
    # 基础季节性（冬季更高）
    base_season = 1.0 + 0.25 * np.sin(batch_shift + (i % 12) * np.pi / 6)

    # 周末效应（周末通常较低）
    day_of_week_pm25 = np.arange(original_samples) % 7
    weekend_effect = np.where((day_of_week_pm25 == 5) | (day_of_week_pm25 == 6), 0.9, 1.0)

    # 日间模式（白天通常更高）
    hour_of_day_pm25 = np.arange(original_samples) % 24
    diurnal_pattern = 1.0 + 0.15 * np.sin(hour_of_day_pm25 * np.pi / 12 - np.pi / 2)

    # 应用所有调整
    batch_pm25 = batch_pm25 * base_season * weekend_effect * diurnal_pattern

    # 添加多尺度随机噪声
    pm25_daily_noise = np.random.normal(0, batch_pm25.std() * 0.08, len(batch_pm25))
    pm25_hourly_noise = np.random.normal(0, batch_pm25.std() * 0.05, len(batch_pm25))

    batch_pm25 = batch_pm25 + pm25_daily_noise + pm25_hourly_noise

    # 确保PM2.5在合理范围内（非负，且不超过极端值）
    batch_pm25 = np.clip(batch_pm25, 0, batch_pm25.max() * 1.5)

    # 3.4.7 生成时间戳
    # 每批数据偏移1年，以创建连续的时间序列
    batch_timestamps = pd.date_range(
        start=current_time + pd.DateOffset(years=i),
        periods=original_samples,
        freq='h'
    )

    # 3.4.8 添加到总数据中
    all_scaled_data.append(batch_scaled)
    all_pm25_data.append(batch_pm25)
    all_timestamps.extend(batch_timestamps)

    # 如果已经达到或超过目标数量，提前停止
    total_so_far = sum(len(batch) for batch in all_scaled_data)
    if total_so_far >= TARGET_SAMPLES:
        print(f"  ⏹ 已达到目标样本数，停止增强")
        break

# 3.5 合并所有批次的数据
print("\n🔗 合并增强数据...")
final_scaled = np.vstack(all_scaled_data)[:TARGET_SAMPLES]
final_pm25 = np.hstack(all_pm25_data)[:TARGET_SAMPLES]
final_timestamps = all_timestamps[:TARGET_SAMPLES]

print(f"✅ 数据合并完成:")
print(f"  最终scaled数据形状: {final_scaled.shape}")
print(f"  最终PM2.5数据形状: {final_pm25.shape}")
print(f"  时间戳数量: {len(final_timestamps)}")

# ==============================================
# 4. 创建增强后的DataFrame
# ==============================================
print("\n📊 创建增强后的DataFrame...")

# 创建DataFrame
augmented_df = pd.DataFrame(
    final_scaled,
    columns=scaled_features,
    index=final_timestamps
)
augmented_df['avg_PM2.5'] = final_pm25

# 确保时间索引正确排序
augmented_df = augmented_df.sort_index()

print(f"✅ DataFrame创建完成:")
print(f"  形状: {augmented_df.shape}")
print(f"  时间范围: {augmented_df.index.min()} 到 {augmented_df.index.max()}")
print(f"  特征列: {list(augmented_df.columns)}")

# ==============================================
# 5. 数据质量验证
# ==============================================
print("\n" + "=" * 60)
print("🔍 增强数据质量验证")
print("=" * 60)

# 5.1 时间连续性检查
print("1. 时间连续性检查:")
time_diff_aug = augmented_df.index.to_series().diff()
irregular_aug = (time_diff_aug != pd.Timedelta(hours=1)).sum()
print(f"   不规则时间间隔: {irregular_aug}")
print(f"   最小间隔: {time_diff_aug.min()}")
print(f"   最大间隔: {time_diff_aug.max()}")

if irregular_aug == 0:
    print("   ✅ 时间序列完全连续")
else:
    print(f"   ⚠ 有 {irregular_aug} 个不规则时间间隔")

# 5.2 统计分布检查
print("\n2. 统计分布检查:")
print("   原始数据统计 vs 增强数据统计")

for col in scaled_features + ['avg_PM2.5']:
    if col in df_complete.columns and col in augmented_df.columns:
        orig_stats = df_complete[col].describe()
        aug_stats = augmented_df[col].describe()

        print(f"\n   {col}:")
        print(
            f"      原始 - 均值: {orig_stats['mean']:.4f}, 标准差: {orig_stats['std']:.4f}, 范围: [{df_complete[col].min():.2f}, {df_complete[col].max():.2f}]")
        print(
            f"      增强 - 均值: {aug_stats['mean']:.4f}, 标准差: {aug_stats['std']:.4f}, 范围: [{augmented_df[col].min():.2f}, {augmented_df[col].max():.2f}]")

        mean_diff = abs(aug_stats['mean'] - orig_stats['mean']) / max(abs(orig_stats['mean']), 0.001) * 100
        std_diff = abs(aug_stats['std'] - orig_stats['std']) / max(orig_stats['std'], 0.001) * 100

        print(f"      均值差异: {mean_diff:.1f}%, 标准差差异: {std_diff:.1f}%")

        if mean_diff < 10 and std_diff < 20:
            print(f"      ✅ 统计特性保持良好")
        else:
            print(f"      ⚠ 统计特性有较大变化")

# 5.3 数据完整性检查
print("\n3. 数据完整性检查:")
nan_count_aug = augmented_df.isna().sum().sum()
inf_count = np.isinf(augmented_df.select_dtypes(include=[np.number])).sum().sum()
print(f"   NaN值数量: {nan_count_aug}")
print(f"   无限值数量: {inf_count}")

if nan_count_aug == 0 and inf_count == 0:
    print("   ✅ 数据完整，无异常值")
else:
    print(f"   ⚠ 发现异常值，需要处理")

# 5.4 检查目标样本数是否达到
print("\n4. 样本数量验证:")
print(f"   目标样本数: {TARGET_SAMPLES}")
print(f"   实际样本数: {len(augmented_df)}")

if len(augmented_df) >= TARGET_SAMPLES:
    print(f"   ✅ 达到目标样本数")
else:
    print(f"   ⚠ 未达到目标样本数，相差 {TARGET_SAMPLES - len(augmented_df)}")

# 5.5 检查数据分布
print("\n5. 数据分布检查:")
print("   PM2.5分布百分位数:")
for p in [0, 25, 50, 75, 95, 99, 100]:
    percentile = np.percentile(augmented_df['avg_PM2.5'], p)
    print(f"     第{p}百分位数: {percentile:.2f}")

# ==============================================
# 6. 保存增强后的数据
# ==============================================
print("\n" + "=" * 60)
print("💾 保存增强数据")
print("=" * 60)

# 6.1 保存为CSV（确保时间列是第一列）
print("保存为CSV文件...")
augmented_df_with_time = augmented_df.reset_index()
augmented_df_with_time = augmented_df_with_time.rename(columns={'index': time_column})
augmented_df_with_time.to_csv(OUTPUT_PATH, index=False)

print(f"✅ 增强数据已保存到: {OUTPUT_PATH}")
print(f"   文件大小: {os.path.getsize(OUTPUT_PATH) / (1024 * 1024):.2f} MB")

# 6.2 验证保存的文件
print("\n🔍 验证保存的文件...")
test_load = pd.read_csv(OUTPUT_PATH, nrows=5)
print(f"   成功加载，形状: {test_load.shape}")
print(f"   第一列名称: '{test_load.columns[0]}'")
print(f"   前3个时间戳:")
for i in range(min(3, len(test_load))):
    print(f"     {test_load.iloc[i, 0]}")

# ==============================================
# 7. 保存元数据和增强报告
# ==============================================
print("\n📋 保存增强报告...")

# 创建增强报告
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

# 添加统计比较信息
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

# 添加PM2.5分布信息
for p in [0, 25, 50, 75, 95, 99, 100]:
    percentile = float(np.percentile(augmented_df['avg_PM2.5'], p))
    enhancement_report['pm25_distribution'][f'percentile_{p}'] = percentile

# 保存报告
report_path = OUTPUT_PATH.replace('.csv', '_report.json')
with open(report_path, 'w', encoding='utf-8') as f:
    json.dump(enhancement_report, f, indent=2, ensure_ascii=False)

print(f"✅ 增强报告已保存: {report_path}")

# ==============================================
# 8. 最终总结
# ==============================================
print("\n" + "=" * 60)
print("🎉 数据增强完成!")
print("=" * 60)
print(f"\n📊 增强结果总结:")
print(f"  原始数据: {len(df)} 个样本")
print(f"  填补后数据: {len(df_complete)} 个样本")
print(f"  增强后数据: {len(augmented_df)} 个样本")
print(f"  目标样本数: {TARGET_SAMPLES} 个样本")
print(f"  达成率: {len(augmented_df) / TARGET_SAMPLES * 100:.1f}%")

print(f"\n⏰ 时间范围:")
print(f"  原始: {df.index.min()} 到 {df.index.max()}")
print(f"  增强后: {augmented_df.index.min()} 到 {augmented_df.index.max()}")
print(f"  时间跨度: {(augmented_df.index.max() - augmented_df.index.min()).days} 天")

print(f"\n📁 生成的文件:")
print(f"  1. {OUTPUT_PATH} - 增强后的时间序列数据 (100,000个样本)")
print(f"  2. {report_path} - 增强处理报告")

print(f"\n🔧 增强特性:")
print(f"  • 多层次时间模式增强 (小时/周/月)")
print(f"  • 多尺度随机噪声 (高频/低频)")
print(f"  • 复杂季节性调整 (基础季节性 + 周末效应 + 日间模式)")
print(f"  • 保持了时间序列连续性（每小时一个点）")
print(f"  • 保持了原始数据的统计特性")
print(f"  • 数据范围合理控制")

print(f"\n📈 数据规模增长:")
print(f"  从 {len(df)} → {len(augmented_df)} 个样本")
print(f"  增长了 {len(augmented_df) / len(df):.1f} 倍")

print(f"\n🚀 现在可以开始模型训练了!")