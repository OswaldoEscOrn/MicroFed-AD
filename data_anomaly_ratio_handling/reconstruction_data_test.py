import pandas as pd
import os


def fix_duplicated_reconstruction_data():
    """修复重复数据的reconstruction_data.csv"""

    input_path = r"D:\Oswaldo's surf project\DR O's database\data_anomaly_ratio_handling\reconstruction_data.csv"
    output_path = r"D:\Oswaldo's surf project\DR O's database\data_anomaly_ratio_handling\reconstruction_data_fixed.csv"

    print("🔍 检测并修复重复数据...")

    # 1. 加载数据
    df = pd.read_csv(input_path)

    # 重命名时间列
    time_column = df.columns[0]
    df.rename(columns={time_column: 'timestamp'}, inplace=True)

    # 转换时间格式
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y/%m/%d %H:%M', errors='coerce')

    print(f"原始数据形状: {df.shape}")
    print(f"时间列: 'timestamp'")

    # 2. 检测重复行（基于所有列）
    duplicate_rows = df.duplicated().sum()
    print(f"完全重复的行数: {duplicate_rows} ({duplicate_rows / len(df) * 100:.1f}%)")

    # 3. 检测时间戳重复
    time_duplicates = df['timestamp'].duplicated().sum()
    print(f"重复的时间戳数: {time_duplicates}")

    # 4. 按时间分组查看
    print("\n📊 按时间戳分组统计:")
    time_groups = df.groupby('timestamp').size().reset_index(name='count')

    # 统计重复次数分布
    duplicate_counts = time_groups['count'].value_counts().sort_index()
    for count, freq in duplicate_counts.items():
        print(f"  出现{count}次的时间戳: {freq}个")

    # 5. 修复：删除重复行，保留第一个
    print("\n🧹 修复数据...")

    # 方法1：按时间戳去重，保留第一个
    df_fixed = df.drop_duplicates(subset=['timestamp'], keep='first')

    print(f"修复后数据形状: {df_fixed.shape}")
    print(f"删除了 {len(df) - len(df_fixed)} 行重复数据")

    # 6. 验证修复结果
    print("\n✅ 验证修复结果:")
    remaining_duplicates = df_fixed['timestamp'].duplicated().sum()
    print(f"剩余重复时间戳: {remaining_duplicates}")

    # 7. 保存修复后的数据
    print(f"\n💾 保存修复后的数据到: {output_path}")
    df_fixed.to_csv(output_path, index=False)

    # 8. 验证文件
    print("\n🔍 验证保存的文件...")
    df_loaded = pd.read_csv(output_path)
    print(f"加载的数据形状: {df_loaded.shape}")

    # 显示前几行
    print("\n前5行数据:")
    print(df_loaded.head())

    # 检查时间序列连续性
    times = pd.to_datetime(df_loaded['timestamp'])
    time_diffs = times.diff().dropna()

    print(f"\n📈 时间间隔统计:")
    print(f"  最小间隔: {time_diffs.min()}")
    print(f"  最大间隔: {time_diffs.max()}")
    print(f"  平均间隔: {time_diffs.mean()}")

    # 检查是否都是1小时间隔
    one_hour = pd.Timedelta(hours=1)
    is_hourly = (time_diffs == one_hour).all()

    if is_hourly:
        print(f"  ✅ 数据是连续的小时间序列")
    else:
        print(f"  ⚠ 数据不是严格的小时间序列")
        irregular = (time_diffs != one_hour).sum()
        print(f"    不规则间隔数量: {irregular}")

    return df_fixed


# 运行修复函数
fixed_df = fix_duplicated_reconstruction_data()