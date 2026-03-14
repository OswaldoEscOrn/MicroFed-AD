# ====================== mixed_data_preparation.py ======================
"""
混合数据预处理脚本 - 修改版：只保留标准化后的_scaled列
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

# ==================== 修改配置 ====================
# 输入：混合数据CSV文件
MIXED_DATA_PATH = r"D:\Oswaldo's surf project\70% Virtual 30% Real Database_Hourly\mixed_dataset_hourly_35040.csv"

# 输出目录：为混合数据创建单独的预处理目录
OUTPUT_DIR = r"D:\Oswaldo's surf project\70% Virtual 30% Real Database_Hourly\preprocessed_data_mixed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 窗口参数（与原脚本保持一致）
WINDOW_SIZE = 24  # 24小时 = 1天
STRIDE = 1  # 滑动步长
PREDICTION_HORIZON = 0  # 异常检测设为0

# ==================== 特征列配置 ====================
# 混合数据中的原始列名（未标准化）
ORIGINAL_FEATURES = [
    'avg_PM2.5',
    'total_noise_duration',
    'noise_event_count',
    'avg_salience'
]

# 标准化后的列名（必须带_scaled后缀）
SCALED_FEATURES = [
    'avg_PM2.5_scaled',
    'total_noise_duration_scaled',
    'noise_event_count_scaled',
    'avg_salience_scaled'
]


# ==================== 主函数 ====================
def main():
    print("=" * 80)
    print("混合数据预处理脚本（只保留_scaled列）")
    print("=" * 80)

    # 1. 加载混合数据
    print("\n1. 加载混合数据...")
    df = pd.read_csv(MIXED_DATA_PATH, parse_dates=['timestamp'])

    # 设置时间戳为索引
    df = df.set_index('timestamp')
    df = df.sort_index()

    print(f"数据形状: {df.shape}")
    print(f"日期范围: {df.index.min()} → {df.index.max()}")
    print("所有列名:", df.columns.tolist())

    # 检查原始特征是否存在
    available_original_features = []
    missing_features = []

    for feat in ORIGINAL_FEATURES:
        if feat in df.columns:
            available_original_features.append(feat)
        else:
            missing_features.append(feat)
            print(f"⚠️ 警告: 特征 {feat} 不在数据中")

    if len(missing_features) > 0:
        print(f"\n🔍 查找可能的替代列名:")
        for missing_feat in missing_features:
            possible_matches = [col for col in df.columns if missing_feat in col]
            if possible_matches:
                print(f"  {missing_feat} → 可能匹配: {possible_matches}")
                # 使用第一个匹配项
                if possible_matches[0] not in available_original_features:
                    available_original_features.append(possible_matches[0])
            else:
                print(f"  {missing_feat} → 没有找到匹配项")

    print(f"\n可用的原始特征: {available_original_features}")

    if len(available_original_features) < 2:
        print("❌ 错误: 可用的特征太少，请检查数据列名")
        return

    # 2. 检查缺失值
    print("\n2. 检查缺失值...")
    missing_counts = df[available_original_features].isna().sum()
    print(missing_counts)

    if missing_counts.sum() > 0:
        print(f"  填充 {missing_counts.sum()} 个缺失值")
        df_filled = df[available_original_features].fillna(0)
    else:
        df_filled = df[available_original_features]

    # 3. ==================== 应用标准化并创建_scaled列 ====================
    print("\n3. 对混合数据进行标准化（只创建_scaled列）...")

    scalers = {}
    normalized_df = pd.DataFrame(index=df_filled.index)

    # 为每个原始特征应用标准化，只创建_scaled列
    for i, original_col in enumerate(available_original_features):
        scaler = StandardScaler()
        original_vals = df_filled[[original_col]].values

        # 应用标准化
        scaled_vals = scaler.fit_transform(original_vals)

        # 使用预定义的_scaled列名
        if i < len(SCALED_FEATURES):
            scaled_col = SCALED_FEATURES[i]
        else:
            scaled_col = f"{original_col}_scaled"

        # 只添加标准化后的列（不保留原始列）
        normalized_df[scaled_col] = scaled_vals

        # 保存标准化器
        scalers[original_col] = scaler

        print(f"  → 标准化 {original_col} → {scaled_col}")
        print(f"     原始: mean={original_vals.mean():.4f}, std={original_vals.std():.4f}")
        print(f"     标准化后: mean={scaled_vals.mean():.6f}, std={scaled_vals.std():.6f}")

    # 获取最终的特征列名（只有_scaled列）
    scaled_columns = normalized_df.columns.tolist()
    print(f"\n标准化后的列名（只包含_scaled列）: {scaled_columns}")

    final_df = normalized_df[scaled_columns]  # 确保只保留_scaled列

    print("\n标准化后数据统计（只包含_scaled列）:")
    for col in scaled_columns:
        print(f"  {col}: mean={final_df[col].mean():.6f}, std={final_df[col].std():.6f}")

    # 4. 创建滑动窗口
    def create_sliding_windows(data, window_size, stride=1, horizon=0):
        """创建3D滑动窗口数组: (n_samples, window_size, n_features)"""
        X = []
        y = [] if horizon > 0 else None

        for i in range(0, len(data) - window_size - horizon + 1, stride):
            window = data[i: i + window_size]
            X.append(window)
            if horizon > 0:
                target = data[i + window_size: i + window_size + horizon]
                y.append(target)

        X = np.array(X)
        if y is not None:
            y = np.array(y)

        return X, y

    print(f"\n4. 创建滑动窗口 (window_size={WINDOW_SIZE}, stride={STRIDE})...")

    # 使用标准化后的特征值（只使用带_scaled后缀的列）
    data_array = final_df[scaled_columns].values

    X_windows, y_windows = create_sliding_windows(
        data_array,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        horizon=PREDICTION_HORIZON
    )

    print(f"窗口数据形状: {X_windows.shape}")  # (n_windows, timesteps, n_features)

    if y_windows is not None:
        print(f"目标数据形状: {y_windows.shape}")

    # 5. 保存处理后的数据
    print("\n5. 保存处理后的数据...")

    # 保存窗口数据
    np.save(os.path.join(OUTPUT_DIR, "X_windows.npy"), X_windows)
    print(f"  → {OUTPUT_DIR}/X_windows.npy")

    if y_windows is not None:
        np.save(os.path.join(OUTPUT_DIR, "y_windows.npy"), y_windows)
        print(f"  → {OUTPUT_DIR}/y_windows.npy")

    # 保存标准化后的小时数据（只包含_scaled列）
    final_df.to_csv(os.path.join(OUTPUT_DIR, "normalized_hourly_data.csv"))
    print(f"  → {OUTPUT_DIR}/normalized_hourly_data.csv")
    print(f"  文件包含的列: {final_df.columns.tolist()}")

    # 保存标准化器
    import joblib
    for col, scaler in scalers.items():
        joblib.dump(scaler, os.path.join(OUTPUT_DIR, f"scaler_{col}.pkl"))
        print(f"  → {OUTPUT_DIR}/scaler_{col}.pkl")

    # 保存所有标准化器的集合
    joblib.dump(scalers, os.path.join(OUTPUT_DIR, "all_scalers.pkl"))
    print(f"  → {OUTPUT_DIR}/all_scalers.pkl")

    # 6. 数据统计信息
    print("\n" + "=" * 80)
    print("数据预处理完成！")
    print("=" * 80)

    n_windows = X_windows.shape[0]
    n_hourly = final_df.shape[0]

    print(f"\n📊 数据统计:")
    print(f"  原始小时数据: {n_hourly:,} 小时")
    print(f"  创建的窗口数: {n_windows:,} 个24小时窗口")
    print(f"  窗口形状: {X_windows.shape}")
    print(f"  特征数量: {X_windows.shape[2]}")
    print(f"  使用的特征列（只包含_scaled列）: {scaled_columns}")

    # 窗口数据统计
    print(f"\n📊 窗口数据统计:")
    print(f"  最小值: {X_windows.min():.6f}")
    print(f"  最大值: {X_windows.max():.6f}")
    print(f"  均值: {X_windows.mean():.6f}")
    print(f"  标准差: {X_windows.std():.6f}")

    # 创建元数据文件
    metadata = {
        "source_data": "mixed_dataset_hourly_35040.csv",
        "preprocessing_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "hourly_samples": int(n_hourly),
        "window_samples": int(n_windows),
        "window_shape": list(X_windows.shape),
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "original_features": available_original_features,
        "scaled_features": scaled_columns,
        "standardized": True,
        "scaler_info": {
            col: {
                "mean": float(scaler.mean_[0]),
                "scale": float(scaler.scale_[0]),
                "var": float(scaler.var_[0])
            } for col, scaler in scalers.items()
        },
        "data_statistics": {
            "min": float(X_windows.min()),
            "max": float(X_windows.max()),
            "mean": float(X_windows.mean()),
            "std": float(X_windows.std())
        },
        "output_files": [
                            "X_windows.npy",
                            "normalized_hourly_data.csv",
                            "all_scalers.pkl"
                        ] + [f"scaler_{col}.pkl" for col in scalers.keys()],
        "note": "normalized_hourly_data.csv只包含标准化后的_scaled列，不包含原始特征列"
    }

    import json
    metadata_path = os.path.join(OUTPUT_DIR, "preprocessing_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n💾 元数据文件: {metadata_path}")

    # 显示CSV文件内容预览
    print("\n📋 标准化后CSV文件前5行预览（只包含_scaled列）:")
    print(final_df.head().to_string())

    print("\n📋 CSV文件列名:")
    for i, col in enumerate(final_df.columns, 1):
        print(f"  {i}. {col}")

    print("\n✅ 预处理完成！")
    print(f"📍 文件保存在: {OUTPUT_DIR}")

    


# ==================== 运行脚本 ====================
if __name__ == "__main__":
    main()