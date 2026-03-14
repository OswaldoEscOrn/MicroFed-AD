# ====================== 完整的数据集合并与保存流程（修正版） ======================
import json
import os
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

warnings.filterwarnings('ignore')


def load_real_data_hourly():
    """加载真实数据集（每小时格式）"""
    print("🔍 加载真实数据集（每小时格式）...")

    # 加载你提供的CSV文件
    real_data_path = r"D:\Oswaldo's surf project\DR O's database\preprocessed_data\normalized_hourly_data.csv"

    if not os.path.exists(real_data_path):
        print(f"❌ 真实数据文件不存在: {real_data_path}")
        return None

    # 读取CSV文件
    df_real = pd.read_csv(real_data_path, index_col=0, parse_dates=True)
    print(f"  真实数据形状: {df_real.shape}")
    print(f"  真实数据列名: {df_real.columns.tolist()}")

    # 提取前4列作为特征
    feature_columns = df_real.columns[:4].tolist()
    print(f"  使用的特征列（前4列）: {feature_columns}")

    # 提取特征数据
    X_real_hourly = df_real[feature_columns].values
    print(f"  特征数据形状: {X_real_hourly.shape}")

    return X_real_hourly, df_real, feature_columns


def load_virtual_data_24hour():
    """加载24小时窗口的虚拟数据集"""
    print("🔍 加载24小时窗口的虚拟数据集...")

    # 虚拟数据路径
    virtual_samples_path = r"D:\Oswaldo's surf project\My Database\Merge_and_DAE_4_extracted_features\virtual_samples_4features_35040.npy"

    if not os.path.exists(virtual_samples_path):
        print(f"❌ 虚拟数据文件不存在: {virtual_samples_path}")
        return None

    # 加载虚拟数据
    virtual_samples = np.load(virtual_samples_path)
    print(f"  虚拟数据原始形状: {virtual_samples.shape}")

    # 检查维度，确保是24小时窗口格式
    if len(virtual_samples.shape) == 3:
        # 已经是3D数据
        if virtual_samples.shape[1] == 24 and virtual_samples.shape[2] == 4:
            print(f"  虚拟数据已经是24小时窗口格式: {virtual_samples.shape}")
            return virtual_samples
        else:
            print(f"⚠️  虚拟数据维度不匹配期望的24小时窗口格式: {virtual_samples.shape}")
            print(f"  尝试调整形状...")
    elif len(virtual_samples.shape) == 2:
        # 如果是2D数据，尝试重塑为24小时窗口
        print(f"  虚拟数据是2D格式，尝试重塑为24小时窗口...")
        total_samples = virtual_samples.shape[0]

        # 检查是否可以重塑为 (n_samples, 24, 4)
        if total_samples % 24 == 0:
            n_samples = total_samples // 24
            virtual_samples = virtual_samples.reshape(n_samples, 24, 4)
            print(f"  成功重塑为: {virtual_samples.shape}")
        else:
            print(f"❌ 无法将虚拟数据重塑为24小时窗口: 总样本数 {total_samples} 不能被24整除")
            return None
    else:
        print(f"❌ 虚拟数据维度错误: {virtual_samples.shape}")
        return None

    return virtual_samples


def create_hourly_data_from_windows(window_data):
    """将窗口数据转换为小时数据"""
    print(f"  将窗口数据转换为小时数据...")

    if len(window_data.shape) == 3:
        # 窗口数据形状: (n_windows, 24, 4)
        n_windows = window_data.shape[0]
        n_hours = window_data.shape[1]
        n_features = window_data.shape[2]

        # 重塑为小时数据
        hourly_data = window_data.reshape(-1, n_features)
        print(f"  转换后小时数据形状: {hourly_data.shape}")
        return hourly_data
    else:
        print(f"❌ 窗口数据格式错误: {window_data.shape}")
        return None


def create_mixed_dataset_hourly(X_real_hourly, X_virtual_hourly, target_hours=35064, real_ratio=0.7):
    """
    创建混合数据集（每小时格式）- 关键修改：均匀混合
    """
    print("\n🎯 创建混合数据集 (7:3比例，每小时格式)...")

    # 检查输入数据
    if X_real_hourly is None or X_virtual_hourly is None:
        print("❌ 输入数据为空")
        return None, None

    print(f"  真实每小时数据形状: {X_real_hourly.shape}")
    print(f"  虚拟每小时数据形状: {X_virtual_hourly.shape}")

    # 计算所需的小时数量
    n_real_hours_needed = int(target_hours * real_ratio)
    n_virtual_hours_needed = target_hours - n_real_hours_needed

    print(f"  目标总小时数: {target_hours:,}")
    print(f"  所需真实小时数: {n_real_hours_needed:,} ({real_ratio * 100:.0f}%)")
    print(f"  所需虚拟小时数: {n_virtual_hours_needed:,} ({(1 - real_ratio) * 100:.0f}%)")

    # ==================== 关键修改：均匀混合 ====================
    print("\n🔀 创建均匀混合的数据集...")

    # 确保我们有足够的真实数据
    if X_real_hourly.shape[0] < n_real_hours_needed:
        print(f"⚠️  真实数据不足，使用所有真实数据")
        n_real_actual = X_real_hourly.shape[0]
        X_real_selected = X_real_hourly[:n_real_actual]  # 保持时间顺序
    else:
        n_real_actual = n_real_hours_needed
        # 从真实数据中均匀选择（保持时间顺序）
        step = max(1, X_real_hourly.shape[0] // n_real_actual)
        indices = np.arange(0, X_real_hourly.shape[0], step)[:n_real_actual]
        X_real_selected = X_real_hourly[indices]

    # 确保我们有足够的虚拟数据
    if X_virtual_hourly.shape[0] < n_virtual_hours_needed:
        print(f"⚠️  虚拟数据不足，使用所有虚拟数据")
        n_virtual_actual = X_virtual_hourly.shape[0]
        X_virtual_selected = X_virtual_hourly[:n_virtual_actual]
    else:
        n_virtual_actual = n_virtual_hours_needed
        # 从虚拟数据中均匀选择
        step = max(1, X_virtual_hourly.shape[0] // n_virtual_actual)
        indices = np.arange(0, X_virtual_hourly.shape[0], step)[:n_virtual_actual]
        X_virtual_selected = X_virtual_hourly[indices]

    # ==================== 均匀混合：交替插入 ====================
    print("\n🔀 交替插入真实和虚拟数据...")

    X_mixed = np.zeros((n_real_actual + n_virtual_actual, X_real_hourly.shape[1]))
    y_mixed = np.zeros(n_real_actual + n_virtual_actual)

    real_idx = 0
    virtual_idx = 0
    mixed_idx = 0

    # 交替插入真实和虚拟数据
    while real_idx < n_real_actual and virtual_idx < n_virtual_actual:
        # 插入真实数据
        X_mixed[mixed_idx] = X_real_selected[real_idx]
        y_mixed[mixed_idx] = 1
        real_idx += 1
        mixed_idx += 1

        # 插入虚拟数据
        X_mixed[mixed_idx] = X_virtual_selected[virtual_idx]
        y_mixed[mixed_idx] = 0
        virtual_idx += 1
        mixed_idx += 1

    # 如果还有剩余的真实数据
    while real_idx < n_real_actual:
        X_mixed[mixed_idx] = X_real_selected[real_idx]
        y_mixed[mixed_idx] = 1
        real_idx += 1
        mixed_idx += 1

    # 如果还有剩余的虚拟数据
    while virtual_idx < n_virtual_actual:
        X_mixed[mixed_idx] = X_virtual_selected[virtual_idx]
        y_mixed[mixed_idx] = 0
        virtual_idx += 1
        mixed_idx += 1

    print(f"✅ 混合数据集创建完成:")
    print(f"  实际总小时数: {len(X_mixed):,}")
    print(f"  实际真实小时数: {np.sum(y_mixed):,} ({np.sum(y_mixed) / len(y_mixed) * 100:.1f}%)")
    print(f"  实际虚拟小时数: {len(y_mixed) - np.sum(y_mixed):,} ({100 - np.sum(y_mixed) / len(y_mixed) * 100:.1f}%)")
    print(f"  混合后数据形状: {X_mixed.shape}")

    return X_mixed, y_mixed


def save_mixed_dataset_hourly(X_mixed, y_mixed, feature_names):
    """保存混合数据集（每小时格式）"""
    print("\n💾 保存混合数据集（每小时格式）...")

    # 创建保存目录
    save_dir = r"D:\Oswaldo's surf project\Mixed_Real_Virtual_Database_Hourly"
    os.makedirs(save_dir, exist_ok=True)

    # 保存NumPy格式（每小时）
    X_save_path = os.path.join(save_dir, "X_mixed_hourly_35040.npy")
    y_save_path = os.path.join(save_dir, "y_mixed_labels_hourly_35040.npy")

    np.save(X_save_path, X_mixed)
    np.save(y_save_path, y_mixed)

    print(f"  NumPy文件保存完成:")
    print(f"    X_mixed_hourly_35040.npy: {X_mixed.shape}")
    print(f"    y_mixed_labels_hourly_35040.npy: {y_mixed.shape}")

    # ==================== 生成CSV文件（35040行） ====================
    print("\n📊 创建并保存CSV文件（35064行，每小时格式）...")

    n_samples, n_features = X_mixed.shape

    print(f"  数据形状: {X_mixed.shape}")
    print(f"  每行代表一小时，总共35040行")

    # 简化特征列名
    clean_feature_names = []
    for name in feature_names:
        clean_name = name.replace('_normalized_scaled', '').replace('_scaled', '')
        clean_feature_names.append(clean_name)

    print(f"  简化后的特征列名: {clean_feature_names}")

    # 生成时间戳（从2013/3/1 1:00开始，每小时一个）
    print("\n📅 生成时间戳...")
    start_date = datetime(2013, 3, 1, 1, 0, 0)
    timestamps = [start_date + timedelta(hours=i) for i in range(n_samples)]

    print(f"  第一个时间戳: {timestamps[0]}")
    print(f"  最后一个时间戳: {timestamps[-1]}")
    print(f"  总共小时数: {len(timestamps)}")

    # 创建DataFrame
    df_hourly = pd.DataFrame({
        'timestamp': timestamps
    })

    # 添加特征列
    for i in range(n_features):
        df_hourly[clean_feature_names[i]] = X_mixed[:, i]

    # 保存CSV文件
    csv_path = os.path.join(save_dir, "mixed_dataset_hourly_35040.csv")
    df_hourly.to_csv(csv_path, index=False, encoding='utf-8-sig')

    print(f"✅ CSV文件保存完成: {csv_path}")
    print(f"  文件大小: {os.path.getsize(csv_path) / (1024 * 1024):.2f} MB")
    print(f"  行数: {len(df_hourly):,}")
    print(f"  列数: {len(df_hourly.columns)}")
    print(f"  列名: {list(df_hourly.columns)}")

    # 显示前几行数据
    print(f"\n📋 CSV文件前10行预览:")
    print(df_hourly.head(10).to_string(index=False))

    # 显示混合情况（检查交替模式）
    print(f"\n🔍 混合情况检查（前20个时间点的标签）:")
    print(f"  标签序列 (1=真实, 0=虚拟): {y_mixed[:20]}")
    print(f"  真实数据比例: {np.sum(y_mixed) / len(y_mixed) * 100:.1f}%")

    # 保存元数据
    metadata = {
        "dataset_name": "Mixed_Real_Virtual_Dataset_Hourly",
        "total_hours": int(n_samples),
        "real_hours": int(np.sum(y_mixed)),
        "virtual_hours": int(len(y_mixed) - np.sum(y_mixed)),
        "real_ratio": float(np.sum(y_mixed) / len(y_mixed)),
        "virtual_ratio": float(1 - np.sum(y_mixed) / len(y_mixed)),
        "data_shape": list(X_mixed.shape),
        "feature_names": clean_feature_names,
        "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "description": "混合数据集: 70%真实数据 + 30%虚拟数据，交替均匀混合",
        "timestamp_start": timestamps[0].strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp_end": timestamps[-1].strftime("%Y-%m-%d %H:%M:%S"),
        "mixing_method": "交替混合（真实-虚拟-真实-虚拟...）",
        "note": "CSV文件包含35064行，每行代表一小时，特征交替混合"
    }

    metadata_path = os.path.join(save_dir, "mixed_dataset_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"  元数据文件保存完成: {metadata_path}")

    return save_dir, csv_path


def main():
    print("=" * 80)
    print("真实与虚拟数据混合工具（解决异常率高问题）")
    print("目标: 创建35064行混合数据集，均匀混合真实和虚拟数据")
    print("=" * 80)

    # 1. 加载真实数据（每小时格式）
    print("\n1. 加载真实数据集...")
    result = load_real_data_hourly()
    if result is None:
        return

    X_real_hourly, df_real, real_feature_names = result
    print(f"   真实数据已有小时数: {X_real_hourly.shape[0]:,}")

    # 2. 加载虚拟数据（24小时窗口格式）
    print("\n2. 加载虚拟数据集...")
    X_virtual_24h = load_virtual_data_24hour()
    if X_virtual_24h is None:
        print("❌ 无法加载虚拟数据，退出")
        return

    # 3. 将虚拟数据转换为每小时格式
    print("\n3. 转换虚拟数据格式...")
    X_virtual_hourly = create_hourly_data_from_windows(X_virtual_24h)
    if X_virtual_hourly is None:
        print("❌ 无法转换虚拟数据格式")
        return

    print(f"   虚拟数据转换后小时数: {X_virtual_hourly.shape[0]:,}")

    # 4. 创建混合数据集（均匀混合）
    print("\n4. 创建混合数据集（均匀混合）...")
    X_mixed, y_mixed = create_mixed_dataset_hourly(
        X_real_hourly,
        X_virtual_hourly,
        target_hours=35064,
        real_ratio=0.7
    )

    if X_mixed is None:
        print("❌ 创建混合数据集失败")
        return

    # 5. 保存数据集
    save_dir, csv_path = save_mixed_dataset_hourly(X_mixed, y_mixed, real_feature_names)

    # 6. 数据集统计
    print("\n📊 数据集统计信息:")
    print(f"  总小时数: {len(X_mixed):,}")
    print(f"  真实小时数: {np.sum(y_mixed):,} ({np.sum(y_mixed) / len(y_mixed) * 100:.1f}%)")
    print(f"  虚拟小时数: {len(y_mixed) - np.sum(y_mixed):,} ({100 - np.sum(y_mixed) / len(y_mixed) * 100:.1f}%)")
    print(f"  数据形状: {X_mixed.shape}")
    print(f"  混合方式: 交替混合（真实-虚拟-真实-虚拟...）")

    print(f"\n✅ 数据集创建完成！")
    print(f"🎯 目标: 35064行CSV文件已生成")
    print(f"📍 所有文件保存到: {save_dir}")
    print(f"\n📝 下一步:")
    print(f"  1. 运行第二个代码 (data_preparation.py)")
    print(f"  2. 它会将35064小时数据转换为窗口数据")
    print(f"  3. 修改第三个代码的路径，指向新的预处理数据")


if __name__ == "__main__":
    main()