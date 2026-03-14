import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


def environmental_augmentation_validation(original_df, augmented_df):
    """
    针对环境监测数据的增强质量验证
    特征：avg_PM2.5_normalized, total_noise_duration_scaled,
          noise_event_count_scaled, avg_salience_scaled, avg_PM2.5
    """

    print("=" * 60)
    print("环境监测数据增强质量验证报告")
    print("=" * 60)

    results = {}

    # 1. 基础特征列表
    features = ['avg_PM2.5_scaled', 'total_noise_duration_scaled',
                'noise_event_count_scaled', 'avg_salience_scaled']

    scaled_features = ['avg_PM2.5_scaled', 'total_noise_duration_scaled',
                       'noise_event_count_scaled', 'avg_salience_scaled']

    # 2. 物理约束检查（最重要！）
    print("\n1. 物理约束检查:")

    # PM2.5不能为负
    # pm25_negative = augmented_df['avg_PM2.5'] < 0
    # if pm25_negative.any():
    #     print(f"  ⚠ 警告: {pm25_negative.sum()} 个PM2.5负值记录")
    # else:
    #     print(f"  ✓ PM2.5数值全部非负")

    # 标准化特征应在合理范围
    for feat in scaled_features:
        outliers = augmented_df[feat].abs() > 5  # 标准化后通常应在[-3,3]
        if outliers.any():
            print(f"  ⚠ {feat}: {outliers.sum()} 个值超出±5范围")

    # 3. 分布特性验证（针对不同类型特征）
    print("\n2. 分布特性验证:")

    # PM2.5通常服从对数正态分布
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        # 原始数据分布
        axes[i].hist(original_df[feat].dropna(), bins=50, alpha=0.5,
                     label='original', density=True, color='blue')
        # 增强数据分布
        axes[i].hist(augmented_df[feat].dropna(), bins=50, alpha=0.5,
                     label='augmented', density=True, color='red')

        # 计算统计检验
        ks_stat, ks_p = stats.ks_2samp(
            original_df[feat].dropna(),
            augmented_df[feat].dropna()
        )

        # 检查偏度和峰度
        orig_skew = stats.skew(original_df[feat].dropna())
        aug_skew = stats.skew(augmented_df[feat].dropna())
        skew_diff = abs(orig_skew - aug_skew)

        axes[i].set_title(f'{feat}\nKS-p={ks_p:.3f}, skew={skew_diff:.2f}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(r"D:\Oswaldo's surf project\Mixed_Real_Virtual_Database_Hourly\100k_environmental_dist_comparison.png", dpi=150)
    plt.close()

    # 4. 相关性保持度验证（环境指标间通常有相关性）
    print("\n3. 特征间相关性验证:")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 原始数据相关性
    orig_corr = original_df[features].corr()
    sns.heatmap(orig_corr, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, ax=ax1)
    ax1.set_title('Original Data Relevance')

    # 增强数据相关性
    aug_corr = augmented_df[features].corr()
    sns.heatmap(aug_corr, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, ax=ax2)
    ax2.set_title('Augmented Data Relevance')

    plt.tight_layout()
    plt.savefig(r"D:\Oswaldo's surf project\Mixed_Real_Virtual_Database_Hourly\100k_augmented_data_relevance",dpi=150)
    plt.close()

    # 计算相关性差异
    corr_diff = (orig_corr - aug_corr).abs()
    max_diff = corr_diff.max().max()
    mean_diff = corr_diff.mean().mean()

    print(f"  最大相关性差异: {max_diff:.3f}")
    print(f"  平均相关性差异: {mean_diff:.3f}")

    if mean_diff < 0.1:
        print(f"  ✓ 相关性结构保持良好")
    else:
        print(f"  ⚠ 警告: 变量间关系有显著改变")

    # 5. 时间序列模式检查（如果有时序信息）
    print("\n4. 时序模式检查（如果适用）:")


    # 6. 多变量异常检测（检测增强数据中的异常模式）
    print("\n5. 多变量模式一致性检查:")

    # 使用原始数据训练孤立森林
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    iso_forest.fit(original_df[features])

    # 预测增强数据中的异常
    aug_predictions = iso_forest.predict(augmented_df[features])
    anomaly_rate = (aug_predictions == -1).mean()

    print(f"  增强数据中，被原始数据模型标记为异常的比例: {anomaly_rate:.2%}")

    if anomaly_rate < 0.1:
        print(f"  ✓ 增强数据与原始数据模式一致")
    elif anomaly_rate < 0.2:
        print(f"  ⚠ 注意: 有{anomaly_rate:.1%}的增强数据与原始模式不同")
    else:
        print(f"  ⚠ 警告: 大量增强数据({anomaly_rate:.1%})与原始模式不一致")

    # 7. 聚类结构保持度
    print("\n6. 聚类结构检查:")

    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    # 对原始数据聚类
    kmeans_orig = KMeans(n_clusters=3, random_state=42)
    orig_labels = kmeans_orig.fit_predict(original_df[features].fillna(0))

    # 对增强数据使用相同聚类中心
    aug_labels = kmeans_orig.predict(augmented_df[features].fillna(0))

    # 为了比较，也单独对增强数据聚类
    kmeans_aug = KMeans(n_clusters=3, random_state=42)
    aug_labels_separate = kmeans_aug.fit_predict(augmented_df[features].fillna(0))

    # 比较聚类中心
    center_diff = np.abs(kmeans_orig.cluster_centers_ - kmeans_aug.cluster_centers_).mean()
    print(f"  聚类中心平均差异: {center_diff:.3f}")

    # 8. 生成综合质量报告
    print("\n" + "=" * 60)
    print("综合质量评分")
    print("=" * 60)

    # 计算各项得分
    scores = {
        # '物理约束': 1.0 if not pm25_negative.any() else 0.5,
        '分布一致性': 1.0 if mean_diff < 0.1 else 0.7 if mean_diff < 0.2 else 0.4,
        '相关性保持': 1.0 if mean_diff < 0.1 else 0.6,
        '异常率': 1.0 if anomaly_rate < 0.1 else 0.8 if anomaly_rate < 0.2 else 0.5,
        '聚类结构': 1.0 if center_diff < 0.2 else 0.7
    }

    overall_score = np.mean(list(scores.values()))

    print(f"\n各项得分:")
    for key, value in scores.items():
        print(f"  {key}: {value:.2f}")

    print(f"\n综合得分: {overall_score:.2f}/1.0")

    if overall_score >= 0.9:
        rating = "优秀 - 可直接使用"
    elif overall_score >= 0.7:
        rating = "良好 - 建议轻度检查"
    elif overall_score >= 0.5:
        rating = "中等 - 需要进一步验证"
    else:
        rating = "需改进 - 建议重新生成"

    print(f"评估: {rating}")

    # 保存详细结果
    results = {
        'features': features,
        # 'physical_constraints_violated': pm25_negative.sum(),
        'correlation_mean_diff': mean_diff,
        'anomaly_rate_in_augmented': anomaly_rate,
        'cluster_center_diff': center_diff,
        'scores': scores,
        'overall_score': overall_score,
        'rating': rating
    }

    return results


# 使用示例
# ====================== 主程序 ======================
def main():
    # 加载数据
    original_df = pd.read_csv(r"D:\Oswaldo's surf project\Mixed_Real_Virtual_Database_Hourly\preprocessed_data_mixed\normalized_hourly_data.csv")
    augmented_df = pd.read_csv(r"D:\Oswaldo's surf project\Mixed_Real_Virtual_Database_Hourly\preprocessed_data_mixed\normalized_hourly_data_augmented_100k.csv")

    print("📊 原始数据列:", original_df.columns.tolist())
    print("📊 增强数据列:", augmented_df.columns.tolist())

    # 处理索引列
    if 'Unnamed: 0' in original_df.columns:
        print("🔄 删除原始数据的索引列...")
        original_df = original_df.drop(columns=['Unnamed: 0'])

    if 'Unnamed: 0' in augmented_df.columns:
        print("🔄 删除增强数据的索引列...")
        augmented_df = augmented_df.drop(columns=['Unnamed: 0'])

    # 验证特征列存在
    required_features = ['avg_PM2.5_scaled', 'total_noise_duration_scaled',
                         'noise_event_count_scaled', 'avg_salience_scaled']

    missing_original = [col for col in required_features if col not in original_df.columns]
    missing_augmented = [col for col in required_features if col not in augmented_df.columns]

    if missing_original:
        print(f"❌ 原始数据缺少列: {missing_original}")
        return

    if missing_augmented:
        print(f"❌ 增强数据缺少列: {missing_augmented}")
        return

    print("✅ 所有必需特征列都存在")
    print(f"📈 原始数据形状: {original_df.shape}")
    print(f"📈 增强数据形状: {augmented_df.shape}")

    # 运行验证
    results = environmental_augmentation_validation(original_df, augmented_df)

    # 保存结果到文件
    report_path = r"D:\Oswaldo's surf project\Mixed_Real_Virtual_Database_Hourly\100k_environmental_augmentation_validation_report.txt"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("环境监测数据增强验证报告\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"验证时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"原始数据样本数: {len(original_df)}\n")
        f.write(f"增强数据样本数: {len(augmented_df)}\n\n")

        for key, value in results.items():
            if key == 'features':
                f.write("验证的特征:\n")
                for feat in value:
                    f.write(f"  - {feat}\n")
            elif key == 'scores':
                f.write("\n各项得分:\n")
                for subkey, subvalue in value.items():
                    f.write(f"  {subkey}: {subvalue:.2f}\n")
            else:
                f.write(f"{key}: {value}\n")

    print(f"💾 详细报告已保存到: {report_path}")

    # 根据评分给出建议
    overall_score = results.get('overall_score', 0)
    print("\n" + "=" * 60)
    print("🎯 验证结果总结")
    print("=" * 60)

    if overall_score >= 0.9:
        print("✅ 优秀：数据增强质量非常高，可直接用于模型训练")
    elif overall_score >= 0.7:
        print("✅ 良好：数据增强质量良好，建议检查分布图后再使用")
    elif overall_score >= 0.5:
        print("⚠️ 中等：数据增强质量一般，建议重新生成或手动调整")
    else:
        print("❌ 需改进：数据增强质量较差，建议重新设计增强策略")


if __name__ == "__main__":
    main()