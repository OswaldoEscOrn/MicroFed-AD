import json
import os
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings('ignore')


# ====================== 1. 加载已生成的特征文件 ======================
def load_extracted_features():
    """
    加载之前生成的特征文件
    """
    # 定义文件路径
    hour_matrix_path = r"D:\Oswaldo's surf project\My Database\Merge_and_DAE24hour_pattern_matrix_scaled.npy"
    global_vector_path = r"D:\Oswaldo's surf project\My Database\Merge_and_DAEglobal_period_pattern_vector_scaled.npy"
    feature_mapping_path = r"D:\Oswaldo's surf project\My Database\Merge_and_DAEfeature_mapping.json"

    # 检查文件是否存在
    for path in [hour_matrix_path, global_vector_path, feature_mapping_path]:
        if not os.path.exists(path):
            print(f"❌ File does not exist: {path}")
            return None, None, None

    # Load files
    try:
        hour_matrix = np.load(hour_matrix_path)
        print(f"✅ 24-hour matrix loaded successfully, shape: {hour_matrix.shape}")

        global_vector = np.load(global_vector_path)
        print(f"✅ Global vector loaded successfully, shape: {global_vector.shape}")

        with open(feature_mapping_path, 'r', encoding='utf-8') as f:
            feature_mapping = json.load(f)
        print(f"✅ Feature mapping loaded successfully, contains {len(feature_mapping)} categories")

    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None, None, None

    return hour_matrix, global_vector, feature_mapping


# ====================== 2. 改进的虚拟样本生成函数 ======================
def generate_virtual_samples(hour_matrix, global_vector, n_samples=35040, noise_strategy='adaptive'):
    """
    基于原始模式生成虚拟样本 - 改进版本，生成更多样化的样本

    参数:
    n_samples: 生成样本数量，默认35040与PDF中一致
    noise_strategy: 噪声策略 - 'adaptive'(自适应), 'time_varying'(时间变化), 'feature_dependent'(特征相关)
    """
    print(f"\n🎲 Generating {n_samples:,} virtual samples...")

    # 确保输入是正确形状
    if len(hour_matrix.shape) == 2:  # 形状为(24, features)
        original_hour_matrix = hour_matrix
        n_hours, hour_features = hour_matrix.shape
    elif len(hour_matrix.shape) == 1:  # 展平的一维数组
        # 假设是24*features的结构
        total_features = hour_matrix.shape[0]
        if total_features % 24 == 0:
            hour_features = total_features // 24
            original_hour_matrix = hour_matrix.reshape(24, hour_features)
            n_hours = 24
        else:
            raise ValueError(
                f"Cannot infer hour feature dimension: total features {total_features} is not a multiple of 24")
    else:
            raise ValueError(f"Unsupported hour matrix shape: {hour_matrix.shape}")

    # 处理全局向量
    if len(global_vector.shape) == 1:
        global_features = global_vector.shape[0]
        original_global = global_vector.reshape(1, -1)
    else:
        global_features = global_vector.shape[1]
        original_global = global_vector

    print(f"  Hour matrix shape: ({n_hours}, {hour_features})")
    print(f"  Global vector shape: (1, {global_features})")

    # 计算原始数据的统计信息
    hour_mean = original_hour_matrix.mean(axis=0, keepdims=True)
    hour_std = original_hour_matrix.std(axis=0, keepdims=True) + 1e-8  # 防止除零

    # 生成虚拟样本
    virtual_hour_matrices = []
    virtual_global_vectors = []

    # 为每个小时创建不同的噪声模式
    hour_noise_factors = np.random.uniform(0.5, 2.0, size=n_hours)

    for i in range(n_samples):
        if noise_strategy == 'adaptive':
            # 自适应噪声：基于特征的标准差
            base_noise = np.random.normal(0, 1, original_hour_matrix.shape)
            hour_noise = base_noise * hour_std * 0.3  # 30%的变异

            # 添加时间相关的噪声
            time_noise = np.random.normal(0, 0.2, original_hour_matrix.shape)
            time_noise = time_noise * hour_noise_factors.reshape(-1, 1)

            # 添加特征相关的噪声
            feature_correlation = 0.7  # 特征间相关性
            correlated_noise = base_noise[:, :1] * feature_correlation + base_noise * (1 - feature_correlation)
            feature_noise = correlated_noise * hour_std * 0.2

            # 组合噪声
            total_noise = hour_noise + time_noise + feature_noise

        elif noise_strategy == 'time_varying':
            # 时间变化噪声：不同小时不同噪声水平
            time_of_day_factor = np.sin(np.linspace(0, 2 * np.pi, n_hours)).reshape(-1, 1) * 0.5 + 1
            base_noise = np.random.normal(0, 1, original_hour_matrix.shape)
            total_noise = base_noise * hour_std * 0.25 * time_of_day_factor

        else:  # 'feature_dependent'
            # 特征相关噪声：基于特征重要性
            feature_importance = np.abs(hour_mean.ravel()) / np.abs(hour_mean.ravel()).sum()
            feature_importance = feature_importance.reshape(1, -1).repeat(n_hours, axis=0)

            base_noise = np.random.normal(0, 1, original_hour_matrix.shape)
            total_noise = base_noise * hour_std * 0.3 * feature_importance

        # 生成虚拟小时矩阵
        virtual_hour = original_hour_matrix + total_noise

        # 确保非负值（如果需要）
        if np.min(original_hour_matrix) >= 0:  # 如果原始数据是非负的
            virtual_hour = np.maximum(virtual_hour, 0)

        # 生成虚拟全局向量
        global_mean = original_global.mean()
        global_std = original_global.std() + 1e-8

        global_noise = np.random.normal(0, global_std * 0.2, original_global.shape)
        virtual_global = original_global + global_noise

        # 如果原始全局数据是非负的，确保非负
        if np.min(original_global) >= 0:
            virtual_global = np.maximum(virtual_global, 0)

        virtual_hour_matrices.append(virtual_hour)
        virtual_global_vectors.append(virtual_global)

        # 进度显示
        if (i + 1) % 5000 == 0:
            print(f"    Generated {i + 1:,} samples...")

    # 转换为numpy数组
    virtual_hour_matrices = np.array(virtual_hour_matrices)
    virtual_global_vectors = np.array(virtual_global_vectors)

    print(f"✅ Virtual sample generation completed:")
    print(f"   Hour matrix shape: {virtual_hour_matrices.shape}")
    print(f"   Global vector shape: {virtual_global_vectors.shape}")

    # 计算样本多样性
    hour_diversity = virtual_hour_matrices.std(axis=0).mean()
    global_diversity = virtual_global_vectors.std(axis=0).mean()
    print(f"    Sample diversity metric - Hour features: {hour_diversity:.4f}, Global features: {global_diversity:.4f}")

    return virtual_hour_matrices, virtual_global_vectors


# ====================== 3. 数据准备函数 ======================
def prepare_dae_input(hour_matrices, global_vectors, mode='combined'):
    """
    准备AutoEncoder输入数据
    mode: 'hour_only', 'global_only', 'combined'
    """
    n_samples = hour_matrices.shape[0]

    if mode == 'hour_only':
        # 只使用小时矩阵 (n_samples, 24, features) → 展平为 (n_samples, 24*features)
        X = hour_matrices.reshape(n_samples, -1)
        print(f"📊 Using hour matrix features: {X.shape}")

    elif mode == 'global_only':
        # 只使用全局+时段向量 (n_samples, features)
        X = global_vectors
        print(f"📊 Using global + temporal features: {X.shape}")

    elif mode == 'combined':
        # 合并所有特征 (n_samples, 24*features + global_features)
        hour_flat = hour_matrices.reshape(n_samples, -1)
        global_flat = global_vectors.reshape(n_samples, -1)
        X = np.concatenate([hour_flat, global_flat], axis=1)
        print(f"📊 Using combined features: {X.shape}")

    return X


# ====================== 4. Deep AutoEncoder构建 ======================
def build_deep_autoencoder(input_dim, encoding_dim=32):
    """
    构建Deep AutoEncoder模型
    """
    print(f"\n🔧 Building Deep AutoEncoder model:")
    print(f"  Input dimension: {input_dim}")
    print(f"  Encoding dimension: {encoding_dim}")

    # ====== 编码器 ======
    encoder = Sequential([
        Input(shape=(input_dim,), name='input'),
        Dense(256, activation='relu', name='encoder_dense1'),
        BatchNormalization(name='encoder_bn1'),
        Dropout(0.3, name='encoder_dropout1'),

        Dense(128, activation='relu', name='encoder_dense2'),
        BatchNormalization(name='encoder_bn2'),
        Dropout(0.3, name='encoder_dropout2'),

        Dense(64, activation='relu', name='encoder_dense3'),
        BatchNormalization(name='encoder_bn3'),
        Dropout(0.2, name='encoder_dropout3'),

        Dense(encoding_dim, activation='relu', name='bottleneck')
    ], name='encoder')

    # ====== 解码器 ======
    decoder = Sequential([
        Input(shape=(encoding_dim,), name='decoder_input'),
        Dense(64, activation='relu', name='decoder_dense1'),
        BatchNormalization(name='decoder_bn1'),
        Dropout(0.2, name='decoder_dropout1'),

        Dense(128, activation='relu', name='decoder_dense2'),
        BatchNormalization(name='decoder_bn2'),
        Dropout(0.3, name='decoder_dropout2'),

        Dense(256, activation='relu', name='decoder_dense3'),
        BatchNormalization(name='decoder_bn3'),
        Dropout(0.3, name='decoder_dropout3'),

        Dense(input_dim, activation='linear', name='output')
    ], name='decoder')

    # ====== 完整AutoEncoder ======
    input_layer = Input(shape=(input_dim,), name='autoencoder_input')
    encoded = encoder(input_layer)
    decoded = decoder(encoded)
    autoencoder = Model(input_layer, decoded, name='autoencoder')

    return autoencoder, encoder, decoder


# ====================== 5. 模型训练 ======================
def train_autoencoder(X_train, X_val, encoding_dim=32, epochs=80):
    """
    训练AutoEncoder
    """
    input_dim = X_train.shape[1]

    # 构建模型
    autoencoder, encoder, decoder = build_deep_autoencoder(input_dim, encoding_dim)

    # 编译模型
    autoencoder.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    print("\n📋 Model Architecture Summary:")
    autoencoder.summary()

    # 计算参数数量
    total_params = autoencoder.count_params()
    trainable_params = np.sum([np.prod(v.shape) for v in autoencoder.trainable_weights])
    non_trainable_params = total_params - trainable_params

    print(f"\n📊 Model Parameter Statistics:")
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  Non-trainable params: {non_trainable_params:,}")
    print(f"  Memory usage: {total_params * 4 / 1024:.2f} KB")

    # 回调函数
    callbacks = [
        # EarlyStopping(
        #     monitor='val_loss',
        #     patience=12,
        #     restore_best_weights=True,
        #     verbose=1
        # ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # 训练模型
    print("\n🚀 start training Deep AutoEncoder...")
    print(f"  Training set size: {X_train.shape}")
    print(f"  Validation set size: {X_val.shape}")
    print(f"  Batch size: 128")
    print(f"  Max epochs: {epochs}")

    history = autoencoder.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=epochs,
        batch_size=128,
        # callbacks=callbacks,
        verbose=1
    )

    return autoencoder, encoder, decoder, history, total_params


# ====================== 6. 评估和异常检测 ======================
# ====================== 6. 评估和异常检测 ======================
def evaluate_and_detect_anomalies(autoencoder, X_val, X_test=None, percentile=95):
    """
    评估模型并检测异常
    """
    print("\n📈 Model Evaluation Results:")

    # Evaluate on validation set
    val_loss, val_mae = autoencoder.evaluate(X_val, X_val, verbose=0)
    print(f"  Validation loss (MSE): {val_loss:.6f}")
    print(f"  Validation MAE: {val_mae:.6f}")

    # Calculate validation set reconstruction errors
    val_reconstructed = autoencoder.predict(X_val, verbose=0)
    val_errors = np.mean((X_val - val_reconstructed) ** 2, axis=1)

    # Calculate threshold (95th percentile)
    threshold = np.percentile(val_errors, percentile)
    print(f"  Validation set reconstruction error {percentile}th percentile threshold: {threshold:.6f}")

    # 如果传入了测试集，在测试集上检测异常
    if X_test is not None:
        test_reconstructed = autoencoder.predict(X_test, verbose=0)
        test_errors = np.mean((X_test - test_reconstructed) ** 2, axis=1)

        # 检测异常
        anomalies = test_errors > threshold
        anomalies_count = np.sum(anomalies)
        anomaly_ratio = anomalies_count / len(X_test) * 100

        print(f"  Detected {anomalies_count} anomalous samples out of {len(X_test)} total samples")
        print(f"  Anomaly ratio: {anomaly_ratio:.2f}%")

        return threshold, anomalies_count, anomaly_ratio, test_errors, test_reconstructed
    else:
        # 只在验证集上检测异常
        anomalies = val_errors > threshold
        anomalies_count = np.sum(anomalies)
        anomaly_ratio = anomalies_count / len(X_val) * 100

        print(f"  Anomalies detected in validation set: {anomalies_count:,} / {len(X_val):,} ({anomaly_ratio:.2f}%)")

        return threshold, anomalies_count, anomaly_ratio, val_errors, val_reconstructed

    # 如果在测试集上检测异常
    anomalies_count = 0
    anomaly_ratio = 0

    # if X_test is not None:
    #     test_reconstructed = autoencoder.predict(X_test, verbose=0)
    #     test_errors = np.mean((X_test - test_reconstructed) ** 2, axis=1)
    #
    #     # 检测异常
    #     anomalies = test_errors > threshold
    #     anomalies_count = np.sum(anomalies)
    #     anomaly_ratio = anomalies_count / len(X_test) * 100
    #
    #     print(f"  检测到 {anomalies_count} 个异常样本，共 {len(X_test)} 个样本")
    #     print(f"  异常比例: {anomaly_ratio:.2f}%")
    #
    #     return threshold, anomalies_count, anomaly_ratio, test_errors, test_reconstructed
    # else:
    #     return threshold, 0, 0, val_errors, val_reconstructed


# ====================== 7. 可视化函数 ======================
def visualize_results(history, X_original, X_reconstructed, errors, threshold,
                      feature_mapping, model_name="Deep AutoEncoder"):
    """
    可视化训练结果和异常检测
    """
    print(f"\n📊 Generating visualization results...")

    # 设置绘图风格
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(20, 15))

    # 1. 训练损失变化
    ax1 = plt.subplot(3, 3, 1)
    epochs = range(1, len(history.history['loss']) + 1)
    ax1.plot(epochs, history.history['loss'], label='Training Loss', linewidth=2)
    ax1.plot(epochs, history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_title(f'{model_name} - Training History', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('MSE Loss', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. MAE变化
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(epochs, history.history['mae'], label='Training MAE', linewidth=2, color='orange')
    if 'val_mae' in history.history:
        ax2.plot(epochs, history.history['val_mae'], label='Validation MAE', linewidth=2, color='red')
    ax2.set_title('MAE History', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Mean Absolute Error', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 重建误差分布
    ax3 = plt.subplot(3, 3, 3)
    n_bins = min(100, len(errors) // 10)
    ax3.hist(errors, bins=n_bins, alpha=0.7, color='skyblue', edgecolor='black', density=True)
    ax3.axvline(x=threshold, color='red', linestyle='--', linewidth=2,
                label=f'Threshold ({threshold:.4f})')

    # 添加高斯分布拟合
    from scipy.stats import norm
    mu, std = norm.fit(errors)
    xmin, xmax = ax3.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    ax3.plot(x, p, 'k', linewidth=2, label=f'Normal fit: μ={mu:.4f}, σ={std:.4f}')

    ax3.set_title('Reconstruction Error Distribution', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Reconstruction Error (MSE)', fontsize=12)
    ax3.set_ylabel('Density', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 原始vs重建特征对比（随机选择5个样本）
    ax4 = plt.subplot(3, 3, 4)
    n_samples_to_show = min(5, len(X_original))

    for i in range(n_samples_to_show):
        sample_idx = np.random.randint(0, len(X_original))
        n_features = min(100, len(X_original[sample_idx]))

        x_indices = range(n_features)
        if i == 0:
            ax4.scatter(x_indices, X_original[sample_idx][:n_features],
                        alpha=0.5, label='Original', s=20)
            ax4.scatter(x_indices, X_reconstructed[sample_idx][:n_features],
                        alpha=0.5, label='Reconstructed', s=20, color='red')
        else:
            ax4.scatter(x_indices, X_original[sample_idx][:n_features],
                        alpha=0.3, s=10)
            ax4.scatter(x_indices, X_reconstructed[sample_idx][:n_features],
                        alpha=0.3, s=10, color='red')

    ax4.set_title(f'Original vs Reconstructed (5 Random Samples)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Feature Index', fontsize=12)
    ax4.set_ylabel('Feature Value', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. 重建误差时间序列
    ax5 = plt.subplot(3, 3, 5)
    sample_indices = range(len(errors))

    # 对误差进行平滑
    window_size = min(100, len(errors) // 10)
    if window_size > 1:
        errors_smooth = np.convolve(errors, np.ones(window_size) / window_size, mode='valid')
        indices_smooth = sample_indices[:len(errors_smooth)]
        ax5.plot(indices_smooth, errors_smooth, linewidth=1, alpha=0.7, color='blue', label='Smoothed Error')

    ax5.plot(sample_indices, errors, linewidth=0.5, alpha=0.3, color='gray', label='Raw Error')
    ax5.axhline(y=threshold, color='red', linestyle='--', linewidth=2,
                label=f'Threshold ({threshold:.4f})')

    # 标记异常
    anomalies = errors > threshold
    anomaly_indices = np.where(anomalies)[0]
    ax5.scatter(anomaly_indices, errors[anomaly_indices],
                color='red', s=10, alpha=0.5, label='Anomalies')

    ax5.set_title('Reconstruction Error Time Series', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Sample Index', fontsize=12)
    ax5.set_ylabel('Reconstruction Error', fontsize=12)
    ax5.legend(loc='upper right', fontsize=10)
    ax5.grid(True, alpha=0.3)

    # 6. 误差箱线图
    ax6 = plt.subplot(3, 3, 6)
    box_data = [errors[~anomalies], errors[anomalies]] if len(anomaly_indices) > 0 else [errors]
    box_labels = ['Normal', 'Anomaly'] if len(anomaly_indices) > 0 else ['All Samples']

    bp = ax6.boxplot(box_data, labels=box_labels, patch_artist=True)

    # 设置颜色
    colors = ['lightblue', 'lightcoral'] if len(anomaly_indices) > 0 else ['lightblue']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax6.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5)
    ax6.set_title('Error Distribution by Category', fontsize=14, fontweight='bold')
    ax6.set_ylabel('Reconstruction Error', fontsize=12)
    ax6.grid(True, alpha=0.3, axis='y')

    # 7. 学习率变化
    ax7 = plt.subplot(3, 3, 7)
    if 'lr' in history.history:
        ax7.plot(epochs, history.history['lr'], linewidth=2, color='green')
        ax7.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax7.set_xlabel('Epoch', fontsize=12)
        ax7.set_ylabel('Learning Rate', fontsize=12)
        ax7.grid(True, alpha=0.3)
    else:
        ax7.text(0.5, 0.5, 'Learning Rate\nData Not Available',
                 ha='center', va='center', transform=ax7.transAxes, fontsize=12)
        ax7.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')

    # 8. 特征重要性热图
    ax8 = plt.subplot(3, 3, 8)
    if len(X_original) > 0:
        feature_errors = np.mean((X_original - X_reconstructed) ** 2, axis=0)

        # 重塑为24小时模式（如果是小时特征）
        if 'hour_feature_names' in feature_mapping:
            hour_feature_count = feature_mapping.get('hour_feature_count', 11)
            if len(feature_errors) >= 24 * hour_feature_count:
                hour_errors = feature_errors[:24 * hour_feature_count].reshape(24, hour_feature_count)
                im = ax8.imshow(hour_errors, aspect='auto', cmap='YlOrRd')
                ax8.set_xlabel('Feature Index', fontsize=12)
                ax8.set_ylabel('Hour of Day', fontsize=12)
                ax8.set_title('Hourly Feature Reconstruction Error', fontsize=14, fontweight='bold')
                plt.colorbar(im, ax=ax8, label='Reconstruction Error')

                # 设置y轴标签
                ax8.set_yticks(range(0, 24, 3))
                ax8.set_yticklabels([f'{h:02d}:00' for h in range(0, 24, 3)])
        else:
            # 显示特征误差条形图
            n_top_features = min(20, len(feature_errors))
            top_indices = np.argsort(feature_errors)[-n_top_features:][::-1]
            top_errors = feature_errors[top_indices]

            ax8.barh(range(n_top_features), top_errors, color='skyblue')
            ax8.set_yticks(range(n_top_features))
            ax8.set_yticklabels([f'Feat_{i}' for i in top_indices])
            ax8.set_title(f'Top {n_top_features} Features by Reconstruction Error',
                          fontsize=14, fontweight='bold')
            ax8.set_xlabel('Average Reconstruction Error', fontsize=12)

    # 9. 异常检测性能
    ax9 = plt.subplot(3, 3, 9)
    if np.sum(errors > threshold) > 0:
        # 计算不同阈值下的性能
        thresholds = np.percentile(errors, range(90, 100))
        anomaly_rates = []

        for t in thresholds:
            anomaly_rate = np.sum(errors > t) / len(errors) * 100
            anomaly_rates.append(anomaly_rate)

        ax9.plot(thresholds, anomaly_rates, 'o-', linewidth=2, markersize=6)
        ax9.axvline(x=threshold, color='red', linestyle='--', linewidth=1.5,
                    label=f'95th percentile ({threshold:.4f})')
        ax9.set_title('Anomaly Rate vs Threshold', fontsize=14, fontweight='bold')
        ax9.set_xlabel('Threshold', fontsize=12)
        ax9.set_ylabel('Anomaly Rate (%)', fontsize=12)
        ax9.legend()
        ax9.grid(True, alpha=0.3)
    else:
        ax9.text(0.5, 0.5, 'No Anomalies Detected\nwith Current Threshold',
                 ha='center', va='center', transform=ax9.transAxes, fontsize=12)
        ax9.set_title('Anomaly Detection Performance', fontsize=14, fontweight='bold')

    plt.tight_layout()
    # 保存图表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"D:\\Oswaldo's surf project\\My Database\\visualization_{timestamp}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📸 Visualization results saved to: {save_path}")

    # plt.show()
    plt.close()

    # 输出特征级分析
    if 'hour_feature_names' in feature_mapping and len(feature_errors) > 0:
        print("\n📊 Hourly Feature Reconstruction Error Analysis (Top 5):")
        hour_feature_names = feature_mapping['hour_feature_names']

        # 计算每个小时的平均误差
        hour_errors_reshaped = feature_errors[:24 * len(hour_feature_names)].reshape(24, len(hour_feature_names))

        for hour in range(min(5, 24)):
            max_error_idx = np.argmax(hour_errors_reshaped[hour])
            max_error_feature = hour_feature_names[max_error_idx]
            max_error_value = hour_errors_reshaped[hour, max_error_idx]

            min_error_idx = np.argmin(hour_errors_reshaped[hour])
            min_error_feature = hour_feature_names[min_error_idx]
            min_error_value = hour_errors_reshaped[hour, min_error_idx]

            print(f"  Hour {hour:02d}:00 - Max error: {max_error_feature} ({max_error_value:.4f}), "
                  f"Min error: {min_error_feature} ({min_error_value:.4f})")


# ====================== 8. 主流程 ======================
def main():
    # 配置路径
    base_path = r"D:\Oswaldo's surf project\My Database\\"
    model_save_path = base_path + "models\\"

    print("=" * 80)
    print("DEEP AUTOENCODER TRAINING PIPELINE")
    print("=" * 80)

    # 1. 加载已提取的特征
    hour_matrix, global_vector, feature_mapping = load_extracted_features()
    if hour_matrix is None:
        print("❌ Failed to load feature files, exiting process")
        return

    # 2. 生成大量虚拟样本 (与PDF中的35040个样本一致)
    n_virtual_samples = 100000
    virtual_hour_matrices, virtual_global_vectors = generate_virtual_samples(
        hour_matrix, global_vector,
        n_samples=n_virtual_samples,
        noise_strategy='adaptive'
    )

    print("\n💾 Saving virtual samples...")
    # Save path
    save_dir = r"D:\Oswaldo's surf project\My Database"
    virtual_samples_path = os.path.join(save_dir, "virtual_samples_100000.npy")
    np.save(virtual_samples_path, virtual_hour_matrices)
    print(f"✅ 24-hour virtual samples saved: {virtual_samples_path}")
    print(f"  Shape: {virtual_hour_matrices.shape}")

    metadata = {
        "n_samples": n_virtual_samples,
        "hour_matrix_shape": list(virtual_hour_matrices.shape),
        "global_vector_shape": list(virtual_global_vectors.shape),
        "generation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "description": "100000 virtual samples generated by DeepAE "
    }

    metadata_path = os.path.join(save_dir, "virtual_samples_metadata2.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"✅ Metadata saved: {metadata_path}")

    # 3. Prepare training data
    print("\n📥 Preparing training data:")
    mode = 'combined'  # You can modify this to choose different modes
    X = prepare_dae_input(virtual_hour_matrices, virtual_global_vectors, mode=mode)

    # 4. Split training, validation and test sets (consistent with 85/15 split in PDF)
    print("\n📊 Dataset splitting (85% training, 15% validation):")
    n_samples = X.shape[0]
    train_ratio = 0.85
    val_ratio = 0.15

    # 按时间顺序划分（假设样本是按时间顺序的）
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)

    X_train = X[:train_end]
    X_val = X[train_end:val_end]

    # X_test = X[val_end:]  # 创建测试集

    print(f"  Training Set: {X_train.shape[0]:,} samples ({train_ratio * 100:.0f}%)")
    print(f"  Validation Set: {X_val.shape[0]:,} samples ({val_ratio * 100:.0f}%)")
    print(f"  Total Samples: {n_samples:,} ")
    print(f"  Actual Sample Distribution: Train:{len(X_train):,} | Validation:{len(X_val):,} | Total:{len(X_train) + len(X_val):,}")

    # 5. 训练AutoEncoder
    autoencoder, encoder, decoder, history, total_params = train_autoencoder(
        X_train, X_val,
        encoding_dim=32,  # 更大的编码维度以捕捉更多信息
        epochs=80
    )

    # 6. 评估和异常检测
    print("\n🔍 Evaluating Deep_AE model on validation set:")
    threshold, val_anomalies_count, val_anomaly_ratio, val_errors, val_reconstructed = evaluate_and_detect_anomalies(
        autoencoder, X_val, X_val, percentile=95
    )

    # 7. 在整个数据集上评估
    print("\n🔍 Evaluating on the complete Deep_AE dataset:")
    # 使用所有35040个样本进行评估
    X_all = X  # 所有生成的虚拟样本

    all_reconstructed = autoencoder.predict(X_all, verbose=0)
    all_errors = np.mean((X_all - all_reconstructed) ** 2, axis=1)

    # 检测异常
    all_anomalies = all_errors > threshold
    all_anomalies_count = np.sum(all_anomalies)
    all_anomaly_ratio = all_anomalies_count / len(X_all) * 100

    print(f"  Total samples: {len(X_all):,}")
    print(f"  Anomalies detected: {all_anomalies_count:,}")
    print(f"  Anomaly proportion: {all_anomaly_ratio:.2f}%")

    # 8. 可视化结果（使用所有35040个样本）
    print("\n📊 Generating visualization for all Deep_AE samples:")
    visualize_results(
        history, X_all, all_reconstructed, all_errors, threshold,
        feature_mapping, model_name="Deep AutoEncoder (100000 samples)"
    )

    # 9. 特征编码和分析
    encoded_features = encode_and_analyze(encoder, X_all, feature_mapping)

    # 10. 保存模型
    save_models(autoencoder, encoder, decoder, model_save_path)

    # 11. 生成分析报告
    print("\n" + "=" * 80)
    print("📋 Deep_AE Training Completion Report")
    print("=" * 80)

    # 输出与PDF一致的格式
    print(f"\n📊 Deep_AE Model Performance Summary:")
    print(f"  Total samples: {len(X_all):,}")  # 改为 X_all 的长度
    print(f"  Actual training samples used: {X_train.shape[0]:,}")
    print(f"  Actual validation samples used: {X_val.shape[0]:,}")
    print(f"  Total params: {total_params:,}")
    print(f"  Validation loss (MSE): {history.history['val_loss'][-1]:.6f}")
    print(f"  95th percentile threshold (validation): {threshold:.6f}")
    print(f"  Anomaly detection (all samples): {all_anomalies_count:,} / {len(X_all):,} ({all_anomaly_ratio:.2f}%)")

    # 与PDF中的模型对比
    # print("\n📈 与PDF模型对比:")
    # print("-" * 80)
    # print(f"{'Model':<15} {'Params':<12} {'Validation MAE':<18} {'Anomaly Rate':<15} {'Samples':<10}")
    # print("-" * 80)
    # print(f"{'LSTM-AE':<15} {'29,124':<12} {'~0.401':<18} {'5.77%':<15} {'35,040':<10}")
    # print(f"{'Conv-AE':<15} {'7,892':<12} {'~0.388':<18} {'6.00%':<15} {'35,040':<10}")
    # print(f"{'Conv-VAE':<15} {'9,444':<12} {'~0.387':<18} {'6.07%':<15} {'35,040':<10}")
    # print(
    #     f"{'Our Model':<15} {f'{total_params:,}':<12} {f'{history.history["val_loss"][-1]:.6f}':<18} {f'{all_anomaly_ratio:.2f}%':<15} {f'{n_virtual_samples:,}':<10}")
    # print("-" * 80)

    # 保存详细结果
    results = {
        'model_name': 'Deep_AutoEncoder',
        'total_params': int(total_params),
        'n_virtual_samples': n_virtual_samples,
        'n_train_samples': int(X_train.shape[0]),
        'n_val_samples': int(X_val.shape[0]),
        'n_all_samples': int(len(X_all)),
        'input_dimension': int(X.shape[1]),
        'latent_dimension': int(encoded_features.shape[1]),
        'compression_ratio': float(X.shape[1] / encoded_features.shape[1]),
        'validation_threshold': float(threshold),
        'validation_anomalies': int(val_anomalies_count),
        'all_anomalies': int(all_anomalies_count),
        'anomaly_ratio_all': float(all_anomaly_ratio),
        'anomaly_ratio_val': float(val_anomaly_ratio),
        'final_train_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
        'final_train_mae': float(history.history['mae'][-1]),
        'final_val_mae': float(history.history.get('val_mae', [0])[-1]),
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    results_path = os.path.join(base_path, 'deep_ae_training_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Training results saved to: {results_path}")
    print("\n🎉 Deep AutoEncoder training pipeline completed!")


# ====================== 9. 辅助函数 ======================
def encode_and_analyze(encoder, X, feature_mapping):
    """
    使用编码器提取特征并分析
    """
    print("\n🔍 Feature Encoding and Analysis:")

    # Extract encoded features
    encoded_features = encoder.predict(X, verbose=0)
    print(f"  Encoded feature shape: {encoded_features.shape}")
    print(
        f"  Compression ratio: {X.shape[1]}:{encoded_features.shape[1]} = {X.shape[1] / encoded_features.shape[1]:.1f}x")

    return encoded_features


def save_models(autoencoder, encoder, decoder, save_path):
    """
    保存训练好的模型
    """
    # 确保保存路径存在
    os.makedirs(save_path, exist_ok=True)

    # 保存为HDF5格式
    autoencoder.save(f"{save_path}autoencoder_model2.h5")
    encoder.save(f"{save_path}encoder_model2.h5")
    decoder.save(f"{save_path}decoder_model2.h5")

    print(f"\n💾 Deep_AE training results saved to: {save_path}")
    print(f"  - autoencoder_model2.h5")
    print(f"  - encoder_model2.h5")
    print(f"  - decoder_model2.h5")



# ====================== 10. 运行主流程 ======================
if __name__ == "__main__":
    # 设置TensorFlow日志级别
    tf.get_logger().setLevel('ERROR')

    # 设置matplotlib中文字体
    # plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 运行主流程
    main()