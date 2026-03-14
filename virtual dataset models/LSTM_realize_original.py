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
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings('ignore')


# ====================== 1. 加载LSTM-AE数据 ======================
def load_lstm_ae_data():
    """
    加载为LSTM-AE准备的数据
    """
    print("📂 Loading LSTM-AE data (using DeepAE-generated virtual samples)...")

    # 直接使用DeepAE生成的虚拟样本
    virtual_samples_path = r"D:\Oswaldo's surf project\My Database\virtual_samples_35040.npy"

    if not os.path.exists(virtual_samples_path):
        print(f"❌ DeepAE virtual samples file not found: {virtual_samples_path}")
        print("Please run DeepAE_realize.py first to generate the data")
        return None, None, None

    try:
        # 加载数据
        X_lstm = np.load(virtual_samples_path)

        print("✅ LSTM-AE data loaded successfully:")
        print(f"  Sequence data shape: {X_lstm.shape}")
        print(f"  Number of samples: {X_lstm.shape[0]:,}")
        print(f"  Timesteps: {X_lstm.shape[1]} hours ({X_lstm.shape[1] // 24} days)")
        print(f"  Number of features: {X_lstm.shape[2]}")
        print(f"  Data source: DeepAE virtual samples")

        # 加载特征映射
        feature_mapping_path = r"D:\Oswaldo's surf project\My Database\Merge_and_DAEfeature_mapping.json"
        with open(feature_mapping_path, 'r', encoding='utf-8') as f:
            feature_mapping = json.load(f)

        # 创建配置
        lstm_config = {
            "n_samples": X_lstm.shape[0],
            "sequence_hours": X_lstm.shape[1],
            "n_features": X_lstm.shape[2]
        }

        return X_lstm, lstm_config, feature_mapping

    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None, None, None

# ====================== 2. LSTM-AE模型构建 ======================
def build_lstm_autoencoder(sequence_length, n_features, lstm_units=128, encoding_dim=32):
    """
    构建LSTM Autoencoder模型
    """
    print(f"\n🔧 Building LSTM-AutoEncoder model:")
    print(f"  Input shape: ({sequence_length}, {n_features})")
    print(f"  LSTM units: {lstm_units}")
    print(f"  Encoding dimension: {encoding_dim}")

    # ====== 编码器 ======
    inputs = Input(shape=(sequence_length, n_features), name='lstm_input')

    # 第一层LSTM
    encoded = LSTM(lstm_units, activation='relu', return_sequences=True,
                   name='encoder_lstm1')(inputs)
    encoded = BatchNormalization(name='encoder_bn1')(encoded)
    encoded = Dropout(0.3, name='encoder_dropout1')(encoded)

    # 第二层LSTM
    encoded = LSTM(lstm_units // 2, activation='relu', return_sequences=True,
                   name='encoder_lstm2')(encoded)
    encoded = BatchNormalization(name='encoder_bn2')(encoded)
    encoded = Dropout(0.3, name='encoder_dropout2')(encoded)

    # 第三层LSTM（不返回序列）
    encoded = LSTM(lstm_units // 4, activation='relu', return_sequences=False,
                   name='encoder_lstm3')(encoded)

    # 瓶颈层
    encoded = Dense(encoding_dim, activation='relu', name='bottleneck')(encoded)

    # ====== 解码器 ======
    # 重复向量
    decoded = RepeatVector(sequence_length, name='repeat_vector')(encoded)

    # 第一层LSTM
    decoded = LSTM(lstm_units // 4, activation='relu', return_sequences=True,
                   name='decoder_lstm1')(decoded)
    decoded = BatchNormalization(name='decoder_bn1')(decoded)
    decoded = Dropout(0.3, name='decoder_dropout1')(decoded)

    # 第二层LSTM
    decoded = LSTM(lstm_units // 2, activation='relu', return_sequences=True,
                   name='decoder_lstm2')(decoded)
    decoded = BatchNormalization(name='decoder_bn2')(decoded)
    decoded = Dropout(0.3, name='decoder_dropout2')(decoded)

    # 第三层LSTM
    decoded = LSTM(lstm_units, activation='relu', return_sequences=True,
                   name='decoder_lstm3')(decoded)
    decoded = BatchNormalization(name='decoder_bn3')(decoded)

    # 输出层
    outputs = TimeDistributed(Dense(n_features, activation='linear'),
                              name='output')(decoded)

    # ====== 完整模型 ======
    autoencoder = Model(inputs, outputs, name='lstm_autoencoder')

    # 编码器模型（用于特征提取）
    encoder = Model(inputs, encoded, name='lstm_encoder')

    return autoencoder, encoder


# ====================== 3. 数据准备函数 ======================
def prepare_lstm_train_val_split(X_lstm, train_ratio=0.85, val_ratio=0.15, shuffle=True):
    """
    为LSTM-AE准备训练和验证数据
    """
    n_samples = X_lstm.shape[0]

    if shuffle:
        # 随机打乱
        indices = np.random.permutation(n_samples)
        X_shuffled = X_lstm[indices]
    else:
        X_shuffled = X_lstm

    # 划分训练集、验证集
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)

    X_train = X_shuffled[:train_end]
    X_val = X_shuffled[train_end:val_end]

    print(f"\n📊 LSTM dataset splitting:")
    print(f"  Training Set: {X_train.shape[0]:,} samples ({train_ratio * 100:.1f}%)")
    print(f"  Validation Set: {X_val.shape[0]:,} samples ({val_ratio * 100:.1f}%)")
    print(f"  Total Samples: {n_samples:,}")

    return X_train, X_val


# ====================== 4. 模型训练 ======================
# 修改你的 LSTM_realize_original.py 训练部分

def train_lstm_autoencoder(X_train, X_val, sequence_length, n_features,
                           lstm_units=128, encoding_dim=32, epochs=80):
    """
    训练LSTM Autoencoder - 优化版本
    """
    print(f"\n🚀 start training LSTM-AutoEncoder...")

    # 构建模型（保持不变）
    autoencoder, encoder = build_lstm_autoencoder(
        sequence_length=sequence_length,
        n_features=n_features,
        lstm_units=lstm_units,
        encoding_dim=encoding_dim
    )

    # 🔍 添加详细参数分析
    print(f"\n📋 Model Architecture Summary:")
    autoencoder.summary(print_fn=lambda x: print(f"  {x}"))

    # 计算详细参数
    total_params = autoencoder.count_params()
    trainable_params = np.sum([tf.keras.backend.count_params(w) for w in autoencoder.trainable_weights])
    non_trainable_params = total_params - trainable_params

    print(f"\n📊 model parameters count:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Non-trainable parameters: {non_trainable_params:,}")
    # print(f"  Trainable ratio: {trainable_params / total_params * 100:.1f}%")

    # 🔧 关键优化1：使用混合精度训练（如果GPU支持）
    from tensorflow.keras import mixed_precision
    try:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print("✅ Mixed precision training enabled (accelerates GPU computation)")
    except:
        print("⚠️ Unable to enable mixed precision training, continuing with default precision")

    # 编译模型
    autoencoder.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    # 🔧 关键优化2：调整批次大小（根据内存调整）
    # 减小批次大小以加速单个epoch
    batch_size = 64  # 从128减少到64

    # 🔧 关键优化3：简化回调
    callbacks = [
        # EarlyStopping(
        #     monitor='val_loss',
        #     patience=8,  # 减少耐心值
        #     restore_best_weights=True,
        #     verbose=1
        # ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,  # 减少耐心值
            min_lr=1e-5,
            verbose=1
        )
    ]

    print(f"\n📈 Training Details:")
    print(f"  Batch Size: {batch_size}")
    print(f"  Max Epochs: {epochs}")

    # 训练模型
    history = autoencoder.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2  # 更简洁的输出
    )

    return autoencoder, encoder, history, autoencoder.count_params()


# ====================== 5. 评估和异常检测 ======================
def evaluate_lstm_anomalies(autoencoder, X_val, X_test=None, percentile=95):
    """
    评估LSTM-AE模型并检测异常
    """
    print("\n📈 LSTM Model Evaluation Results:")

    # Evaluate on validation set
    val_loss, val_mae = autoencoder.evaluate(X_val, X_val, verbose=0)
    print(f"  Validation loss (MSE): {val_loss:.6f}")
    print(f"  Validation MAE: {val_mae:.6f}")

    # Calculate validation set reconstruction errors
    val_reconstructed = autoencoder.predict(X_val, verbose=0)

    # Compute average MSE per sample (averaged across timesteps and features)
    val_errors = np.mean((X_val - val_reconstructed) ** 2, axis=(1, 2))

    # Calculate threshold (95th percentile)
    threshold = np.percentile(val_errors, percentile)
    print(f"  Validation set reconstruction error {percentile}th percentile threshold: {threshold:.6f}")

    # 如果在测试集上检测异常
    anomalies_count = 0
    anomaly_ratio = 0

    if X_test is not None:
        test_reconstructed = autoencoder.predict(X_test, verbose=0)
        test_errors = np.mean((X_test - test_reconstructed) ** 2, axis=(1, 2))

        # 检测异常
        anomalies = test_errors > threshold
        anomalies_count = np.sum(anomalies)
        anomaly_ratio = anomalies_count / len(X_test) * 100

        print(f"  Detected {anomalies_count} anomalous samples out of {len(X_test)} total samples")
        print(f"  Anomaly ratio: {anomaly_ratio:.2f}%")

        return threshold, anomalies_count, anomaly_ratio, test_errors, test_reconstructed, val_errors
    else:
        return threshold, 0, 0, val_errors, val_reconstructed, val_errors


# ====================== 6. 可视化函数 ======================
def visualize_lstm_results(history, X_original, X_reconstructed, errors, threshold,
                           feature_mapping, model_name="LSTM AutoEncoder"):
    """
    可视化LSTM-AE训练结果和异常检测
    """
    print(f"\n📊 Generating LSTM visualization results...")

    # 设置绘图风格
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(20, 12))

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

    # 4. 原始vs重建特征对比（第一个样本的第一个特征）
    ax4 = plt.subplot(3, 3, 4)
    if len(X_original) > 0:
        sample_idx = 0
        feature_idx = 0  # 显示第一个特征

        # 获取第一个样本的所有时间步
        time_steps = range(X_original[sample_idx].shape[0])
        original_values = X_original[sample_idx][:, feature_idx]
        reconstructed_values = X_reconstructed[sample_idx][:, feature_idx]

        ax4.plot(time_steps, original_values, label='Original', linewidth=2, alpha=0.8)
        ax4.plot(time_steps, reconstructed_values, label='Reconstructed', linewidth=2,
                 alpha=0.8, color='red', linestyle='--')

        ax4.set_title(f'Sample {sample_idx} - Feature {feature_idx} Comparison',
                      fontsize=14, fontweight='bold')
        ax4.set_xlabel('Time Step', fontsize=12)
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
        ax5.plot(indices_smooth, errors_smooth, linewidth=1, alpha=0.7, color='blue',
                 label='Smoothed Error')

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

    # 8. 特征重建误差热图（第一个样本）
    ax8 = plt.subplot(3, 3, 8)
    if len(X_original) > 0:
        sample_idx = 0
        feature_errors = (X_original[sample_idx] - X_reconstructed[sample_idx]) ** 2

        im = ax8.imshow(feature_errors.T, aspect='auto', cmap='YlOrRd')
        ax8.set_xlabel('Time Step', fontsize=12)
        ax8.set_ylabel('Feature Index', fontsize=12)
        ax8.set_title(f'Sample {sample_idx} Feature-wise Reconstruction Error',
                      fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax8, label='Reconstruction Error')

        # 设置y轴标签（特征名称）
        if 'hour_feature_names' in feature_mapping:
            hour_feature_names = feature_mapping['hour_feature_names']
            if len(hour_feature_names) == X_original.shape[2]:
                ax8.set_yticks(range(0, len(hour_feature_names), 3))
                ax8.set_yticklabels([hour_feature_names[i] for i in range(0, len(hour_feature_names), 3)])

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
    save_path = f"D:\\Oswaldo's surf project\\My Database\\lstm_visualization_{timestamp}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📸 LSTM visualization results saved to: {save_path}")

    plt.close()

    # 输出特征级分析
    if len(X_original) > 0:
        print("\n📊 LSTM Reconstruction Error Analysis (First Sample):")
        feature_errors = np.mean((X_original[0] - X_reconstructed[0]) ** 2, axis=0)

        if 'hour_feature_names' in feature_mapping:
            hour_feature_names = feature_mapping['hour_feature_names']
            if len(hour_feature_names) == len(feature_errors):
                print("  Feature Reconstruction Error Ranking:")
                sorted_indices = np.argsort(feature_errors)[::-1]
                for i, idx in enumerate(sorted_indices[:5]):
                    print(f"    {i + 1}. {hour_feature_names[idx]}: {feature_errors[idx]:.6f}")


# ====================== 7. 编码特征分析 ======================
def encode_and_analyze_lstm(encoder, X, feature_mapping):
    """
    Extract features using LSTM encoder and perform analysis
    """
    print("\n🔍 LSTM Feature Encoding and Analysis:")

    # Extract encoded features
    encoded_features = encoder.predict(X, verbose=0)
    print(f"  Encoded feature shape: {encoded_features.shape}")

    # Calculate compression ratio
    original_size = X.shape[1] * X.shape[2]  # timesteps × features
    encoded_size = encoded_features.shape[1]
    compression_ratio = original_size / encoded_size

    print(f"  Compression ratio: {original_size}:{encoded_size} = {compression_ratio:.1f}x")
    print(f"  Original dimension: {original_size} → Encoded dimension: {encoded_size}")

    return encoded_features


# ====================== 8. 保存模型 ======================
def save_lstm_models(autoencoder, encoder, save_path):
    """
    保存训练好的LSTM模型
    """
    # 确保保存路径存在
    os.makedirs(save_path, exist_ok=True)

    # 保存为HDF5格式
    autoencoder.save(f"{save_path}lstm_autoencoder_model.h5")
    encoder.save(f"{save_path}lstm_encoder_model.h5")

    print(f"\n💾 LSTM model saved to: {save_path}")
    print(f"  - lstm_autoencoder_model.h5")
    print(f"  - lstm_encoder_model.h5")


def setup_environment():
    """设置优化环境"""
    import os
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'  # 启用oneDNN优化
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 减少TensorFlow日志
    os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())  # 使用所有CPU核心
    print(f"✅ uses {os.cpu_count()} CPU core")

# ====================== 9. 主流程 ======================
def main():
    print("=" * 80)
    print("LSTM AUTOENCODER TRAINING PIPELINE")
    print("=" * 80)

    # 配置路径
    base_path = r"D:\Oswaldo's surf project\My Database"
    model_save_path = os.path.join(base_path, "models\\")

    # 1. 加载LSTM-AE数据
    X_lstm, lstm_config, feature_mapping = load_lstm_ae_data()
    if X_lstm is None:
        print("❌ Failed to load LSTM data, exiting process")
        return

    # 2. 准备训练数据
    X_train, X_val = prepare_lstm_train_val_split(
        X_lstm,
        train_ratio=0.85,
        val_ratio=0.15,
        shuffle=True
    )

    # 3. 训练LSTM Autoencoder
    sequence_length = X_lstm.shape[1]
    n_features = X_lstm.shape[2]

    autoencoder, encoder, history, total_params = train_lstm_autoencoder(
        X_train, X_val,
        sequence_length=sequence_length,
        n_features=n_features,
        lstm_units=128,
        encoding_dim=32,
        epochs=80
    )

    # 4. 在验证集上评估
    print("\n🔍 Evaluating LSTM model on validation set:")
    threshold, val_anomalies_count, val_anomaly_ratio, val_errors, val_reconstructed, val_errors_full = evaluate_lstm_anomalies(
        autoencoder, X_val, X_val, percentile=95
    )

    # 5. 在整个数据集上评估
    print("\n🔍 Evaluating on the complete LSTM dataset:")
    # 使用所有35040个样本进行评估
    X_all = X_lstm

    all_reconstructed = autoencoder.predict(X_all, verbose=0)
    all_errors = np.mean((X_all - all_reconstructed) ** 2, axis=(1, 2))

    # 检测异常
    all_anomalies = all_errors > threshold
    all_anomalies_count = np.sum(all_anomalies)
    all_anomaly_ratio = all_anomalies_count / len(X_all) * 100

    print(f"  Total samples: {len(X_all):,}")
    print(f"  Anomalies detected: {all_anomalies_count:,}")
    print(f"  Anomaly proportion: {all_anomaly_ratio:.2f}%")

    # 6. 可视化结果
    print("\n📊 Generating visualization for all LSTM samples:")
    visualize_lstm_results(
        history, X_all, all_reconstructed, all_errors, threshold,
        feature_mapping, model_name="LSTM AutoEncoder (35040 samples)"
    )

    # 7. 特征编码和分析
    encoded_features = encode_and_analyze_lstm(encoder, X_all, feature_mapping)

    # 8. 保存模型
    save_lstm_models(autoencoder, encoder, model_save_path)

    # 9. 生成分析报告
    print("\n" + "=" * 80)
    print("📋 LSTM Training Completion Report")
    print("=" * 80)

    # Output in consistent format with DeepAE
    print(f"\n📊 LSTM Model Performance Summary:")
    print(f"  Total samples: {X_lstm.shape[0]:,}")
    print(f"  Actual training samples used: {X_train.shape[0]:,}")
    print(f"  Actual validation samples used: {X_val.shape[0]:,}")
    print(f"  Total params: {total_params:,}")
    print(f"  Validation loss (MSE): {history.history['val_loss'][-1]:.6f}")
    print(f"  95th percentile threshold (validation): {threshold:.6f}")
    print(f"  Anomaly detection (all samples): {all_anomalies_count:,} / {len(X_all):,} ({all_anomaly_ratio:.2f}%)")

    # 与DeepAE模型对比
    # print("\n📈 模型对比表:")
    # print("-" * 80)
    # print(f"{'Model':<15} {'Params':<12} {'Validation MAE':<18} {'Anomaly Rate':<15} {'Samples':<10}")
    # print("-" * 80)

    # DeepAE结果（来自你的截图）
    # deepae_params = "29,769"  # 示例值，需要根据实际DeepAE参数调整
    # deepae_mae = "0.178500"  # 来自你的截图
    # deepae_anomaly = "4.56%"  # 来自你的截图
    #
    # print(f"{'DeepAE':<15} {deepae_params:<12} {deepae_mae:<18} {deepae_anomaly:<15} {'35,040':<10}")
    # print(
    #     f"{'LSTM-AE':<15} {f'{total_params:,}':<12} {f'{history.history.get("val_mae", [0])[-1]:.6f}':<18} {f'{all_anomaly_ratio:.2f}%':<15} {f'{X_lstm.shape[0]:,}':<10}")
    # print("-" * 80)

    # 10. 保存详细结果
    results = {
        'model_name': 'LSTM_AutoEncoder',
        'total_params': int(total_params),
        'n_samples': X_lstm.shape[0],
        'n_train_samples': X_train.shape[0],
        'n_val_samples': X_val.shape[0],
        'sequence_length': int(sequence_length),
        'sequence_days': int(sequence_length // 24),
        'n_features': int(n_features),
        'lstm_units': 128,
        'encoding_dim': 32,
        'validation_threshold': float(threshold),
        'validation_anomalies': int(val_anomalies_count),
        'all_anomalies': int(all_anomalies_count),
        'anomaly_ratio_all': float(all_anomaly_ratio),
        'anomaly_ratio_val': float(val_anomaly_ratio),
        'final_train_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
        'final_train_mae': float(history.history['mae'][-1]),
        'final_val_mae': float(history.history.get('val_mae', [0])[-1]),
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'compression_ratio': float((sequence_length * n_features) / 32),
        'data_shape': list(X_lstm.shape)
    }

    results_path = os.path.join(base_path, 'lstm_ae_training_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n💾 LSTM training results saved to: {results_path}")
    print("\n🎉 LSTM AutoEncoder training pipeline completed！")


# ====================== 10. 运行主流程 ======================
if __name__ == "__main__":
    # 设置TensorFlow日志级别
    tf.get_logger().setLevel('ERROR')

    # 设置matplotlib中文字体
    # plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 运行主流程
    main()