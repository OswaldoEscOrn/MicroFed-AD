import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
import os
import warnings
import json
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, UpSampling1D,
                                     Dense, Flatten, Reshape, BatchNormalization,
                                     Dropout, Lambda, Layer)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

warnings.filterwarnings('ignore')


# ====================== 1. 自定义层 ======================
class Sampling(Layer):
    """重参数化技巧层"""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class VAELossLayer(Layer):
    """VAE损失计算层"""

    def __init__(self, kl_weight=0.001, **kwargs):
        super(VAELossLayer, self).__init__(**kwargs)
        self.kl_weight = kl_weight
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.recon_loss_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    def call(self, inputs):
        x_true, x_pred, z_mean, z_log_var = inputs

        # 计算重建损失
        reconstruction_loss = K.mean(K.abs(x_true - x_pred))

        # 计算KL散度
        kl_loss = -0.5 * K.sum(
            1 + z_log_var - K.square(z_mean) - K.exp(z_log_var),
            axis=1
        )
        kl_loss = K.mean(kl_loss) * self.kl_weight

        total_loss = reconstruction_loss + kl_loss

        self.add_loss(total_loss)
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return x_pred

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.recon_loss_tracker, self.kl_loss_tracker]

    def compute_output_shape(self, input_shape):
        return input_shape[1]


# ====================== 2. 数据加载函数 ======================
def load_data():
    """加载数据，使用新的文件路径"""
    print("=" * 80)
    print("CONVOLUTIONAL VARIATIONAL AUTOENCODER TRAINING PIPELINE")
    print("=" * 80)

    print("\n📂 Loading preprocessed sliding windows...")

    # 新的配置参数
    DATA_DIR = r"D:\Oswaldo's surf project\70% Virtual 30% Real Database_Hourly\100k_windows_forming"
    X_PATH = os.path.join(DATA_DIR, "x_windows_100k.npy")
    SCALED_DF_PATH = os.path.join(DATA_DIR, "processed_time_series_augmented_100k.csv")

    # 加载窗口数据
    if not os.path.exists(X_PATH):
        raise FileNotFoundError(f"X_windows.npy not found at {X_PATH}")

    X = np.load(X_PATH)
    print(f"Loaded X shape: {X.shape}")  # (n_windows, 24, n_features)

    n_samples, timesteps, n_features = X.shape
    print(f"Data dimensions: {n_samples} samples, {timesteps} timesteps, {n_features} features")

    # 加载小时数据用于后续分析
    print("\n📂 Loading preprocessed time series data...")
    if os.path.exists(SCALED_DF_PATH):
        df_hourly = pd.read_csv(SCALED_DF_PATH, index_col=0, parse_dates=True)
        print(f"Hourly data shape: {df_hourly.shape}")

        # 获取实际的特征名称
        actual_features = df_hourly.columns.tolist()
        print(f"Actual features in data: {actual_features}")
    else:
        print(f"Warning: {SCALED_DF_PATH} not found, creating placeholder hourly data")
        # 创建占位数据
        n_hours = n_samples + timesteps - 1
        date_range = pd.date_range(start='2013-03-01', periods=n_hours, freq='H')
        df_hourly = pd.DataFrame(
            np.random.randn(n_hours, n_features),
            index=date_range,
            columns=[f'Feature_{i}' for i in range(n_features)]
        )
        actual_features = df_hourly.columns.tolist()

    # 检查数据范围
    print(f"\n📊 Data statistics:")
    print(f"  Min: {X.min():.4f}")
    print(f"  Max: {X.max():.4f}")
    print(f"  Mean: {X.mean():.4f}")
    print(f"  Std: {X.std():.4f}")

    return X, df_hourly, actual_features, timesteps, n_features


# ====================== 3. 数据划分函数 ======================
def split_data(X, split_ratio=0.85):
    """数据划分"""
    n_samples = X.shape[0]
    split_idx = int(split_ratio * n_samples)

    X_train = X[:split_idx]
    X_val = X[split_idx:]

    print(f"\n📊 Dataset splitting:")
    print(f"  Training windows   : {X_train.shape} ({split_ratio * 100:.1f}%)")
    print(f"  Validation windows : {X_val.shape} ({100 - split_ratio * 100:.1f}%)")
    print(f"  Total windows      : {n_samples}")

    return X_train, X_val


# ====================== 4. 构建Conv-VAE模型 ======================
def build_conv_vae(timesteps, n_features, latent_dim=16, kl_weight=0.001):
    """构建一维卷积变分自编码器"""
    print(f"\n🔧 Building Conv-VAE model:")
    print(f"  Input shape: ({timesteps}, {n_features})")
    print(f"  Latent dimension: {latent_dim}")
    print(f"  KL weight: {kl_weight}")

    encoder_inputs = Input(shape=(timesteps, n_features), name='encoder_input')

    # 编码器
    x = Conv1D(32, kernel_size=5, activation='relu', padding='same')(encoder_inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2, padding='same')(x)
    x = Dropout(0.2)(x)

    x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2, padding='same')(x)
    x = Dropout(0.2)(x)

    x = Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2, padding='same')(x)

    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)

    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    z = Sampling()([z_mean, z_log_var])

    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')

    # 解码器
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    conv_shape = (timesteps // 8, 128)  # 经过3次池化，每次除以2

    x = Dense(int(np.prod(conv_shape)), activation='relu')(latent_inputs)
    x = Reshape(conv_shape)(x)

    x = Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling1D(size=2)(x)

    x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling1D(size=2)(x)

    x = Conv1D(32, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling1D(size=2)(x)

    # 使用sigmoid激活函数
    decoder_outputs = Conv1D(n_features, kernel_size=5, activation='sigmoid', padding='same')(x)

    decoder = Model(latent_inputs, decoder_outputs, name='decoder')

    # 完整VAE模型
    z_mean, z_log_var, z = encoder(encoder_inputs)
    outputs = decoder(z)
    final_outputs = VAELossLayer(kl_weight=kl_weight)([encoder_inputs, outputs, z_mean, z_log_var])

    vae = Model(encoder_inputs, final_outputs, name='conv_vae')

    # 编译模型
    vae.compile(optimizer=Adam(learning_rate=0.001))

    # 计算参数
    total_params = vae.count_params()
    trainable_params = np.sum([np.prod(v.shape) for v in vae.trainable_weights])
    non_trainable_params = total_params - trainable_params

    print(f"\n📊 Model parameters count:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Non-trainable parameters: {non_trainable_params:,}")

    return vae, encoder, decoder, total_params


# ====================== 5. 模型训练 ======================
def train_model(vae, X_train, X_val, epochs=80, batch_size=128, patience=12):
    """训练模型"""
    print(f"\n🚀 Starting training...")

    # 显示模型摘要
    vae.summary()

    # 回调函数
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

    print(f"\n📈 Training configuration:")
    print(f"  Batch Size: {batch_size}")
    print(f"  Max Epochs: {epochs}")

    # 训练模型
    history = vae.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    return history


# ====================== 6. 异常检测 ======================
def detect_anomalies(vae, X, X_val, threshold_percentile=95, batch_size=128):
    """异常检测"""
    print("\n🔍 Computing reconstruction errors...")

    # 验证集重建误差
    reconstructions_val = vae.predict(X_val, batch_size=batch_size, verbose=0)
    mae_val = np.mean(np.abs(X_val - reconstructions_val), axis=(1, 2))

    # 计算阈值
    threshold = np.percentile(mae_val, threshold_percentile)
    print(f"Validation (assumed normal) MAE {threshold_percentile}th percentile threshold: {threshold:.6f}")

    # 全数据集重建误差
    reconstructions_full = vae.predict(X, batch_size=batch_size, verbose=0)
    mae_full = np.mean(np.abs(X - reconstructions_full), axis=(1, 2))

    # 检测异常
    anomaly_flags = (mae_full > threshold).astype(int)
    anomaly_count = np.sum(anomaly_flags)
    anomaly_ratio = anomaly_count / len(mae_full) * 100

    print(f"\nDetected {anomaly_count} anomalous windows out of {len(mae_full)} ({anomaly_ratio:.2f}%)")

    return mae_full, anomaly_flags, threshold, reconstructions_full


# ====================== 7. 可视化函数 ======================
def visualize_results(df_hourly, mae_full, anomaly_flags, threshold,
                      X_original=None, X_reconstructed=None, history=None,
                      actual_features=None, model_name="Conv-VAE", timesteps=24):
    """可视化结果"""
    print("\n📊 Generating visualizations...")

    # 设置保存路径
    VISUALIZATION_SAVE_DIR = r"D:\Oswaldo's surf project\70% Virtual 30% Real Database_Hourly\visualizations"
    os.makedirs(VISUALIZATION_SAVE_DIR, exist_ok=True)

    # 创建综合可视化图表
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(20, 15))

    # 1. 训练历史
    if history is not None:
        ax1 = plt.subplot(3, 3, 1)
        epochs = range(1, len(history.history['loss']) + 1)
        ax1.plot(epochs, history.history['loss'], label='Training Loss', linewidth=2)
        ax1.plot(epochs, history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax1.set_title(f'{model_name} - Training History', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Total Loss', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # 2. 重建误差时间序列
    ax2 = plt.subplot(3, 3, 2 if history is None else 2)

    # 创建时间索引
    if len(df_hourly) >= len(mae_full):
        time_indices = df_hourly.index[timesteps - 1: timesteps - 1 + len(mae_full)]
    else:
        time_indices = range(len(mae_full))

    ax2.plot(time_indices, mae_full, label='Reconstruction MAE',
             color='purple', alpha=0.7, linewidth=1)
    ax2.axhline(threshold, color='red', linestyle='--', linewidth=2,
                label=f'Threshold ({threshold:.5f})')

    # 标记异常点
    anomaly_indices = np.where(anomaly_flags == 1)[0]
    if len(time_indices) == len(mae_full):
        ax2.scatter(time_indices[anomaly_indices], mae_full[anomaly_indices],
                    color='red', marker='o', s=30, label='Detected Anomaly', alpha=0.6)

    ax2.set_title(f'{model_name} - Reconstruction Error & Detected Anomalies',
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Mean Absolute Error (per window)', fontsize=12)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)

    # 3. 重建误差分布
    ax3 = plt.subplot(3, 3, 3 if history is None else 3)
    n_bins = min(50, len(mae_full) // 20)
    ax3.hist(mae_full, bins=n_bins, alpha=0.7, color='skyblue',
             edgecolor='black', density=True)
    ax3.axvline(x=threshold, color='red', linestyle='--', linewidth=2,
                label=f'Threshold ({threshold:.4f})')

    # 添加高斯分布拟合
    from scipy.stats import norm
    mu, std = norm.fit(mae_full)
    xmin, xmax = ax3.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    ax3.plot(x, p, 'k', linewidth=2,
             label=f'Normal fit: μ={mu:.4f}, σ={std:.4f}')

    ax3.set_title('Reconstruction Error Distribution', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Reconstruction Error (MAE)', fontsize=12)
    ax3.set_ylabel('Density', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 原始vs重建对比（第一个样本）
    ax4 = plt.subplot(3, 3, 4 if history is None else 4)
    if X_original is not None and X_reconstructed is not None and len(X_original) > 0:
        sample_idx = 0
        time_steps = range(timesteps)

        # 绘制第一个特征
        if actual_features and len(actual_features) > 0:
            feature_name = actual_features[0]
        else:
            feature_name = 'Feature_0'

        ax4.plot(time_steps, X_original[sample_idx, :, 0],
                 label='Original', linewidth=2, alpha=0.8)
        ax4.plot(time_steps, X_reconstructed[sample_idx, :, 0],
                 label='Reconstructed', linewidth=2, alpha=0.8,
                 color='red', linestyle='--')

        ax4.set_title(f'Sample {sample_idx} - {feature_name}',
                      fontsize=14, fontweight='bold')
        ax4.set_xlabel('Time Step (hours)', fontsize=12)
        ax4.set_ylabel('Feature Value', fontsize=12)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    # 5. 损失分量（如果有历史）
    ax5 = plt.subplot(3, 3, 5 if history is None else 5)
    if history is not None:
        # 尝试从历史中获取重构损失和KL损失
        if 'recon_loss' in history.history and 'kl_loss' in history.history:
            ax5.plot(epochs, history.history['recon_loss'],
                     label='Reconstruction Loss', linewidth=2, color='green')
            ax5.plot(epochs, history.history['kl_loss'],
                     label='KL Loss', linewidth=2, color='orange')
            ax5.set_title('Loss Components', fontsize=14, fontweight='bold')
            ax5.set_xlabel('Epoch', fontsize=12)
            ax5.set_ylabel('Loss', fontsize=12)
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'Loss Components\nNot Available',
                     ha='center', va='center', transform=ax5.transAxes, fontsize=12)
            ax5.set_title('Loss Components', fontsize=14, fontweight='bold')
            ax5.axis('off')

    # 6. 异常检测性能分析
    ax6 = plt.subplot(3, 3, 6 if history is None else 6)
    thresholds = np.percentile(mae_full, range(90, 100))
    anomaly_rates = []

    for t in thresholds:
        anomaly_rate = np.sum(mae_full > t) / len(mae_full) * 100
        anomaly_rates.append(anomaly_rate)

    ax6.plot(thresholds, anomaly_rates, 'o-', linewidth=2, markersize=6)
    ax6.axvline(x=threshold, color='red', linestyle='--', linewidth=1.5,
                label=f'95th percentile ({threshold:.4f})')
    ax6.set_title('Anomaly Rate vs Threshold', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Threshold', fontsize=12)
    ax6.set_ylabel('Anomaly Rate (%)', fontsize=12)
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # 7. 误差箱线图
    ax7 = plt.subplot(3, 3, 7 if history is None else 7)
    if len(anomaly_indices) > 0:
        normal_errors = mae_full[anomaly_flags == 0]
        anomaly_errors = mae_full[anomaly_flags == 1]

        box_data = [normal_errors, anomaly_errors]
        box_labels = ['Normal', 'Anomaly']

        bp = ax7.boxplot(box_data, labels=box_labels, patch_artist=True)

        colors = ['lightblue', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax7.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5)
        ax7.set_title('Error Distribution by Category', fontsize=14, fontweight='bold')
        ax7.set_ylabel('Reconstruction MAE', fontsize=12)
        ax7.grid(True, alpha=0.3, axis='y')

    # 8. 模型比较
    ax8 = plt.subplot(3, 3, 8 if history is None else 8)
    models = ['LSTM-AE', 'Conv-AE', 'Conv-VAE']
    # 近似参数数量（根据实际训练调整）
    if X_original is not None:
        conv_vae_params = history.model.count_params() if history else 'N/A'
        params = [29124, 7892, conv_vae_params if isinstance(conv_vae_params, int) else 12000]
        colors = ['lightgreen', 'lightcoral', 'gold']

        bars = ax8.bar(models, params, color=colors, alpha=0.8)
        ax8.set_title('Model Parameter Comparison', fontsize=14, fontweight='bold')
        ax8.set_ylabel('Number of Parameters', fontsize=12)
        ax8.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for bar, param in zip(bars, params):
            if isinstance(param, int):
                ax8.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(params) * 0.05,
                         f'{param:,}', ha='center', va='bottom', fontsize=9)

    # 9. 性能总结
    ax9 = plt.subplot(3, 3, 9 if history is None else 9)
    ax9.axis('off')

    if history is not None:
        best_val_loss = min(history.history['val_loss'])
        best_epoch = np.argmin(history.history['val_loss']) + 1
        final_val_loss = history.history['val_loss'][-1]
        val_loss_text = f"Best Val Loss: {best_val_loss:.4f} (epoch {best_epoch})\nFinal Val Loss: {final_val_loss:.4f}"
    else:
        val_loss_text = f"Validation Threshold: {threshold:.4f}"

    summary_text = f"""{model_name} Performance Summary

Total Samples: {len(mae_full):,}
Anomalies Detected: {np.sum(anomaly_flags):,}
Anomaly Ratio: {np.mean(anomaly_flags) * 100:.2f}%

Anomaly Threshold: {threshold:.4f}

Latent Dimension: 16
KL Weight: 0.001

{val_loss_text}
"""

    ax9.text(0.1, 0.95, summary_text, fontsize=11,
             verticalalignment='top', family='monospace')

    plt.tight_layout()

    # 保存图表
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(VISUALIZATION_SAVE_DIR, f"100k_conv_vae_visualization_{timestamp}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📸 Visualization saved to: {save_path}")

    plt.close(fig)

    return save_path


# ====================== 7.1 类型转换辅助函数 ======================
def convert_numpy_types(obj):
    """递归转换numpy类型为Python原生类型以便JSON序列化"""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (bool, str, int, float)):
        return obj
    elif obj is None:
        return None
    else:
        # 对于其他类型，尝试转换为字符串
        try:
            return str(obj)
        except:
            return f"<{type(obj).__name__}>"


# ====================== 8. 保存结果 ======================
def save_results(results_dict, save_dir):
    """保存训练结果到JSON文件"""
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(save_dir, f"100k_conv_vae_results_{timestamp}.json")

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)

    print(f"💾 Results saved to: {results_path}")
    return results_path


# ====================== 9. 主流程 ======================
def main():
    """主函数"""
    # 设置TensorFlow优化
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    print("=" * 80)
    print("CONVOLUTIONAL VARIATIONAL AUTOENCODER TRAINING PIPELINE")
    print("=" * 80)

    import time
    start_time=time.time()

    try:
        # 1. 加载数据
        X, df_hourly, actual_features, timesteps, n_features = load_data()

        print(f"\n📊 Data Information:")
        print(f"  Number of features: {n_features}")
        print(f"  Timesteps per window: {timesteps}")
        print(f"  Feature names: {actual_features}")

        # 2. 划分数据
        X_train, X_val = split_data(X, split_ratio=0.85)

        # 3. 构建模型
        print("\n" + "=" * 80)
        print("MODEL CONSTRUCTION")
        print("=" * 80)

        vae, encoder, decoder, total_params = build_conv_vae(
            timesteps=timesteps,
            n_features=n_features,
            latent_dim=16,
            kl_weight=0.001
        )

        # 4. 训练模型
        print("\n" + "=" * 80)
        print("MODEL TRAINING")
        print("=" * 80)

        history = train_model(
            vae, X_train, X_val,
            epochs=80,
            batch_size=128,
            patience=12
        )

        # 计算最佳验证损失
        if history is not None and 'val_loss' in history.history:
            best_val_loss = min(history.history['val_loss'])
            best_epoch = np.argmin(history.history['val_loss']) + 1
            final_train_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1]

            print(f"\n" + "=" * 60)
            print("VALIDATION LOSS SUMMARY")
            print("=" * 60)
            print(f"Training completed:")
            print(f"  - Best validation loss: {best_val_loss:.6f} (epoch {best_epoch})")
            print(f"  - Final training loss: {final_train_loss:.6f}")
            print(f"  - Final validation loss: {final_val_loss:.6f}")
            print(f"  - Total training epochs: {len(history.history['loss'])}")
            print("=" * 60)
        else:
            best_val_loss = history.history['val_loss'][-1] if history else None
            best_epoch = None

        # 5. 保存模型
        model_save_dir = r"D:\Oswaldo's surf project\70% Virtual 30% Real Database_Hourly\models"
        os.makedirs(model_save_dir, exist_ok=True)

        # 保存完整VAE模型
        vae_path = os.path.join(model_save_dir, "100k_conv_vae_autoencoder.h5")

        # 注意：保存和加载自定义层模型需要指定custom_objects
        vae.save(vae_path, save_format='h5')
        print(f"\n💾 Model saved to: {vae_path}")

        # 保存编码器
        encoder_path = os.path.join(model_save_dir, "100k_conv_vae_encoder.h5")
        encoder.save(encoder_path, save_format='h5')
        print(f"💾 Encoder saved to: {encoder_path}")

        # 保存解码器
        decoder_path = os.path.join(model_save_dir, "100k_conv_vae_decoder.h5")
        decoder.save(decoder_path, save_format='h5')
        print(f"💾 Decoder saved to: {decoder_path}")

        # 6. 异常检测
        print("\n" + "=" * 80)
        print("ANOMALY DETECTION")
        print("=" * 80)

        mae_full, anomaly_flags, threshold, reconstructions_full = detect_anomalies(
            vae, X, X_val, threshold_percentile=95, batch_size=128
        )

        # 7. 可视化
        print("\n" + "=" * 80)
        print("VISUALIZATION")
        print("=" * 80)

        viz_path = visualize_results(
            df_hourly, mae_full, anomaly_flags, threshold,
            X_original=X, X_reconstructed=reconstructions_full, history=history,
            actual_features=actual_features, model_name="Conv-VAE", timesteps=timesteps
        )

        # 计算总运行时间
        end_time = time.time()
        total_time_seconds = end_time - start_time
        total_time_minutes = total_time_seconds / 60
        total_time_hours = total_time_minutes / 60

        # 输出总运行时间
        print(f"\n⏱️  Total Execution Time:")
        print(f"  Total time: {total_time_seconds:.2f} seconds")
        print(f"            : {total_time_minutes:.2f} minutes")
        print(f"            : {total_time_hours:.2f} hours")

        # 8. 准备结果字典
        results_dict = {
            'model_name': 'Conv-VAE',
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_info': {
                'total_samples': X.shape[0],
                'timesteps': timesteps,
                'n_features': n_features,
                'training_samples': X_train.shape[0],
                'validation_samples': X_val.shape[0],
                'actual_features': actual_features
            },
            'model_info': {
                'total_params': total_params,
                'batch_size': 128,
                'epochs_trained': len(history.history['loss']),
                'final_train_loss': float(history.history['loss'][-1]),
                'final_val_loss': float(history.history['val_loss'][-1]),
                'best_val_loss': float(best_val_loss) if best_val_loss else None,
                'best_epoch': best_epoch,
                'latent_dim': 16,
                'kl_weight': 0.001
            },
            'anomaly_detection': {
                'threshold_percentile': 95,
                'threshold_value': float(threshold),
                'anomalies_detected': int(np.sum(anomaly_flags)),
                'total_windows': len(mae_full),
                'anomaly_ratio': float(np.mean(anomaly_flags) * 100)
            },
            'file_paths': {
                'model_path': vae_path,
                'encoder_path': encoder_path,
                'decoder_path': decoder_path,
                'visualization_path': viz_path,
                'data_path': r"D:\Oswaldo's surf project\70% Virtual 30% Real Database_Hourly\100k_windows_forming"
            }
        }

        # 转换numpy类型为Python原生类型
        results_dict = convert_numpy_types(results_dict)

        # 9. 保存结果
        results_path = save_results(results_dict, model_save_dir)

        # 10. 打印总结报告
        print("\n" + "=" * 80)
        print("SUMMARY REPORT")
        print("=" * 80)
        print(f"📊 Conv-VAE Training Complete")
        print(f"\n📈 Performance Summary:")
        print(f"  - Total samples processed: {X.shape[0]:,}")
        print(f"  - Training samples: {X_train.shape[0]:,}")
        print(f"  - Validation samples: {X_val.shape[0]:,}")
        print(f"  - Model parameters: {total_params:,}")
        print(f"  - Final training loss: {history.history['loss'][-1]:.6f}")
        print(f"  - Final validation loss: {history.history['val_loss'][-1]:.6f}")
        print(f"  - Best validation loss: {best_val_loss:.6f} (epoch {best_epoch})")
        print(f"  - Anomaly threshold (95th percentile): {threshold:.6f}")
        print(
            f"  - Anomalies detected: {np.sum(anomaly_flags):,} out of {len(mae_full):,} ({np.mean(anomaly_flags) * 100:.2f}%)")

        print(f"\n📁 Output Files:")
        print(f"  - VAE Model: {vae_path}")
        print(f"  - Encoder: {encoder_path}")
        print(f"  - Decoder: {decoder_path}")
        print(f"  - Visualization: {viz_path}")
        print(f"  - Results: {results_path}")

        print(f"\n✅ Conv-VAE training pipeline completed successfully!")
        print(f"   - Model combines probabilistic latent space with convolutional efficiency")
        print(f"   - Validation loss properly calculated and reported")

    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


# ====================== 执行主函数 ======================
if __name__ == "__main__":
    # 设置matplotlib
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 100

    # 运行主函数
    exit_code = main()
    exit(exit_code)