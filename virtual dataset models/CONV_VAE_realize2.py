"""
CONVOLUTIONAL VARIATIONAL AUTOENCODER (CONV-VAE) 实现
使用与DeepAE/LSTM/CONV-AE相同的35040个24小时虚拟样本
"""

import json
import os
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, UpSampling1D,
                                     Dense, Flatten, Reshape, BatchNormalization,
                                     Dropout, Lambda, Layer)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns

warnings.filterwarnings('ignore')


# ====================== 1. 自定义VAE层 ======================
class Sampling(Layer):
    """重参数化技巧层"""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# ====================== 1. 自定义VAE损失层（使用新API） ======================
class VAELossLayer(Layer):
    """VAE损失计算层 - 使用新Keras API"""

    def __init__(self, **kwargs):
        super(VAELossLayer, self).__init__(**kwargs)
        # 创建指标跟踪器
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    def build(self, input_shape):
        super(VAELossLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs包含: [x_true, x_pred, z_mean, z_log_var]
        x_true, x_pred, z_mean, z_log_var = inputs

        # 计算重建损失 (MSE)
        reconstruction_loss = K.mean(K.square(x_true - x_pred), axis=[1, 2])

        # 计算KL散度
        kl_loss = -0.5 * K.sum(
            1 + z_log_var - K.square(z_mean) - K.exp(z_log_var),
            axis=1
        )

        # 总损失
        total_loss = reconstruction_loss + kl_loss

        # 添加损失（总损失）
        self.add_loss(K.mean(total_loss))

        # 更新指标跟踪器
        self.reconstruction_loss_tracker.update_state(K.mean(reconstruction_loss))
        self.kl_loss_tracker.update_state(K.mean(kl_loss))

        # 返回预测值
        return x_pred

    @property
    def metrics(self):
        # 返回指标列表
        return [self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def compute_output_shape(self, input_shape):
        # 输出形状与x_pred相同
        return input_shape[1]
# class VAE_Loss(Layer):
#     """自定义VAE损失层"""
#
#     def __init__(self, **kwargs):
#         super(VAE_Loss, self).__init__(**kwargs)
#         self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
#         self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
#         self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
#
#     def call(self, inputs):
#         x, x_decoded, z_mean, z_log_var = inputs
#
#         # 修复：正确计算重建损失
#         # 使用 mean squared error，在特征维度上平均
#         reconstruction_loss = tf.keras.losses.mse(x, x_decoded)  # 形状: (batch_size, sequence_length)
#         reconstruction_loss = tf.reduce_mean(reconstruction_loss, axis=1)  # 在时间步上平均，形状: (batch_size,)
#         reconstruction_loss = tf.reduce_mean(reconstruction_loss)  # 在所有批次上平均，标量
#
#         # 计算KL散度
#         kl_loss = -0.5 * tf.reduce_mean(
#             tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
#         )
#
#         total_loss = reconstruction_loss + kl_loss
#
#         # 更新跟踪器
#         self.add_metric(reconstruction_loss, name='reconstruction_loss')
#         self.add_metric(kl_loss, name='kl_loss')
#         self.add_metric(total_loss, name='loss')
#
#         return x_decoded
#
#     def compute_output_shape(self, input_shapes):
#         # 输出形状与输入x相同
#         return input_shapes[0]


# ====================== 2. 加载DeepAE生成的虚拟样本 ======================
def load_conv_vae_data():
    """
    加载为CONV-VAE准备的数据 - 使用DeepAE生成的100000个虚拟样本
    """
    print("📂 Loading CONV-VAE data (using DeepAE-generated virtual samples)...")

    # 使用DeepAE生成的虚拟样本路径
    data_dir = r"D:\Oswaldo's surf project\My Database"
    virtual_samples_path = os.path.join(data_dir, "virtual_samples_100000.npy")

    # 检查文件是否存在
    if not os.path.exists(virtual_samples_path):
        print(f"❌ DeepAE virtual samples file not found: {virtual_samples_path}")
        print("Please run DeepAE_realize.py first to generate the data")
        return None, None, None

    try:
        # Load virtual samples
        X_vae = np.load(virtual_samples_path)

        # Check data shape
        print(f"✅ Successfully loaded DeepAE virtual samples:")
        print(f"  Original shape: {X_vae.shape}")
        print(f"  Number of samples: {X_vae.shape[0]:,}")
        print(f"  Timesteps: {X_vae.shape[1]} hours")
        print(f"  Number of features: {X_vae.shape[2]}")
        print(f"  Data source: DeepAE virtual samples")
        print(f"  Using the exact same 100000 samples as DeepAE/LSTM/CONV-AE")

        # Load DeepAE feature mapping
        feature_mapping_path = os.path.join(data_dir, "Merge_and_DAEfeature_mapping.json")
        if os.path.exists(feature_mapping_path):
            with open(feature_mapping_path, 'r', encoding='utf-8') as f:
                feature_mapping = json.load(f)
            print(f"✅ Successfully loaded DeepAE feature mapping")
        else:
            # Create simple feature mapping
            feature_mapping = {
                "hour_feature_names": [f"feature_{i}" for i in range(X_vae.shape[2])],
                "note": "Auto-generated feature mapping",
                "data_source": "DeepAE virtual samples"
            }
            print("⚠️  Using auto-generated feature mapping")

        # 创建CONV-VAE配置
        vae_config = {
            "n_samples": X_vae.shape[0],
            "sequence_hours": X_vae.shape[1],
            "n_features": X_vae.shape[2],
            "data_source": "DeepAE_virtual_samples",
            "description": "CONV-VAE uses 35,040 virtual 24-hour samples generated by DeepAE",
            "note": "Uses the exact same 35,040 samples as other models for fair comparison"
        }

        return X_vae, vae_config, feature_mapping

    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None, None, None


# ====================== 3. 构建Conv-VAE模型（使用自定义损失层） ======================
def build_conv_vae(sequence_length=24, n_features=11, latent_dim=16):
    """
    构建一维卷积变分自编码器 (Conv1D VAE)
    """
    print(f"\n🔧 create CONV-VAE model:")
    print(f"  Input shape: ({sequence_length}, {n_features})")
    print(f"  Latent space dimension: {latent_dim}")

    # ====== 编码器 ======
    encoder_inputs = Input(shape=(sequence_length, n_features), name='encoder_input')

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

    # 展平并输出均值和方差
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)

    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # 重参数化
    z = Sampling()([z_mean, z_log_var])

    # 编码器模型
    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')

    # ====== 解码器 ======
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')

    # 计算重塑后的形状
    conv_shape = (sequence_length // 8, 128)  # 经过3次池化 (24/2/2/2=3)

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

    decoder_outputs = Conv1D(n_features, kernel_size=5, activation='linear', padding='same')(x)

    # 解码器模型
    decoder = Model(latent_inputs, decoder_outputs, name='decoder')

    # ====== 完整VAE模型 ======
    # 获取编码器的输出
    z_mean, z_log_var, z = encoder(encoder_inputs)
    outputs = decoder(z)

    # 使用自定义损失层计算损失
    final_outputs = VAELossLayer()([encoder_inputs, outputs, z_mean, z_log_var])

    # 创建VAE模型
    vae = Model(encoder_inputs, final_outputs, name='vae')

    # 编译模型（不需要指定损失函数，因为已经在VAELossLayer中添加了）
    vae.compile(optimizer=Adam(learning_rate=0.001),
                metrics=['mse'])
    # 添加MSE作为额外指标
    # 计算参数
    total_params = vae.count_params()
    trainable_params = np.sum([np.prod(v.shape) for v in vae.trainable_weights])
    non_trainable_params = total_params - trainable_params

    print(f"\n📊 model parameters count:")
    print(f"  total parameters: {total_params:,}")
    print(f"  trainable parameters: {trainable_params:,}")
    print(f"  Non-trainable parameters: {non_trainable_params:,}")
    print(f"  Target parameters: ~9,444 (Similar to the Conv-VAE in the paper)")

    return vae, encoder, decoder, total_params


# ====================== 4. 数据准备函数 ======================
def prepare_vae_train_val_split(X_vae, train_ratio=0.85, val_ratio=0.15, shuffle=True):
    """
    为CONV-VAE准备训练和验证数据
    """
    n_samples = X_vae.shape[0]

    if shuffle:
        # 随机打乱
        indices = np.random.permutation(n_samples)
        X_shuffled = X_vae[indices]
    else:
        X_shuffled = X_vae

    # 划分训练集、验证集
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)

    X_train = X_shuffled[:train_end]
    X_val = X_shuffled[train_end:val_end]

    print(f"\n📊 CONV-VAE dataset splitting:")
    print(f"  Training Set: {X_train.shape[0]:,} samples ({train_ratio * 100:.1f}%)")
    print(f"  Validation Set: {X_val.shape[0]:,} samples ({val_ratio * 100:.1f}%)")
    print(f"  Total Samples: {n_samples:,}")

    return X_train, X_val


# ====================== 5. 模型训练 ======================
def train_conv_vae(X_train, X_val, sequence_length, n_features, epochs=80):
    """
    训练CONV-VAE
    """
    print(f"\n🚀 start training CONV-VAE...")

    # 构建模型
    vae, encoder, decoder, total_params = build_conv_vae(
        sequence_length=sequence_length,
        n_features=n_features,
        latent_dim=16  # 潜在空间维度
    )

    # 回调函数
    callbacks = [
        # EarlyStopping(
        #     monitor='val_loss',
        #     patience=15,
        #     restore_best_weights=True,
        #     verbose=1
        # ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # 训练模型
    print(f"\n📈 Training Details:")
    print(f"  Batch Size: 128")
    print(f"  Max Epochs: {epochs}")

    history = vae.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=epochs,
        batch_size=128,
        # callbacks=callbacks,
        verbose=1
    )

    return vae, encoder, decoder, history, total_params


# ====================== 6. 评估和异常检测 ======================
# ====================== 6. 评估和异常检测 ======================
def evaluate_vae_anomalies(vae, encoder, X_val, X_test=None, percentile=95):
    """
    评估CONV-VAE模型并检测异常
    """
    print("\n📈 CONV-VAE Model Evaluation Results:")

    # 在验证集上评估
    val_loss = vae.evaluate(X_val, X_val, verbose=0)

    # 获取重建损失和KL损失
    if isinstance(val_loss, list):
        val_total_loss = val_loss[0]
        val_reconstruction_loss = val_loss[1] if len(val_loss) > 1 else None
        val_kl_loss = val_loss[2] if len(val_loss) > 2 else None
    else:
        val_total_loss = val_loss
        val_reconstruction_loss = None
        val_kl_loss = None

    # 主要显示总损失（与LSTM的MSE对应）
    print(f"  Validation loss (MSE): {val_total_loss:.6f}")
    if val_reconstruction_loss:
        print(f"  Validation reconstruction loss: {val_reconstruction_loss:.6f}")
    if val_kl_loss:
        print(f"  Validation KL divergence loss: {val_kl_loss:.6f}")

    # 计算验证集重建误差
    val_reconstructed = vae.predict(X_val, verbose=0)

    # 计算每个样本的平均MSE（在时间步和特征维度上平均）
    val_errors = np.mean((X_val - val_reconstructed) ** 2, axis=(1, 2))
    val_mae = np.mean(np.abs(X_val - val_reconstructed), axis=(1, 2)).mean()

    print(f"  Validation MAE: {val_mae:.6f}")

    # Calculate threshold (95th percentile)
    threshold = np.percentile(val_errors, percentile)
    print(f"  Validation set reconstruction error {percentile}th percentile threshold: {threshold:.6f}")

    # ✅ 关键修改：添加验证集异常检测输出（与LSTM保持一致）
    val_anomalies = val_errors > threshold
    val_anomalies_count = np.sum(val_anomalies)
    val_anomaly_ratio = val_anomalies_count / len(X_val) * 100

    print(f"  Detected {val_anomalies_count} anomalous samples out of {len(X_val)} total samples")
    print(f"  Anomaly ratio: {val_anomaly_ratio:.2f}%")

    # 如果传入了测试集
    if X_test is not None:
        test_reconstructed = vae.predict(X_test, verbose=0)
        test_errors = np.mean((X_test - test_reconstructed) ** 2, axis=(1, 2))

        anomalies = test_errors > threshold
        anomalies_count = np.sum(anomalies)
        anomaly_ratio = anomalies_count / len(X_test) * 100

        return (threshold, val_anomalies_count, val_anomaly_ratio,
                test_errors, test_reconstructed, val_errors, val_mae)
    else:
        # ✅ 返回验证集异常计数和比例
        return threshold, val_anomalies_count, val_anomaly_ratio, val_errors, val_reconstructed, val_errors, val_mae


# ====================== 7. 潜在空间可视化 ======================
def visualize_latent_space(encoder, X_sample, labels=None, n_samples=1000):
    """
    可视化潜在空间分布
    """
    print("\n🔍 Visualizing latent space...")

    if len(X_sample) > n_samples:
        indices = np.random.choice(len(X_sample), n_samples, replace=False)
        X_sample = X_sample[indices]
        if labels is not None:
            labels = labels[indices]

    # 获取潜在编码
    z_mean, z_log_var, z = encoder.predict(X_sample, verbose=0)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. 潜在空间前两个维度
    ax1 = axes[0, 0]
    if labels is not None and len(np.unique(labels)) < 10:
        scatter = ax1.scatter(z[:, 0], z[:, 1], c=labels, cmap='tab10', alpha=0.7)
        plt.colorbar(scatter, ax=ax1)
    else:
        ax1.scatter(z[:, 0], z[:, 1], alpha=0.7)
    ax1.set_title('Latent Space (First 2 Dimensions)', fontsize=12)
    ax1.set_xlabel('z[0]')
    ax1.set_ylabel('z[1]')
    ax1.grid(True, alpha=0.3)

    # 2. 潜在变量分布直方图
    ax2 = axes[0, 1]
    for i in range(min(5, z.shape[1])):
        ax2.hist(z[:, i], bins=30, alpha=0.5, label=f'z[{i}]')
    ax2.set_title('Latent Variable Distributions', fontsize=12)
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Frequency')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3. 均值与方差关系
    ax3 = axes[0, 2]
    scatter = ax3.scatter(z_mean[:, 0], np.exp(z_log_var[:, 0]), alpha=0.7)
    ax3.set_title('Mean vs Variance (First Dimension)', fontsize=12)
    ax3.set_xlabel('z_mean[0]')
    ax3.set_ylabel('z_variance[0]')
    ax3.grid(True, alpha=0.3)

    # 4. 协方差矩阵
    ax4 = axes[1, 0]
    cov_matrix = np.cov(z.T)
    im = ax4.imshow(cov_matrix, cmap='viridis', aspect='auto')
    ax4.set_title('Latent Space Covariance Matrix', fontsize=12)
    ax4.set_xlabel('Latent Dimension')
    ax4.set_ylabel('Latent Dimension')
    plt.colorbar(im, ax=ax4)

    # 5. 潜在空间PCA降维
    from sklearn.decomposition import PCA
    ax5 = axes[1, 1]
    if z.shape[1] > 2:
        pca = PCA(n_components=2)
        z_pca = pca.fit_transform(z)
        ax5.scatter(z_pca[:, 0], z_pca[:, 1], alpha=0.7)
        ax5.set_title(f'PCA of Latent Space (Var: {pca.explained_variance_ratio_.sum():.2%})', fontsize=12)
        ax5.set_xlabel('PC1')
        ax5.set_ylabel('PC2')
    else:
        ax5.text(0.5, 0.5, 'Latent dimension <= 2\nNo PCA needed',
                 ha='center', va='center', transform=ax5.transAxes, fontsize=12)
        ax5.set_title('PCA of Latent Space', fontsize=12)
    ax5.grid(True, alpha=0.3)

    # 6. 潜在空间t-SNE降维
    ax6 = axes[1, 2]
    try:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(z) - 1))
        z_tsne = tsne.fit_transform(z[:500])  # t-SNE计算较慢，只取前500个样本
        ax6.scatter(z_tsne[:, 0], z_tsne[:, 1], alpha=0.7)
        ax6.set_title('t-SNE of Latent Space', fontsize=12)
        ax6.set_xlabel('t-SNE1')
        ax6.set_ylabel('t-SNE2')
    except Exception as e:
        ax6.text(0.5, 0.5, f't-SNE failed:\n{e}',
                 ha='center', va='center', transform=ax6.transAxes, fontsize=10)
        ax6.set_title('t-SNE of Latent Space', fontsize=12)
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ====================== 8. 可视化函数 ======================
def visualize_vae_results(history, X_original, X_reconstructed, errors, threshold,
                          encoder, decoder, feature_mapping, total_params,model_name="CONV-VAE"):
    """
    可视化CONV-VAE训练结果和异常检测
    """
    print(f"\n📊 Visualize CONV-VAE results...")

    # 设置正确的保存路径
    base_dir = r"D:\Oswaldo's surf project\My Database"
    os.makedirs(base_dir, exist_ok=True)

    # 设置绘图风格
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(20, 15))

    # 1. 训练损失变化
    ax1 = plt.subplot(3, 4, 1)
    epochs = range(1, len(history.history['loss']) + 1)
    ax1.plot(epochs, history.history['loss'], label='Training Loss', linewidth=2)
    ax1.plot(epochs, history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_title(f'{model_name} - Training History', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Total Loss', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 重建损失变化
    ax2 = plt.subplot(3, 4, 2)
    if 'reconstruction_loss' in history.history:
        ax2.plot(epochs, history.history['reconstruction_loss'],
                 label='Training Recon Loss', linewidth=2, color='orange')
    if 'val_reconstruction_loss' in history.history:
        ax2.plot(epochs, history.history['val_reconstruction_loss'],
                 label='Validation Recon Loss', linewidth=2, color='red')
    ax2.set_title('Reconstruction Loss History', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Reconstruction Loss', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. KL散度损失变化
    ax3 = plt.subplot(3, 4, 3)
    if 'kl_loss' in history.history:
        ax3.plot(epochs, history.history['kl_loss'],
                 label='Training KL Loss', linewidth=2, color='green')
    if 'val_kl_loss' in history.history:
        ax3.plot(epochs, history.history['val_kl_loss'],
                 label='Validation KL Loss', linewidth=2, color='purple')
    ax3.set_title('KL Divergence Loss History', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('KL Loss', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 重建误差分布
    ax4 = plt.subplot(3, 4, 4)
    n_bins = min(100, len(errors) // 10)
    ax4.hist(errors, bins=n_bins, alpha=0.7, color='skyblue', edgecolor='black', density=True)
    ax4.axvline(x=threshold, color='red', linestyle='--', linewidth=2,
                label=f'Threshold ({threshold:.4f})')

    # 添加高斯分布拟合
    from scipy.stats import norm
    mu, std = norm.fit(errors)
    xmin, xmax = ax4.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    ax4.plot(x, p, 'k', linewidth=2, label=f'Normal fit: μ={mu:.4f}, σ={std:.4f}')

    ax4.set_title('Reconstruction Error Distribution', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Reconstruction Error (MSE)', fontsize=12)
    ax4.set_ylabel('Density', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. 重建误差时间序列
    ax5 = plt.subplot(3, 4, 5)
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
    ax5.legend(loc='upper right', fontsize=9)
    ax5.grid(True, alpha=0.3)

    # 6. 潜在空间可视化
    ax6 = plt.subplot(3, 4, 6)
    if len(X_original) > 100:
        # 获取潜在编码
        z_mean, z_log_var, z = encoder.predict(X_original[:500], verbose=0)

        if z.shape[1] >= 2:
            # 使用前两个潜在维度
            ax6.scatter(z[:, 0], z[:, 1], alpha=0.5, s=10)
            ax6.set_title('Latent Space (First 2 Dimensions)', fontsize=14, fontweight='bold')
            ax6.set_xlabel('Latent Dimension 1')
            ax6.set_ylabel('Latent Dimension 2')
        else:
            ax6.text(0.5, 0.5, 'Latent dimension < 2\nCannot visualize',
                     ha='center', va='center', transform=ax6.transAxes, fontsize=12)
            ax6.set_title('Latent Space', fontsize=14, fontweight='bold')
    else:
        ax6.text(0.5, 0.5, 'Not enough data\nfor latent space visualization',
                 ha='center', va='center', transform=ax6.transAxes, fontsize=12)
        ax6.set_title('Latent Space', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3)

    # 7. 样本生成可视化
    ax7 = plt.subplot(3, 4, 7)
    if len(X_original) > 0:
        # 生成新样本
        n_generated = 5
        latent_dim = decoder.input_shape[1]

        # 从标准正态分布采样
        z_samples = np.random.normal(0, 1, (n_generated, latent_dim))
        generated_samples = decoder.predict(z_samples, verbose=0)

        # 显示第一个生成样本
        time_steps = range(generated_samples.shape[1])
        for i in range(min(3, n_generated)):
            if i == 0:
                ax7.plot(time_steps, generated_samples[i, :, 0],
                         alpha=0.7, label=f'Generated {i + 1}')
            else:
                ax7.plot(time_steps, generated_samples[i, :, 0], alpha=0.7)

        # 显示一个真实样本作为对比
        ax7.plot(time_steps, X_original[0, :, 0], 'k--', alpha=0.7, linewidth=2, label='Real Sample')

        ax7.set_title('Generated vs Real Samples (Feature 0)', fontsize=14, fontweight='bold')
        ax7.set_xlabel('Time Step', fontsize=12)
        ax7.set_ylabel('Feature Value', fontsize=12)
        ax7.legend(fontsize=9)
        ax7.grid(True, alpha=0.3)

    # 8. 原始vs重建特征对比
    ax8 = plt.subplot(3, 4, 8)
    if len(X_original) > 0:
        sample_idx = 0
        feature_idx = 0

        time_steps = range(X_original[sample_idx].shape[0])
        original_values = X_original[sample_idx][:, feature_idx]
        reconstructed_values = X_reconstructed[sample_idx][:, feature_idx]

        ax8.plot(time_steps, original_values, label='Original', linewidth=2, alpha=0.8)
        ax8.plot(time_steps, reconstructed_values, label='Reconstructed', linewidth=2,
                 alpha=0.8, color='red', linestyle='--')

        ax8.set_title(f'Sample {sample_idx} - Feature {feature_idx} Comparison',
                      fontsize=14, fontweight='bold')
        ax8.set_xlabel('Time Step', fontsize=12)
        ax8.set_ylabel('Feature Value', fontsize=12)
        ax8.legend()
        ax8.grid(True, alpha=0.3)

    # 9. 误差箱线图
    ax9 = plt.subplot(3, 4, 9)
    box_data = [errors[~anomalies], errors[anomalies]] if len(anomaly_indices) > 0 else [errors]
    box_labels = ['Normal', 'Anomaly'] if len(anomaly_indices) > 0 else ['All Samples']

    bp = ax9.boxplot(box_data, labels=box_labels, patch_artist=True)

    # 设置颜色
    colors = ['lightblue', 'lightcoral'] if len(anomaly_indices) > 0 else ['lightblue']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax9.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5)
    ax9.set_title('Error Distribution by Category', fontsize=14, fontweight='bold')
    ax9.set_ylabel('Reconstruction Error', fontsize=12)
    ax9.grid(True, alpha=0.3, axis='y')

    # 10. 学习率变化
    ax10 = plt.subplot(3, 4, 10)
    if 'lr' in history.history:
        ax10.plot(epochs, history.history['lr'], linewidth=2, color='green')
        ax10.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax10.set_xlabel('Epoch', fontsize=12)
        ax10.set_ylabel('Learning Rate', fontsize=12)
        ax10.grid(True, alpha=0.3)
    else:
        ax10.text(0.5, 0.5, 'Learning Rate\nData Not Available',
                  ha='center', va='center', transform=ax10.transAxes, fontsize=12)
        ax10.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')

    # 11. 模型参数对比
    ax11 = plt.subplot(3, 4, 11)
    # 这里需要从历史结果获取其他模型的参数
    # 暂时使用占位值
    models = ['DeepAE', 'LSTM-AE', 'CONV-AE', 'CONV-VAE']
    params = [29769, 1200000, 7892, total_params]  # 需要从实际结果更新
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'gold']

    bars = ax11.bar(models, params, color=colors, alpha=0.8)
    ax11.set_title('Model Parameter Comparison', fontsize=14, fontweight='bold')
    ax11.set_ylabel('Number of Parameters', fontsize=12)
    ax11.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for bar, param in zip(bars, params):
        height = bar.get_height()
        ax11.text(bar.get_x() + bar.get_width() / 2, height + max(params) * 0.05,
                  f'{param:,}', ha='center', va='bottom', fontsize=9)

    # 12. 异常检测性能
    ax12 = plt.subplot(3, 4, 12)
    if np.sum(errors > threshold) > 0:
        # 计算不同阈值下的性能
        thresholds = np.percentile(errors, range(90, 100))
        anomaly_rates = []

        for t in thresholds:
            anomaly_rate = np.sum(errors > t) / len(errors) * 100
            anomaly_rates.append(anomaly_rate)

        ax12.plot(thresholds, anomaly_rates, 'o-', linewidth=2, markersize=6)
        ax12.axvline(x=threshold, color='red', linestyle='--', linewidth=1.5,
                     label=f'95th percentile ({threshold:.4f})')
        ax12.set_title('Anomaly Rate vs Threshold', fontsize=14, fontweight='bold')
        ax12.set_xlabel('Threshold', fontsize=12)
        ax12.set_ylabel('Anomaly Rate (%)', fontsize=12)
        ax12.legend()
        ax12.grid(True, alpha=0.3)
    else:
        ax12.text(0.5, 0.5, 'No Anomalies Detected\nwith Current Threshold',
                  ha='center', va='center', transform=ax12.transAxes, fontsize=12)
        ax12.set_title('Anomaly Detection Performance', fontsize=14, fontweight='bold')

    plt.tight_layout()

    # 保存主图表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(base_dir, f"conv_vae_visualization_{timestamp}.png")

    try:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📸 CONV-VAE visualization results saved to: {save_path}")
    except Exception as e:
        print(f"⚠️  Failed to save images: {e}")
        alt_path = f"conv_vae_visualization_{timestamp}.png"
        plt.savefig(alt_path, dpi=150, bbox_inches='tight')
        print(f"📸 Saved to current directory: {alt_path}")

    plt.close()

    # 单独保存潜在空间可视化
    if len(X_original) > 100:
        latent_fig = visualize_latent_space(encoder, X_original[:1000])
        latent_path = os.path.join(base_dir, f"conv_vae_latent_space_{timestamp}.png")
        latent_fig.savefig(latent_path, dpi=150, bbox_inches='tight')
        print(f"📸 Latent space visualization saved to: {latent_path}")
        plt.close(latent_fig)

    # 输出特征级分析
    if len(X_original) > 0:
        print("\n📊 CONV-VAE Reconstruction Error Analysis (First Sample):")
        feature_errors = np.mean((X_original[0] - X_reconstructed[0]) ** 2, axis=0)

        if 'hour_feature_names' in feature_mapping:
            hour_feature_names = feature_mapping['hour_feature_names']
            if len(hour_feature_names) == len(feature_errors):
                print("  Feature Reconstruction Error Ranking:")
                sorted_indices = np.argsort(feature_errors)[::-1]
                for i, idx in enumerate(sorted_indices[:5]):
                    print(f"    {i + 1}. {hour_feature_names[idx]}: {feature_errors[idx]:.6f}")


# ====================== 9. 保存模型 ======================
def save_vae_models(vae, encoder, decoder, save_path):
    """
    保存训练好的CONV-VAE模型
    """
    # 确保保存路径存在
    os.makedirs(save_path, exist_ok=True)

    # 保存为HDF5格式
    try:
        vae.save(os.path.join(save_path, "conv_vae_model2.h5"))
        encoder.save(os.path.join(save_path, "conv_vae_encoder2.h5"))
        decoder.save(os.path.join(save_path, "conv_vae_decoder2.h5"))

        print(f"\n💾 CONV-VAE model saved to: {save_path}")
        print(f"  - conv_vae_model2.h5")
        print(f"  - conv_vae_encoder2.h5")
        print(f"  - conv_vae_decoder2.h5")
    except Exception as e:
        print(f"⚠️  Failed to save model: {e}")
        print("Trying to save to current directory...")
        vae.save("conv_vae_model2.h5")
        encoder.save("conv_vae_encoder2.h5")
        decoder.save("conv_vae_decoder2.h5")


# # ====================== 10. 全面模型对比 ======================
# def create_final_comparison(vae_results, other_results=None):
#     """
#     创建最终的全面模型对比分析
#     """
#     print("\n" + "=" * 80)
#     print("📊 最终模型对比分析")
#     print("=" * 80)
#
#     # 默认值（如果没有提供其他模型结果）
#     if other_results is None:
#         other_results = {
#             'deepae': {'params': 29769, 'mae': 0.1785, 'anomaly_rate': 4.56},
#             'lstm': {'params': 1200000, 'mae': 0.2065, 'anomaly_rate': 4.61},
#             'conv': {'params': 7892, 'mae': 0.388, 'anomaly_rate': 6.00}
#         }
#
#     # 创建对比数据
#     comparison_data = {
#         'Model': ['DeepAE', 'LSTM-AE', 'CONV-AE', 'CONV-VAE'],
#         'Params': [f"{other_results['deepae']['params']:,}",
#                    f"{other_results['lstm']['params']:,}",
#                    f"{other_results['conv']['params']:,}",
#                    f"{vae_results['total_params']:,}"],
#         'Val_MAE': [f"{other_results['deepae']['mae']:.4f}",
#                     f"{other_results['lstm']['mae']:.4f}",
#                     f"{other_results['conv']['mae']:.4f}",
#                     f"{vae_results.get('val_mae', 0):.4f}"],
#         'Anomaly_Rate': [f"{other_results['deepae']['anomaly_rate']:.2f}%",
#                          f"{other_results['lstm']['anomaly_rate']:.2f}%",
#                          f"{other_results['conv']['anomaly_rate']:.2f}%",
#                          f"{vae_results.get('all_anomaly_ratio', 0):.2f}%"],
#         'Architecture': ['Dense Layers', 'LSTM Layers', 'Conv1D AE', 'Conv1D VAE'],
#         'Latent_Space': ['Deterministic', 'Deterministic', 'Deterministic', 'Probabilistic']
#     }
#
#     # 打印对比表格
#     print(f"\n{'模型':<12} {'参数':<15} {'验证MAE':<12} {'异常率':<12} {'架构':<15} {'潜在空间':<15}")
#     print("-" * 81)
#
#     for i in range(len(comparison_data['Model'])):
#         print(f"{comparison_data['Model'][i]:<12} "
#               f"{comparison_data['Params'][i]:<15} "
#               f"{comparison_data['Val_MAE'][i]:<12} "
#               f"{comparison_data['Anomaly_Rate'][i]:<12} "
#               f"{comparison_data['Architecture'][i]:<15} "
#               f"{comparison_data['Latent_Space'][i]:<15}")
#
#     print("-" * 81)
#
#     # 计算VAE相对于论文的性能
#     print(f"\n📈 CONV-VAE性能分析:")
#     print(f"  论文中CONV-VAE MAE: ~0.387")
#     print(f"  我们的CONV-VAE MAE: {vae_results.get('val_mae', 0):.4f}")
#     print(f"  差异: {vae_results.get('val_mae', 0) - 0.387:+.4f}")
#
#     print(f"\n  论文中CONV-VAE异常率: ~6.07%")
#     print(f"  我们的CONV-VAE异常率: {vae_results.get('all_anomaly_ratio', 0):.2f}%")
#     print(f"  差异: {vae_results.get('all_anomaly_ratio', 0) - 6.07:+.2f}%")
#
#     # 保存为CSV
#     df = pd.DataFrame(comparison_data)
#     save_path = r"D:\Oswaldo's surf project\My Database\final_model_comparison.csv"
#     df.to_csv(save_path, index=False, encoding='utf-8-sig')
#     print(f"\n💾 最终对比结果已保存为CSV: {save_path}")
#
#     return df


# ====================== 11. 主流程 ======================
def main():
    print("=" * 80)
    print("CONVOLUTIONAL VARIATIONAL AUTOENCODER (CONV-VAE) Training Pipeline")
    print("=" * 80)

    # 配置路径
    base_path = r"D:\Oswaldo's surf project\My Database"
    model_save_path = os.path.join(base_path, "models\\")

    # 确保目录存在
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(model_save_path, exist_ok=True)

    # 1. 加载CONV-VAE数据（使用DeepAE虚拟样本）
    X_vae, vae_config, feature_mapping = load_conv_vae_data()
    if X_vae is None:
        print("❌ Failed to load CONV-VAE data, exiting process")
        return

    # 2. 准备训练数据
    X_train, X_val = prepare_vae_train_val_split(
        X_vae,
        train_ratio=0.85,
        val_ratio=0.15,
        shuffle=True
    )

    # 3. 训练CONV-VAE
    sequence_length = X_vae.shape[1]
    n_features = X_vae.shape[2]

    vae, encoder, decoder, history, total_params = train_conv_vae(
        X_train, X_val,
        sequence_length=sequence_length,
        n_features=n_features,
        epochs=80
    )

    # 4. 在验证集上评估
    print("\n🔍 Evaluating CONV-VAE model on validation set:")
    threshold, val_anomalies_count, val_anomaly_ratio, val_errors, val_reconstructed, _, val_mae = evaluate_vae_anomalies(
        vae, encoder, X_val, X_val, percentile=95
    )

    # 5. 在整个数据集上评估
    print("\n🔍 Evaluating on the complete CONV-VAE dataset:")
    X_all = X_vae

    all_reconstructed = vae.predict(X_all, verbose=0)
    all_errors = np.mean((X_all - all_reconstructed) ** 2, axis=(1, 2))

    # 检测异常
    all_anomalies = all_errors > threshold
    all_anomalies_count = np.sum(all_anomalies)
    all_anomaly_ratio = all_anomalies_count / len(X_all) * 100

    print(f"  Total samples: {len(X_all):,}")
    print(f"  Anomalies detected: {all_anomalies_count:,}")
    print(f"  Anomaly proportion: {all_anomaly_ratio:.2f}%")

    # 6. 可视化结果
    print("\n📊 Generating visualization for all CONV-VAE samples:")
    visualize_vae_results(
        history, X_all, all_reconstructed, all_errors, threshold,
        encoder, decoder, feature_mapping, total_params,model_name="CONV-VAE (100000 samples)"
    )

    # 7. 保存模型
    save_vae_models(vae, encoder, decoder, model_save_path)

    # ✅ 新增：输出模型性能总结（与LSTM一致）
    print("\n" + "=" * 80)
    print("📋 CONV-VAE Training Completion Report")
    print("=" * 80)

    print(f"\n📊 CONV-VAE Model Performance Summary:")
    print(f"  Total samples: {X_vae.shape[0]:,}")
    print(f"  Actual training samples used: {X_train.shape[0]:,}")
    print(f"  Actual validation samples used: {X_val.shape[0]:,}")
    print(f"  Total params: {total_params:,}")
    print(f"  Validation loss (Total): {history.history['val_loss'][-1]:.6f}")

    if 'val_reconstruction_loss' in history.history:
        print(f"  Validation reconstruction loss: {history.history['val_reconstruction_loss'][-1]:.6f}")
    if 'val_kl_loss' in history.history:
        print(f"  Validation KL divergence loss: {history.history['val_kl_loss'][-1]:.6f}")

    print(f"  Validation MAE: {val_mae:.6f}")
    print(f"  95th percentile threshold (validation): {threshold:.6f}")
    print(f"  Anomaly detection (all samples): {all_anomalies_count:,} / {len(X_all):,} ({all_anomaly_ratio:.2f}%)")

    # 8. 生成最终对比分析
    vae_results = {
        'total_params': total_params,
        'val_mae': val_mae,
        'val_loss': history.history['val_loss'][-1],
        'all_anomaly_ratio': all_anomaly_ratio,
        'all_anomalies_count': all_anomalies_count,
        'threshold': threshold
    }

    # 创建最终对比
    # create_final_comparison(vae_results)

    # 9. 保存详细结果
    results = {
        'model_name': 'CONV_VAE',
        'total_params': int(total_params),
        'n_samples': X_vae.shape[0],
        'n_train_samples': X_train.shape[0],
        'n_val_samples': X_val.shape[0],
        'sequence_length': int(sequence_length),
        'n_features': int(n_features),
        'latent_dim': 16,
        'validation_threshold': float(threshold),
        'validation_anomalies': int(val_anomalies_count),
        'all_anomalies': int(all_anomalies_count),
        'anomaly_ratio_all': float(all_anomaly_ratio),
        'anomaly_ratio_val': float(val_anomaly_ratio),
        'final_train_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
        'final_train_reconstruction_loss': float(history.history.get('reconstruction_loss', [0])[-1]),
        'final_val_reconstruction_loss': float(history.history.get('val_reconstruction_loss', [0])[-1]),
        'final_train_kl_loss': float(history.history.get('kl_loss', [0])[-1]),
        'final_val_kl_loss': float(history.history.get('val_kl_loss', [0])[-1]),
        'val_mae': float(val_mae),
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'architecture': 'Conv1D Variational AutoEncoder',
        'data_source': 'DeepAE virtual samples',
        'comparison_with_paper': {
            'paper_conv_vae_mae': 0.387,
            'paper_conv_vae_anomaly_rate': 6.07,
            'our_mae': float(val_mae),
            'our_anomaly_rate': float(all_anomaly_ratio),
            'mae_difference': float(val_mae - 0.387),
            'anomaly_rate_difference': float(all_anomaly_ratio - 6.07)
        }
    }

    results_path = os.path.join(base_path, 'conv_vae_training_results2.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n💾 CONV-VAE training results saved to: {results_path}")
    print("\n🎉 CONV-VAE training pipeline completed!")


# ====================== 12. 运行主流程 ======================
if __name__ == "__main__":
    # 设置TensorFlow日志级别
    tf.get_logger().setLevel('ERROR')

    # 设置matplotlib中文字体
    # plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 运行主流程
    main()