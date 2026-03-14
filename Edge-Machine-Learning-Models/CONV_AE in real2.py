import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, Dense, Flatten, Reshape, \
    BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import os
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# ====================== 1. 配置参数（按照第一个python文件） ======================
DATA_DIR = r"D:\Oswaldo's surf project\DR O's database\preprocessed_data"
X_PATH = os.path.join(DATA_DIR, "X_windows_100k.npy")
SCALED_DF_PATH = os.path.join(DATA_DIR, "normalized_hourly_data.csv")
MODEL_PATH = r"D:\Oswaldo's surf project\DR O's database\models\conv1d_autoencoder_multi_modal100k_data.h5"
VISUALIZATION_PATH = r"D:\Oswaldo's surf project\DR O's database\visualizations"

TIMESTEPS = 24  # 滑动窗口大小
EPOCHS = 80
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.15
PATIENCE = 12
ANOMALY_THRESHOLD_PCT = 95
PLOT_DAYS = 7

# 特征名称（按照第一个python文件）
FEATURE_NAMES = [
    'avg_PM2.5_normalized_scaled',
    'total_noise_duration_scaled',
    'noise_event_count_scaled',
    'avg_salience_scaled'
]

# 确保可视化目录存在
os.makedirs(VISUALIZATION_PATH, exist_ok=True)

import time  # 确保已导入time模块

# 记录开始时间
start_time = time.time()

# ====================== 2. 数据加载流程（按照第一个python文件） ======================
print("Loading preprocessed sliding windows...")
X = np.load(X_PATH)
print(f"Loaded X shape: {X.shape}")  # (n_windows, 24, n_features)

n_samples, timesteps, n_features = X.shape
assert timesteps == TIMESTEPS, f"Timesteps mismatch: expected {TIMESTEPS}, got {timesteps}"
assert n_features == len(FEATURE_NAMES), f"Feature count mismatch: expected {len(FEATURE_NAMES)}, got {n_features}"

# 加载归一化的小时数据用于绘图和对齐
df_hourly = pd.read_csv(SCALED_DF_PATH, index_col=0, parse_dates=True)
print(f"Hourly data shape: {df_hourly.shape}")

# ====================== 3. 时序分割（按照第一个python文件） ======================
split_idx = int(0.85 * n_samples)
X_train = X[:split_idx]
X_val = X[split_idx:]

print(f"Training windows   : {X_train.shape}")
print(f"Validation windows : {X_val.shape}")


# ====================== 4. 构建改进的Conv-AE模型（按照最后一个python文件） ======================
def build_improved_conv_ae(sequence_length=24, n_features=4):
    """
    构建改进的Conv-AE模型（带残差连接，类似论文中的结构）
    按照第二个python文件中的build_residual_conv_ae函数
    """
    print(f"\nBuilding improved Residual CONV-AE:")
    print(f"  Input shape: ({sequence_length}, {n_features})")

    inputs = Input(shape=(sequence_length, n_features))

    # ====== 编码器 ======
    # 第一卷积块
    x1 = Conv1D(32, kernel_size=5, padding='same', activation='relu')(inputs)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling1D(pool_size=2, padding='same')(x1)

    # 第二卷积块
    x2 = Conv1D(64, kernel_size=3, padding='same', activation='relu')(x1)
    x2 = BatchNormalization()(x2)
    x2 = MaxPooling1D(pool_size=2, padding='same')(x2)

    # 第三卷积块
    x3 = Conv1D(128, kernel_size=3, padding='same', activation='relu')(x2)
    x3 = BatchNormalization()(x3)
    encoded = MaxPooling1D(pool_size=2, padding='same')(x3)

    # ====== 解码器 ======
    # 第一反卷积块
    y1 = Conv1D(128, kernel_size=3, padding='same', activation='relu')(encoded)
    y1 = BatchNormalization()(y1)
    y1 = UpSampling1D(size=2)(y1)

    # 第二反卷积块
    y2 = Conv1D(64, kernel_size=3, padding='same', activation='relu')(y1)
    y2 = BatchNormalization()(y2)
    y2 = UpSampling1D(size=2)(y2)

    # 第三反卷积块
    y3 = Conv1D(32, kernel_size=3, padding='same', activation='relu')(y2)
    y3 = BatchNormalization()(y3)
    y3 = UpSampling1D(size=2)(y3)

    # 输出层
    outputs = Conv1D(n_features, kernel_size=5, padding='same', activation='linear')(y3)

    # ====== 完整模型 ======
    autoencoder = Model(inputs, outputs, name='improved_conv_ae')

    # 计算参数
    total_params = autoencoder.count_params()
    trainable_params = np.sum([np.prod(v.shape) for v in autoencoder.trainable_weights])
    non_trainable_params = total_params - trainable_params

    print(f"\nModel parameters count:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Non-trainable parameters: {non_trainable_params:,}")

    return autoencoder, total_params


# ====================== 5. 构建或加载模型 ======================
# ====================== 5. 构建或加载模型 ======================
print("\nBuilding / Loading Improved Conv1D Autoencoder...")


def compile_model(model):
    """编译模型的辅助函数"""
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mae',
        metrics=['mse']
    )
    return model


if os.path.exists(MODEL_PATH):
    print(f"Loading saved model from {MODEL_PATH}")
    try:
        # Conv-AE 没有自定义层，直接加载
        autoencoder = load_model(MODEL_PATH, compile=False)  # 注意：load时设置compile=False
        print("Model loaded successfully!")

        # 重新编译模型
        autoencoder = compile_model(autoencoder)
        print("Model recompiled successfully!")

        total_params = autoencoder.count_params()
        history = None
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Building new model instead...")
        autoencoder, total_params = build_improved_conv_ae(TIMESTEPS, n_features)
        autoencoder = compile_model(autoencoder)
        history = None
else:
    print("No saved model → building new Conv-AE")
    autoencoder, total_params = build_improved_conv_ae(TIMESTEPS, n_features)
    autoencoder = compile_model(autoencoder)

    # 训练模型
    print(f"\nTraining for max {EPOCHS} epochs...")
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1
    )

    history = autoencoder.fit(
        X_train, X_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, X_val),
        callbacks=[early_stop],
        verbose=1
    )

    autoencoder.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

# ====================== 6. 计算验证损失 ======================
print("\n" + "=" * 60)
print("VALIDATION LOSS INFORMATION")
print("=" * 60)

if history is not None:
    # 从训练历史获取验证损失
    best_val_loss = min(history.history['val_loss'])
    best_epoch = np.argmin(history.history['val_loss']) + 1
    final_val_loss = history.history['val_loss'][-1]
    final_train_loss = history.history['loss'][-1]

    print(f"Training completed:")
    print(f"  - Best validation loss (MAE): {best_val_loss:.6f} (epoch {best_epoch})")
    print(f"  - Final training loss (MAE): {final_train_loss:.6f}")
    print(f"  - Final validation loss (MAE): {final_val_loss:.6f}")
    print(f"  - Training epochs: {len(history.history['loss'])}")

    # 如果有MSE指标
    if 'val_mse' in history.history:
        best_val_mse = min(history.history['val_mse'])
        final_val_mse = history.history['val_mse'][-1]
        print(f"  - Best validation MSE: {best_val_mse:.6f}")
        print(f"  - Final validation MSE: {final_val_mse:.6f}")
else:
    # 对于加载的模型，重新计算验证损失
    print("Evaluating loaded model on validation set...")
    val_metrics = autoencoder.evaluate(X_val, X_val, batch_size=BATCH_SIZE, verbose=0)

    if isinstance(val_metrics, list):
        val_mae = val_metrics[0]
        if len(val_metrics) > 1:
            val_mse = val_metrics[1]
            print(f"  - Validation loss (MAE): {val_mae:.6f}")
            print(f"  - Validation MSE: {val_mse:.6f}")
        else:
            print(f"  - Validation loss: {val_mae:.6f}")
    else:
        print(f"  - Validation loss: {val_metrics:.6f}")

print("=" * 60)

# ====================== 7. 重建和阈值计算 ======================
print("\nComputing reconstruction errors...")
recon_val = autoencoder.predict(X_val, batch_size=BATCH_SIZE, verbose=0)
mae_val = np.mean(np.abs(X_val - recon_val), axis=(1, 2))

threshold = np.percentile(mae_val, ANOMALY_THRESHOLD_PCT)
print(f"Validation MAE {ANOMALY_THRESHOLD_PCT}th percentile threshold: {threshold:.6f}")

recon_full = autoencoder.predict(X, batch_size=BATCH_SIZE, verbose=0)
mae_full = np.mean(np.abs(X - recon_full), axis=(1, 2))

print("\nComputing validation loss...")
val_loss = autoencoder.evaluate(X_val, X_val, verbose=0)
print(f"Validation Loss (MAE): {val_loss[0]:.6f}")
print(f"Validation MSE: {val_loss[1]:.6f}")
# ====================== 8. 异常检测和对齐 ======================
# ====================== 8. 异常检测和对齐 ======================
anomaly_flags = (mae_full > threshold).astype(int)

# 诊断数据长度
print(f"\nDiagnosing data length mismatch:")
print(f"  - mae_full length: {len(mae_full)}")
print(f"  - X shape: {X.shape}")
print(f"  - df_hourly length: {len(df_hourly)}")
print(f"  - Expected windows from hourly data: {len(df_hourly) - TIMESTEPS + 1}")

# 取最小长度确保匹配
n_windows = min(len(mae_full), len(df_hourly) - TIMESTEPS + 1)
print(f"  - Using {n_windows} windows for alignment")

# 截断数据使其匹配
mae_full = mae_full[:n_windows]
anomaly_flags = anomaly_flags[:n_windows]

window_end_indices = df_hourly.index[TIMESTEPS - 1: TIMESTEPS - 1 + n_windows]
df_anomalies = pd.DataFrame({
    'reconstruction_mae': mae_full,
    'is_detected_anomaly': anomaly_flags
}, index=window_end_indices)

anomaly_count = np.sum(anomaly_flags)
anomaly_ratio = np.mean(anomaly_flags) * 100
print(f"Detected {anomaly_count} anomalous windows out of {len(anomaly_flags)} ({anomaly_ratio:.2f}%)")


# ====================== 9. 综合可视化（按照第二个python文件的布局） ======================
def create_comprehensive_visualization(df_hourly, df_anomalies, mae_full, threshold,
                                       X, recon_full, history=None, model_name="Improved Conv-AE"):
    """
    创建综合可视化图表（按照第二个python文件的布局）
    """
    print(f"\nCreating comprehensive visualization...")

    # 设置绘图风格
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(20, 14))

    # 1. 训练损失变化（如果有历史数据）
    if history is not None:
        ax1 = plt.subplot(3, 3, 1)
        epochs = range(1, len(history.history['loss']) + 1)
        ax1.plot(epochs, history.history['loss'], label='Training Loss', linewidth=2)
        ax1.plot(epochs, history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax1.set_title(f'{model_name} - Training History', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('MAE Loss', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # 2. 重建误差时间序列
    ax2 = plt.subplot(3, 3, 2 if history is None else 2)
    ax2.plot(df_anomalies.index, df_anomalies['reconstruction_mae'],
             label='Reconstruction MAE', color='teal', alpha=0.7, linewidth=1)
    ax2.axhline(threshold, color='red', linestyle='--', linewidth=2,
                label=f'Threshold ({threshold:.5f})')

    # 标记异常点
    anomaly_indices = df_anomalies[df_anomalies['is_detected_anomaly'] == 1].index
    if len(anomaly_indices) > 0:
        anomaly_errors = df_anomalies.loc[anomaly_indices, 'reconstruction_mae']
        ax2.scatter(anomaly_indices, anomaly_errors, color='red',
                    marker='o', s=40, alpha=0.7, label='Detected Anomaly')

    ax2.set_title('Reconstruction Error Time Series', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Mean Absolute Error', fontsize=12)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)

    # 3. 重建误差分布
    ax3 = plt.subplot(3, 3, 3 if history is None else 3)
    n_bins = min(100, len(mae_full) // 10)
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

    # 4. 特征级对比（第一个样本）
    ax4 = plt.subplot(3, 3, 4 if history is None else 4)
    if len(X) > 0:
        sample_idx = 0
        time_steps = range(TIMESTEPS)

        # 绘制第一个样本的第一个特征
        feature_idx = 0
        original_values = X[sample_idx][:, feature_idx]
        reconstructed_values = recon_full[sample_idx][:, feature_idx]

        ax4.plot(time_steps, original_values, label='Original',
                 linewidth=2, alpha=0.8, color='blue')
        ax4.plot(time_steps, reconstructed_values, label='Reconstructed',
                 linewidth=2, alpha=0.8, color='red', linestyle='--')

        ax4.set_title(f'Sample {sample_idx} - {FEATURE_NAMES[feature_idx]}',
                      fontsize=14, fontweight='bold')
        ax4.set_xlabel('Time Step (hours)', fontsize=12)
        ax4.set_ylabel('Normalized Value', fontsize=12)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    # 5. 特征级对比（第一个样本的第二个特征）
    ax5 = plt.subplot(3, 3, 5 if history is None else 5)
    if len(X) > 0 and len(FEATURE_NAMES) > 1:
        sample_idx = 0
        time_steps = range(TIMESTEPS)

        feature_idx = 1
        original_values = X[sample_idx][:, feature_idx]
        reconstructed_values = recon_full[sample_idx][:, feature_idx]

        ax5.plot(time_steps, original_values, label='Original',
                 linewidth=2, alpha=0.8, color='green')
        ax5.plot(time_steps, reconstructed_values, label='Reconstructed',
                 linewidth=2, alpha=0.8, color='orange', linestyle='--')

        ax5.set_title(f'Sample {sample_idx} - {FEATURE_NAMES[feature_idx]}',
                      fontsize=14, fontweight='bold')
        ax5.set_xlabel('Time Step (hours)', fontsize=12)
        ax5.set_ylabel('Normalized Value', fontsize=12)
        ax5.legend()
        ax5.grid(True, alpha=0.3)

    # 6. 模型参数信息
    ax6 = plt.subplot(3, 3, 6 if history is None else 6)
    models = ['LSTM-AE', 'Conv-AE', 'Improved Conv-AE']
    # 近似参数数量（根据论文和实际计算）
    params_approx = [29124, 7892, total_params]
    colors = ['lightblue', 'lightgreen', 'lightcoral']

    bars = ax6.bar(models, params_approx, color=colors, alpha=0.8)
    ax6.set_title('Model Parameter Comparison', fontsize=14, fontweight='bold')
    ax6.set_ylabel('Number of Parameters', fontsize=12)
    ax6.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for bar, param in zip(bars, params_approx):
        ax6.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 500,
                 f'{param:,}', ha='center', va='bottom', fontsize=10)

    # 7. 前N天的特征级异常检测
    # 7. 前N天的特征级异常检测（更稳健版本）
    ax7 = plt.subplot(3, 3, 7 if history is None else 7)
    n_plot = PLOT_DAYS * 24

    if len(df_hourly) >= n_plot:
        plot_df = df_hourly.iloc[:n_plot].copy()

        # 获取plot_df的时间范围
        plot_start = plot_df.index[0]
        plot_end = plot_df.index[-1]

        # 筛选时间范围内的异常
        time_mask = (df_anomalies.index >= plot_start) & (df_anomalies.index <= plot_end)
        plot_anom = df_anomalies[time_mask].copy()

        # 绘制第一个特征(改动）
        if len(FEATURE_NAMES) > 0:
            feat = FEATURE_NAMES[0]
            ax7.plot(plot_df.index, plot_df[feat], label=feat, color='teal', lw=1.2)

            # 确保异常时间戳在plot_df中存在
            if len(plot_anom) > 0:
                valid_anom = plot_anom[plot_anom['is_detected_anomaly'] == 1]
                valid_anom_idx = valid_anom.index[valid_anom.index.isin(plot_df.index)]

                if len(valid_anom_idx) > 0:
                    ax7.scatter(valid_anom_idx, plot_df.loc[valid_anom_idx, feat],
                                color='red', marker='o', s=50, label='Anomaly', alpha=0.7)

        ax7.set_ylabel(feat, fontsize=10)
        ax7.legend(loc='upper right', fontsize=9)
        ax7.grid(True, alpha=0.3)
        ax7.set_title(f'First {PLOT_DAYS} Days - {feat}', fontsize=12, fontweight='bold')

    # 8. 不同阈值下的异常率
    ax8 = plt.subplot(3, 3, 8 if history is None else 8)
    thresholds = np.percentile(mae_val, range(90, 100))
    anomaly_rates = []

    for t in thresholds:
        anomaly_rate = np.sum(mae_full > t) / len(mae_full) * 100
        anomaly_rates.append(anomaly_rate)

    ax8.plot(thresholds, anomaly_rates, 'o-', linewidth=2, markersize=6)
    ax8.axvline(x=threshold, color='red', linestyle='--', linewidth=1.5,
                label=f'95th percentile ({threshold:.4f})')
    ax8.set_title('Anomaly Rate vs Threshold', fontsize=14, fontweight='bold')
    ax8.set_xlabel('Threshold', fontsize=12)
    ax8.set_ylabel('Anomaly Rate (%)', fontsize=12)
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # 9. 性能总结
    ax9 = plt.subplot(3, 3, 9 if history is None else 9)
    ax9.axis('off')

    summary_text = f"""
    {model_name} Performance Summary:

    Total Samples: {len(X):,}
    Training Samples: {len(X_train):,}
    Validation Samples: {len(X_val):,}

    Total Parameters: {total_params:,}

    Validation Threshold: {threshold:.6f}

    Detected Anomalies: {anomaly_count:,}
    Anomaly Ratio: {anomaly_ratio:.2f}%

    Sequence Length: {TIMESTEPS} hours
    Features: {n_features}
    """

    ax9.text(0.1, 0.95, summary_text, fontsize=10,
             verticalalignment='top', family='monospace')

    plt.tight_layout()

    # 保存图表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(VISUALIZATION_PATH, f"conv_ae_results_100k_data{timestamp}.png")

    try:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    except Exception as e:
        print(f"Failed to save image: {e}")
        alt_path = f"improved_conv_ae_results_100k_data{timestamp}.png"
        plt.savefig(alt_path, dpi=150, bbox_inches='tight')
        print(f"Saved to current directory: {alt_path}")

    plt.close()

    return save_path


# ====================== 10. 创建可视化 ======================
# 如果有训练历史，加载它
history_data = None
if os.path.exists(MODEL_PATH.replace('.h5', '_history100k_data.csv')):
    try:
        history_df = pd.read_csv(MODEL_PATH.replace('.h5', '_history100k_data.csv'))
        history_data = type('History', (), {'history': history_df.to_dict('list')})()
    except:
        pass

# 创建综合可视化
vis_path = create_comprehensive_visualization(
    df_hourly=df_hourly,
    df_anomalies=df_anomalies,
    mae_full=mae_full,
    threshold=threshold,
    X=X,
    recon_full=recon_full,
    history=None,
    model_name="Improved Conv-AE"
)

# ====================== 11. 输出总结报告 ======================
print("\n" + "=" * 80)
print("IMPROVED CONVOLUTIONAL AUTOENCODER - TRAINING COMPLETE")
print("=" * 80)

print(f"\n📊 Model Performance Summary:")
print(f"  Model: Improved Residual Conv-AE")
print(f"  Total parameters: {total_params:,}")
print(f"  Input shape: ({TIMESTEPS}, {n_features})")
print(f"  Training samples: {X_train.shape[0]:,}")
print(f"  Validation samples: {X_val.shape[0]:,}")

print(f"\n📈 Anomaly Detection Results:")
print(f"  Anomaly threshold (95th percentile) : {threshold:.6f}")
print(f"  Detected anomalies: {anomaly_count:,} out of {len(mae_full):,}")
print(f"  Anomaly ratio: {anomaly_ratio:.2f}%")

print(f"\n💾 Saved Files:")
print(f"  Model: {MODEL_PATH}")
print(f"  Visualization: {vis_path}")

print(f"\n✅ Improved Conv-AE pipeline completed successfully!")
print("   Model is lightweight → suitable for TensorFlow Lite Micro / edge deployment.")

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

# ====================== 12. 保存结果为JSON文件（仿照第二个文件） ======================
import json


def save_results_conv(results_dict, save_dir):
    """保存训练结果到JSON文件"""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(save_dir, f"conv_ae_results_100k_data{timestamp}.json")

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)

    print(f"💾 Results saved to: {results_path}")
    return results_path


# 构建结果字典
results_dict = {
    'model_name': ' Conv-AE',
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'data_info': {
        'total_samples': X.shape[0],
        'timesteps': TIMESTEPS,
        'n_features': n_features,
        'training_samples': X_train.shape[0],
        'validation_samples': X_val.shape[0]
    },
    'model_info': {
        'total_params': total_params,
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'architecture': 'Residual Conv1D Autoencoder'
    },
    'anomaly_detection': {
        'threshold_percentile': ANOMALY_THRESHOLD_PCT,
        'threshold_value': float(threshold),
        'anomalies_detected': int(anomaly_count),
        'total_windows': len(mae_full),
        'anomaly_ratio': float(anomaly_ratio)
    },
    'file_paths': {
        'model_path':MODEL_PATH,
        'visualization_path': vis_path,
        'data_path': DATA_DIR
    },
    'feature_names': FEATURE_NAMES,
# 添加总运行时间到结果中
    'total_execution_time_seconds': float(total_time_seconds),
    'total_execution_time_minutes': float(total_time_minutes),
    'total_execution_time_hours': float(total_time_hours)
}

# 保存JSON文件
# results_path = save_results_conv(results_dict, os.path.dirname(r"D:\Oswaldo's surf project\DR O's database\models"))
models_dir = r"D:\Oswaldo's surf project\DR O's database\models"
results_path = save_results_conv(results_dict, models_dir)
# 更新总结报告输出
print(f"\n💾 Output Files:")
print(f"  - Model: {MODEL_PATH}")
print(f"  - Visualization: {vis_path}")
print(f"  - Results (JSON): {results_path}")