import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import warnings

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ====================== 配置参数 ======================
# 数据路径 - 修改为新的路径
DATA_DIR = r"D:\Oswaldo's surf project\DR O's database\final_preprocessed_data_complete"
X_PATH = os.path.join(DATA_DIR, "X_windows.npy")
SCALED_DF_PATH = os.path.join(DATA_DIR, "preprocessed_time_series_augmented.csv")

# 模型保存路径 - 修改为新的路径
MODEL_SAVE_DIR = r"D:\Oswaldo's surf project\DR O's database\final_preprocessed_data_complete\models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "lstm_autoencoder_hybrid.h5")

# 可视化保存路径 - 修改为新的路径
VISUALIZATION_SAVE_DIR = r"D:\Oswaldo's surf project\DR O's database\final_preprocessed_data_complete\visualizations"
os.makedirs(VISUALIZATION_SAVE_DIR, exist_ok=True)

# 模型参数
TIMESTEPS = 24
LSTM_UNITS = 128
ENCODING_DIM = 32
EPOCHS = 80
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.15
PATIENCE = 12

# 异常检测参数
ANOMALY_THRESHOLD_PCT = 95
PLOT_DAYS = 7

# 特征名称 - 根据实际数据调整
FEATURE_NAMES = [
    'avg_PM2.5_normalized_scaled',
    'total_noise_duration_scaled',
    'noise_event_count_scaled',
    'avg_salience_scaled'
]


# ====================== 1. 数据加载 ======================
def load_data():
    """加载数据"""
    print("=" * 80)
    print("LSTM-AE HYBRID IMPLEMENTATION")
    print("=" * 80)

    print("\n📂 Loading preprocessed sliding windows...")

    # 加载窗口数据
    if not os.path.exists(X_PATH):
        raise FileNotFoundError(f"X_windows.npy not found at {X_PATH}")

    X = np.load(X_PATH)
    print(f"Loaded X shape: {X.shape}")  # 预期: (n_windows, 24, n_features)

    n_samples, timesteps, n_features = X.shape
    print(f"Data dimensions: {n_samples} samples, {timesteps} timesteps, {n_features} features")

    # 加载每小时数据用于后续分析
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
        n_hours = n_samples + TIMESTEPS - 1
        date_range = pd.date_range(start='2013-03-01', periods=n_hours, freq='H')
        df_hourly = pd.DataFrame(
            np.random.randn(n_hours, n_features),
            index=date_range,
            columns=[f'Feature_{i}' for i in range(n_features)]
        )
        actual_features = df_hourly.columns.tolist()

    return X, df_hourly, actual_features


# ====================== 2. 数据划分 ======================
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


# ====================== 3. 模型构建 ======================
def build_lstm_autoencoder(sequence_length, n_features):
    """构建LSTM Autoencoder"""
    print(f"\n🔧 Building LSTM-AutoEncoder model:")
    print(f"  Input shape: ({sequence_length}, {n_features})")
    print(f"  LSTM units: {LSTM_UNITS}")
    print(f"  Encoding dimension: {ENCODING_DIM}")

    # ====== 编码器 ======
    inputs = Input(shape=(sequence_length, n_features), name='lstm_input')

    # 第一层LSTM
    encoded = LSTM(LSTM_UNITS, activation='tanh', return_sequences=True,
                   name='encoder_lstm1')(inputs)
    encoded = BatchNormalization(name='encoder_bn1')(encoded)
    encoded = Dropout(0.2, name='encoder_dropout1')(encoded)

    # 第二层LSTM
    encoded = LSTM(LSTM_UNITS // 2, activation='tanh', return_sequences=True,
                   name='encoder_lstm2')(encoded)
    encoded = BatchNormalization(name='encoder_bn2')(encoded)
    encoded = Dropout(0.2, name='encoder_dropout2')(encoded)

    # 第三层LSTM（不返回序列）
    encoded = LSTM(LSTM_UNITS // 4, activation='tanh', return_sequences=False,
                   name='encoder_lstm3')(encoded)

    # 瓶颈层
    encoded = Dense(ENCODING_DIM, activation='tanh', name='bottleneck')(encoded)

    # ====== 解码器 ======
    # 重复向量
    decoded = RepeatVector(sequence_length, name='repeat_vector')(encoded)

    # 第一层LSTM
    decoded = LSTM(LSTM_UNITS // 4, activation='tanh', return_sequences=True,
                   name='decoder_lstm1')(decoded)
    decoded = BatchNormalization(name='decoder_bn1')(decoded)
    decoded = Dropout(0.2, name='decoder_dropout1')(decoded)

    # 第二层LSTM
    decoded = LSTM(LSTM_UNITS // 2, activation='tanh', return_sequences=True,
                   name='decoder_lstm2')(decoded)
    decoded = BatchNormalization(name='decoder_bn2')(decoded)
    decoded = Dropout(0.2, name='decoder_dropout2')(decoded)

    # 第三层LSTM
    decoded = LSTM(LSTM_UNITS, activation='tanh', return_sequences=True,
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


# ====================== 4. 模型训练 ======================
def train_model(autoencoder, X_train, X_val):
    """训练模型"""
    print(f"\n🚀 Starting training...")

    # 显示模型摘要
    autoencoder.summary()

    # 计算参数数量
    total_params = autoencoder.count_params()
    print(f"\n📊 Model parameters count:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Parameters size: {total_params * 4 / 1024:.2f} KB (float32)")

    # 编译模型 - 使用MAE损失函数
    autoencoder.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mae',
        metrics=['mse']
    )

    # 回调函数
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
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
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Max Epochs: {EPOCHS}")
    print(f"  Validation split: {VALIDATION_SPLIT}")

    # 训练模型
    history = autoencoder.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    return history, total_params


# ====================== 5. 异常检测 ======================
def detect_anomalies(autoencoder, X, X_val, threshold_percentile=95):
    """异常检测"""
    print("\n🔍 Computing reconstruction errors...")

    # 验证集重建误差
    reconstructions_val = autoencoder.predict(X_val, batch_size=BATCH_SIZE, verbose=0)
    mae_val = np.mean(np.abs(X_val - reconstructions_val), axis=(1, 2))

    # 计算阈值
    threshold = np.percentile(mae_val, threshold_percentile)
    print(f"Validation (assumed normal) MAE {threshold_percentile}th percentile threshold: {threshold:.6f}")

    # 全数据集重建误差
    reconstructions_full = autoencoder.predict(X, batch_size=BATCH_SIZE, verbose=0)
    mae_full = np.mean(np.abs(X - reconstructions_full), axis=(1, 2))

    # 检测异常
    anomaly_flags = (mae_full > threshold).astype(int)
    anomaly_count = np.sum(anomaly_flags)
    anomaly_ratio = anomaly_count / len(mae_full) * 100

    print(f"\nDetected {anomaly_count} anomalous windows out of {len(mae_full)} ({anomaly_ratio:.2f}%)")

    return mae_full, anomaly_flags, threshold, reconstructions_full


# ====================== 6. 可视化 ======================
def visualize_results(df_hourly, mae_full, anomaly_flags, threshold,
                      X_original=None, X_reconstructed=None, history=None,
                      actual_features=None, model_name="LSTM AutoEncoder"):
    """可视化结果"""
    print("\n📊 Generating visualizations...")

    # 设置绘图风格
    plt.style.use('seaborn-v0_8-darkgrid')

    # 创建主图
    fig = plt.figure(figsize=(20, 15))

    # 1. 训练历史
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
    ax2 = plt.subplot(3, 3, 2)

    # 创建时间索引
    if len(df_hourly) >= len(mae_full):
        time_indices = df_hourly.index[TIMESTEPS - 1: TIMESTEPS - 1 + len(mae_full)]
    else:
        time_indices = range(len(mae_full))

    ax2.plot(time_indices, mae_full, label='Reconstruction MAE',
             color='steelblue', alpha=0.7, linewidth=1)
    ax2.axhline(threshold, color='red', linestyle='--',
                label=f'Threshold ({threshold:.5f})', linewidth=2)

    # 标记异常点
    anomaly_indices = np.where(anomaly_flags == 1)[0]
    if len(time_indices) == len(mae_full):
        ax2.scatter(time_indices[anomaly_indices], mae_full[anomaly_indices],
                    color='red', marker='o', s=30, label='Detected Anomaly', alpha=0.6)

    ax2.set_title(f'{model_name} - Reconstruction Error & Detected Anomalies',
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Mean Absolute Error (per window)', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # 3. 重建误差分布
    ax3 = plt.subplot(3, 3, 3)
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
    ax3.plot(x, p, 'k', linewidth=2, label=f'Normal fit: μ={mu:.4f}, σ={std:.4f}')

    ax3.set_title('Reconstruction Error Distribution', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Reconstruction MAE', fontsize=12)
    ax3.set_ylabel('Density', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 原始特征可视化（前7天）
    ax4 = plt.subplot(3, 3, 4)
    n_plot_hours = PLOT_DAYS * 24
    if len(df_hourly) >= n_plot_hours:
        plot_data = df_hourly.iloc[:n_plot_hours]
        if actual_features and len(actual_features) > 0:
            feature_name = actual_features[0]
        else:
            feature_name = 'Feature_0'
        ax4.plot(plot_data.index, plot_data[feature_name],
                 label=feature_name, color='teal', linewidth=1.5)
        ax4.set_title(f'{feature_name} (First {PLOT_DAYS} Days)',
                      fontsize=14, fontweight='bold')
        ax4.set_xlabel('Time', fontsize=12)
        ax4.set_ylabel('Normalized Value', fontsize=12)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    # 5. 多特征对比图
    ax5 = plt.subplot(3, 3, 5)
    if len(df_hourly) >= n_plot_hours and actual_features:
        plot_data = df_hourly.iloc[:n_plot_hours]
        for i, feat in enumerate(actual_features[:4]):  # 最多显示4个特征
            ax5.plot(plot_data.index, plot_data[feat], label=feat,
                     linewidth=1, alpha=0.8)
        ax5.set_title(f'Feature Comparison (First {PLOT_DAYS} Days)',
                      fontsize=14, fontweight='bold')
        ax5.set_xlabel('Time', fontsize=12)
        ax5.set_ylabel('Normalized Value', fontsize=12)
        ax5.legend(loc='upper right', fontsize=9)
        ax5.grid(True, alpha=0.3)

    # 6. 异常检测性能分析
    ax6 = plt.subplot(3, 3, 6)
    thresholds = np.percentile(mae_full, range(90, 100))
    anomaly_rates = []

    for t in thresholds:
        anomaly_rate = np.sum(mae_full > t) / len(mae_full) * 100
        anomaly_rates.append(anomaly_rate)

    ax6.plot(thresholds, anomaly_rates, 'o-', linewidth=2, markersize=6)
    ax6.axvline(x=threshold, color='red', linestyle='--', linewidth=1.5,
                label=f'{ANOMALY_THRESHOLD_PCT}th percentile ({threshold:.4f})')
    ax6.set_title('Anomaly Rate vs Threshold', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Threshold', fontsize=12)
    ax6.set_ylabel('Anomaly Rate (%)', fontsize=12)
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # 7. 原始vs重建对比（第一个样本）
    ax7 = plt.subplot(3, 3, 7)
    if X_original is not None and X_reconstructed is not None and len(X_original) > 0:
        sample_idx = 0
        time_steps = range(TIMESTEPS)

        # 绘制第一个特征
        if actual_features and len(actual_features) > 0:
            feature_name = actual_features[0]
        else:
            feature_name = 'Feature_0'

        ax7.plot(time_steps, X_original[sample_idx, :, 0],
                 label='Original', linewidth=2, alpha=0.8)
        ax7.plot(time_steps, X_reconstructed[sample_idx, :, 0],
                 label='Reconstructed', linewidth=2, alpha=0.8,
                 color='red', linestyle='--')

        ax7.set_title(f'Sample {sample_idx} - {feature_name}',
                      fontsize=14, fontweight='bold')
        ax7.set_xlabel('Time Step (hours)', fontsize=12)
        ax7.set_ylabel('Feature Value', fontsize=12)
        ax7.legend()
        ax7.grid(True, alpha=0.3)

    # 8. 误差箱线图
    ax8 = plt.subplot(3, 3, 8)
    if len(anomaly_indices) > 0:
        normal_errors = mae_full[anomaly_flags == 0]
        anomaly_errors = mae_full[anomaly_flags == 1]

        box_data = [normal_errors, anomaly_errors]
        box_labels = ['Normal', 'Anomaly']

        bp = ax8.boxplot(box_data, labels=box_labels, patch_artist=True)

        colors = ['lightblue', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax8.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5)
        ax8.set_title('Error Distribution by Category', fontsize=14, fontweight='bold')
        ax8.set_ylabel('Reconstruction MAE', fontsize=12)
        ax8.grid(True, alpha=0.3, axis='y')

    # 9. 特征热图（重建误差）
    ax9 = plt.subplot(3, 3, 9)
    if X_original is not None and X_reconstructed is not None and len(X_original) > 0:
        # 计算第一个样本的特征级误差
        sample_idx = 0
        feature_errors = np.abs(X_original[sample_idx] - X_reconstructed[sample_idx])

        im = ax9.imshow(feature_errors.T, aspect='auto', cmap='YlOrRd')
        ax9.set_xlabel('Time Step', fontsize=12)
        ax9.set_ylabel('Feature Index', fontsize=12)
        ax9.set_title(f'Sample {sample_idx} Feature-wise MAE',
                      fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax9, label='MAE')

        # 设置特征标签
        n_features = feature_errors.shape[1]
        if actual_features and len(actual_features) >= n_features:
            feature_labels = [f'F{i + 1}:{actual_features[i][:10]}...' if len(
                actual_features[i]) > 10 else f'F{i + 1}:{actual_features[i]}'
                              for i in range(n_features)]
        else:
            feature_labels = [f'F{i + 1}' for i in range(n_features)]

        ax9.set_yticks(range(n_features))
        ax9.set_yticklabels(feature_labels)

    # 调整布局
    plt.tight_layout()

    # 保存图表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(VISUALIZATION_SAVE_DIR, f"lstm_ae_visualization_{timestamp}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📸 Visualization saved to: {save_path}")

    plt.close(fig)

    return save_path           
# 转换numpy类型为Python原生类型
# results_dict = convert_numpy_types(results_dict)

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
# ====================== 7. 保存结果 ======================
def save_results(results_dict, save_dir):
    """保存训练结果到JSON文件"""
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(save_dir, f"lstm_ae_results_{timestamp}.json")

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)

    print(f"💾 Results saved to: {results_path}")
    return results_path


# ====================== 8. 主流程 ======================
def main():
    """主函数"""
    # 设置TensorFlow优化
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    print("=" * 80)
    print("LSTM AUTOENCODER TRAINING PIPELINE")
    print("=" * 80)

    import time
    start_time = time.time()

    try:
        # 1. 加载数据
        X, df_hourly, actual_features = load_data()

        # 使用实际的特征数量
        n_features = X.shape[2]
        print(f"\n📊 Data Information:")
        print(f"  Number of features: {n_features}")
        print(f"  Feature names: {actual_features if 'actual_features' in locals() else 'Not available'}")

        # 2. 划分数据
        X_train, X_val = split_data(X, split_ratio=0.85)

        # 3. 构建模型
        print("\n" + "=" * 80)
        print("MODEL CONSTRUCTION")
        print("=" * 80)

        autoencoder, encoder = build_lstm_autoencoder(TIMESTEPS, n_features)

        # 4. 训练模型
        print("\n" + "=" * 80)
        print("MODEL TRAINING")
        print("=" * 80)
        history, total_params = train_model(autoencoder, X_train, X_val)

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
        autoencoder.save(MODEL_PATH)
        print(f"\n💾 Model saved to: {MODEL_PATH}")

        # 单独保存编码器
        encoder_path = os.path.join(MODEL_SAVE_DIR, "lstm_ae_encoder.h5")
        encoder.save(encoder_path)
        print(f"💾 Encoder saved to: {encoder_path}")

        # 6. 异常检测
        print("\n" + "=" * 80)
        print("ANOMALY DETECTION")
        print("=" * 80)
        mae_full, anomaly_flags, threshold, reconstructions_full = detect_anomalies(
            autoencoder, X, X_val, ANOMALY_THRESHOLD_PCT
        )

        # 7. 可视化
        print("\n" + "=" * 80)
        print("VISUALIZATION")
        print("=" * 80)
        viz_path = visualize_results(
            df_hourly, mae_full, anomaly_flags, threshold,
            X_original=X, X_reconstructed=reconstructions_full, history=history,
            actual_features=actual_features, model_name="LSTM AutoEncoder"
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
            'model_name': 'LSTM_AutoEncoder',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_info': {
                'total_samples': X.shape[0],
                'timesteps': TIMESTEPS,
                'n_features': n_features,
                'training_samples': X_train.shape[0],
                'validation_samples': X_val.shape[0],
                'actual_features': actual_features
            },
            'model_info': {
                'total_params': total_params,
                'lstm_units': LSTM_UNITS,
                'encoding_dim': ENCODING_DIM,
                'batch_size': BATCH_SIZE,
                'epochs_trained': len(history.history['loss']),
                'final_train_loss': float(history.history['loss'][-1]),
                'final_val_loss': float(history.history['val_loss'][-1]),
                'best_val_loss': float(best_val_loss) if best_val_loss else None,
                'best_epoch': best_epoch
            },
            'anomaly_detection': {
                'threshold_percentile': ANOMALY_THRESHOLD_PCT,
                'threshold_value': float(threshold),
                'anomalies_detected': int(np.sum(anomaly_flags)),
                'total_windows': len(mae_full),
                'anomaly_ratio': float(np.mean(anomaly_flags) * 100)
            },
            'file_paths': {
                'model_path': MODEL_PATH,
                'encoder_path': encoder_path,
                'visualization_path': viz_path,
                'data_path': DATA_DIR
            }
        }
        # 转换numpy类型为Python原生类型
        results_dict = convert_numpy_types(results_dict)
        # 9. 保存结果
        results_path = save_results(results_dict, MODEL_SAVE_DIR)

        # 10. 打印总结报告
        print("\n" + "=" * 80)
        print("SUMMARY REPORT")
        print("=" * 80)
        print(f"📊 LSTM AutoEncoder Training Complete")
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
        print(f"  - Model: {MODEL_PATH}")
        print(f"  - Encoder: {encoder_path}")
        print(f"  - Visualization: {viz_path}")
        print(f"  - Results: {results_path}")

        print("\n✅ Process completed successfully!")

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