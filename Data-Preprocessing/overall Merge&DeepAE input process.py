import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# ====================== 核心函数 - 分层模式提取（方案三核心，无修改） ======================
def extract_hierarchical_patterns(df_aircraft, df_pm25):
    """
    分层提取模式特征（全局 + 24小时级 + 时段级）
    :param df_aircraft: 已预处理的飞机噪音数据
    :param df_pm25: 已预处理的PM2.5数据
    :return: 分层模式字典
    """
    patterns = {}
    pm25_city_cols = [col for col in df_pm25.columns if col != 'timestamp']  # 所有城市PM2.5列
    
    # 第一层：全局统计特征
    patterns['global'] = {
        'total_aircraft_records': len(df_aircraft),
        'overall_noise_mean': df_aircraft['max_slow'].mean() if len(df_aircraft) > 0 else 0,
        'overall_noise_std': df_aircraft['max_slow'].std() if len(df_aircraft) > 1 else 0,
        'B738_global_ratio': (df_aircraft['type'] == 'B738').mean() if len(df_aircraft) > 0 else 0,
        'total_pm25_records': len(df_pm25),
        'overall_pm25_mean': df_pm25[pm25_city_cols].values.flatten().mean() if len(df_pm25) > 0 else 0,
        'overall_pm25_std': df_pm25[pm25_city_cols].values.flatten().std() if len(df_pm25) > 1 else 0,
        'city_pm25_variance': df_pm25[pm25_city_cols].var(axis=1).mean() if len(df_pm25) > 0 else 0
    }
    
    # 第二层：24小时级模式特征（核心）
    for hour in range(24):
        aircraft_hour = df_aircraft[df_aircraft['timestamp'].dt.hour == hour]
        pm25_hour = df_pm25[df_pm25['timestamp'].dt.hour == hour]
        hour_features = {}
        
        # 飞机小时特征
        hour_features['aircraft_count'] = len(aircraft_hour)
        hour_features['noise_mean'] = aircraft_hour['max_slow'].mean() if len(aircraft_hour) > 0 else 0
        hour_features['noise_std'] = aircraft_hour['max_slow'].std() if len(aircraft_hour) > 1 else 0
        hour_features['B738_ratio'] = (aircraft_hour['type'] == 'B738').mean() if len(aircraft_hour) > 0 else 0
        hour_features['A319_ratio'] = (aircraft_hour['type'] == 'A319').mean() if len(aircraft_hour) > 0 else 0
        hour_features['aircraft_type_diversity'] = aircraft_hour['type'].nunique() / len(aircraft_hour) if len(aircraft_hour) > 0 else 0
        
        # PM2.5小时特征
        if len(pm25_hour) > 0:
            pm25_hour_values = pm25_hour[pm25_city_cols].values.flatten()
            hour_features['pm25_mean'] = pm25_hour_values.mean()
            hour_features['pm25_std'] = pm25_hour_values.std() if len(pm25_hour_values) > 1 else 0
            hour_features['pm25_max'] = pm25_hour_values.max()
            hour_features['pm25_min'] = pm25_hour_values.min()
            hour_features['city_pm25_variance'] = pm25_hour[pm25_city_cols].var(axis=1).mean()
        else:
            hour_features['pm25_mean'] = hour_features['pm25_std'] = hour_features['pm25_max'] = hour_features['pm25_min'] = hour_features['city_pm25_variance'] = 0
        
        patterns[f'hour_{hour}'] = hour_features
    
    # 第三层：时段级模式特征
    time_windows = {'morning': (6,10), 'daytime': (10,17), 'evening': (17,21), 'night': (21,6)}
    for period, (start, end) in time_windows.items():
        period_features = {}
        
        # 飞机时段筛选
        if start < end:
            aircraft_mask = (df_aircraft['timestamp'].dt.hour >= start) & (df_aircraft['timestamp'].dt.hour < end)
        else:
            aircraft_mask = (df_aircraft['timestamp'].dt.hour >= start) | (df_aircraft['timestamp'].dt.hour < end)
        aircraft_period = df_aircraft[aircraft_mask]
        period_features['aircraft_count'] = len(aircraft_period)
        period_features['noise_mean'] = aircraft_period['max_slow'].mean() if len(aircraft_period) > 0 else 0
        period_features['B738_ratio'] = (aircraft_period['type'] == 'B738').mean() if len(aircraft_period) > 0 else 0
        
        # PM2.5时段筛选
        if start < end:
            pm25_mask = (df_pm25['timestamp'].dt.hour >= start) & (df_pm25['timestamp'].dt.hour < end)
        else:
            pm25_mask = (df_pm25['timestamp'].dt.hour >= start) | (df_pm25['timestamp'].dt.hour < end)
        pm25_period = df_pm25[pm25_mask]
        period_features['pm25_mean'] = pm25_period[pm25_city_cols].values.flatten().mean() if len(pm25_period) > 0 else 0
        
        patterns[f'period_{period}'] = period_features
    
    print("✅ 分层模式特征提取完成（全局+24小时+4时段）")
    return patterns

# ====================== 模式特征转DAE输入（无修改） ======================
def patterns_to_dae_features(patterns_dict):
    """转换为DeepAutoEncoder可用的标准化特征"""
    # 24小时特征矩阵（核心）
    hour_feature_names = ['aircraft_count', 'noise_mean', 'noise_std', 'B738_ratio', 'A319_ratio',
                         'aircraft_type_diversity', 'pm25_mean', 'pm25_std', 'pm25_max', 'pm25_min', 'city_pm25_variance']
    hour_matrix = np.array([[patterns_dict[f'hour_{h}'][feat] for feat in hour_feature_names] for h in range(24)])
    
    # 全局+时段特征向量
    global_feats = [patterns_dict['global'][f] for f in ['total_aircraft_records', 'overall_noise_mean', 'overall_noise_std', 'B738_global_ratio',
                                                     'total_pm25_records', 'overall_pm25_mean', 'overall_pm25_std', 'city_pm25_variance']]
    period_feats = []
    for p in ['morning', 'daytime', 'evening', 'night']:
        period_feats.extend([patterns_dict[f'period_{p}']['aircraft_count'], patterns_dict[f'period_{p}']['noise_mean'],
                           patterns_dict[f'period_{p}']['B738_ratio'], patterns_dict[f'period_{p}']['pm25_mean']])
    global_period_vector = np.array(global_feats + period_feats).reshape(1, -1)
    
    # 标准化
    scaler = MinMaxScaler(feature_range=(0,1))
    hour_matrix_scaled = scaler.fit_transform(hour_matrix)
    global_period_scaled = scaler.fit_transform(global_period_vector)
    
    # 特征名称映射
    feature_mapping = {
        'hour_feature_names': hour_feature_names,
        'global_feature_names': ['total_aircraft_records', 'overall_noise_mean', 'overall_noise_std', 'B738_global_ratio',
                               'total_pm25_records', 'overall_pm25_mean', 'overall_pm25_std', 'city_pm25_variance'],
        'period_feature_names': [f'period_{p}_{f}' for p in ['morning', 'daytime', 'evening', 'night'] for f in ['aircraft_count', 'noise_mean', 'B738_ratio', 'pm25_mean']]
    }
    
    print("✅ 特征转换完成：")
    print(f"  - 24小时特征矩阵形状：{hour_matrix_scaled.shape}")
    print(f"  - 全局+时段特征向量形状：{global_period_scaled.shape}")
    return hour_matrix_scaled, global_period_scaled, feature_mapping

# ====================== 主流程（直接读取预处理后的数据） ======================
if __name__ == "__main__":
    # ---------------------- 关键：读取你已预处理好的文件 ----------------------
    # 请确认以下路径是你预处理后文件的实际路径，修改后运行！
    processed_pm25_path = r"D:\Oswaldo's surf project\My Database\PM2.5\random_city3PM25_data.csv"
    processed_noise_path = r"D:\Oswaldo's surf project\My Database\aircraft noise_data_test4.csv"
    
    # 读取数据（已预处理，直接加载）
    pm25_df = pd.read_csv(processed_pm25_path)
    noise_df = pd.read_csv(processed_noise_path)
    
    # 转换时间列格式（确保dt.hour可用，必须保留）
    pm25_df['timestamp'] = pd.to_datetime(pm25_df['timestamp'])
    noise_df['timestamp'] = pd.to_datetime(noise_df['timestamp'])
    
    print(f"✅ 已加载预处理后的数据：")
    print(f"  - PM2.5数据：{pm25_df.shape} 条记录，缺失值：{pm25_df.isnull().sum().sum()} 个")
    print(f"  - 飞机噪音数据：{noise_df.shape} 条记录，缺失值：{noise_df.isnull().sum().sum()} 个")
    
    # 提取模式特征
    hierarchical_patterns = extract_hierarchical_patterns(noise_df, pm25_df)
    
    # 转换为DAE特征并保存
    hour_matrix_scaled, global_period_scaled, feature_mapping = patterns_to_dae_features(hierarchical_patterns)
    
    # 保存结果（路径可修改）
    save_path = r"D:\Oswaldo's surf project\My Database\Merge_and_DAE"
    np.save(save_path + "24hour_pattern_matrix_scaled.npy", hour_matrix_scaled)
    np.save(save_path + "global_period_pattern_vector_scaled.npy", global_period_scaled)
    import json
    with open(save_path + "feature_mapping.json", 'w', encoding='utf-8') as f:
        json.dump(feature_mapping, f, ensure_ascii=False, indent=2)
    
    # 打印验证信息
    print("\n📊 示例：0点模式特征（标准化后）：")
    print(f"  - 0点平均噪音：{hour_matrix_scaled[0][1]:.3f}")
    print(f"  - 0点PM2.5均值：{hour_matrix_scaled[0][6]:.3f}")
    print(f"  - 0点B738占比：{hour_matrix_scaled[0][3]:.3f}")
    
    print("\n🎉 运行完成！已保存3个文件到指定路径，可直接用于DAE训练～")