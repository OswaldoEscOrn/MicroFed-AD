import pandas as pd
import random
import os
import numpy as np

# ==================== 配置路径 ====================
input_path = r"D:\Oswaldo's surf project\My Database\PM2.5\PM25_data.csv"
output_path = r"D:\Oswaldo's surf project\My Database\PM2.5\random_city3PM25_data.csv"

# 确保输出目录存在
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# ==================== 1. 读取原始数据 ====================
print("🔍 读取原始数据...")
df = pd.read_csv(input_path, encoding='gbk')  # 原始文件含中文，使用GBK编码
print(f"原始数据形状: {df.shape}")

# 获取所有列名
all_columns = df.columns.tolist()
date_column = all_columns[0]
hour_column = all_columns[1]
city_columns = all_columns[2:]  # 第三列开始为城市

print(f"日期列: {date_column}, 小时列: {hour_column}")
print(f"共有 {len(city_columns)} 个城市列")

# ==================== 2. 随机选择5个城市 ====================
selected_cities = random.sample(city_columns, 5)
print(f"\n🎲 随机选择的5个城市: {selected_cities}")

# 构建包含date、hour和5个城市的新DataFrame
selected_columns = [date_column, hour_column] + selected_cities
df_selected = df[selected_columns].copy()
print(f"选择后数据形状: {df_selected.shape}")

# ==================== 3. 构建timestamp列 ====================
print("\n⏰ 构建timestamp列...")
# 将date转换为字符串，hour补零成两位，合并后解析为datetime
df_selected['timestamp'] = pd.to_datetime(
    df_selected[date_column].astype(str) + df_selected[hour_column].astype(str).str.zfill(2),
    format='%Y%m%d%H'
)
# 格式化为易读的字符串形式（可选，便于保存查看）
df_selected['timestamp'] = df_selected['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

# 重排列，将timestamp放在第一列
cols = ['timestamp'] + selected_cities
df_final = df_selected[cols].copy()
print(f"添加timestamp后数据形状: {df_final.shape}")

# ==================== 4. 处理缺失值 ====================
print("\n🧹 处理缺失值...")

# 4.1 timestamp列缺失值（理论上不会缺失，但保留前向填充逻辑）
df_final['timestamp'] = pd.to_datetime(df_final['timestamp'])  # 先转回datetime以便填充
df_final['timestamp'] = df_final['timestamp'].fillna(method='ffill')
df_final['timestamp'] = df_final['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')  # 转回字符串

# 4.2 城市列缺失值：用各列均值填充
for city in selected_cities:
    city_mean = df_final[city].mean()
    df_final[city] = df_final[city].fillna(city_mean)
    print(f"  列 {city} 均值: {city_mean:.2f}")

# 检查处理后缺失值
print("\n处理后缺失值统计:")
print(df_final.isnull().sum())

# ==================== 5. 保存最终文件 ====================
df_final.to_csv(output_path, index=False, encoding='utf-8')
print(f"\n✅ 数据已保存到: {output_path}")
print(f"最终数据形状: {df_final.shape}")

# ==================== 6. 预览数据 ====================
print("\n📋 数据前5行预览:")
print(df_final.head())