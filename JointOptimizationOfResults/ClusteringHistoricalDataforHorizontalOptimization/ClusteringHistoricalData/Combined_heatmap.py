import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from matplotlib.colors import LinearSegmentedColormap

# 读取数据并解析时间戳
data = pd.read_csv('D:/pythonProject/pythonProject/dataset/val/logs/baseline_1_vp(exp).csv', parse_dates=['date'])

# 预处理：转换时间戳为日期格式，并提取时间特征
data['date'] = pd.to_datetime(data['date'])
data['Month'] = data['date'].dt.month
data['Hour'] = data['date'].dt.hour
data['Day'] = data['date'].dt.day
data['Year'] = data['date'].dt.year

# 按时间段分组的逻辑
def time_period(hour):
    if 0 <= hour < 6:
        return 'Night'
    elif 6 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 18:
        return 'Afternoon'
    else:
        return 'Evening'

data['Time_Period'] = data['Hour'].apply(time_period)

# 按油温（OT）进行KMeans聚类
kmeans = KMeans(n_clusters=4, random_state=0)
data['Electricity_Cluster'] = kmeans.fit_predict(data[['predicted_values']])

# 计算每个用电区间的平均油温
cluster_means = data.groupby('Electricity_Cluster')['predicted_values'].mean()
print("Average oil temperature per cluster:\n", cluster_means)

# 创建一个函数标记每个时刻的用电等级
def label_electricity_level(cluster):
    if cluster == 0:
        return 'Low'
    elif cluster == 1:
        return 'Medium'
    elif cluster == 2:
        return 'High'
    else:
        return 'Very High'

data['Electricity_Level'] = data['Electricity_Cluster'].apply(label_electricity_level)

# 按时间分区统计每个时间段的平均油温
time_period_means = data.groupby(['Time_Period', 'Electricity_Level'])['predicted_values'].mean().unstack()
print("Average oil temperature by time period and electricity level:\n", time_period_means)

# 按月份和小时计算油温平均值
pivot_table = data.pivot_table(values='predicted_values', index='Hour', columns='Month', aggfunc='mean')

# 创建自定义的颜色映射
colors = ['#3480b8', '#82afda', '#add3e2', '#e7dbd3', '#fa8878', '#c82423']
cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)

# 绘制热力图并设置边线为黑色
plt.figure(figsize=(12, 6))
sns.heatmap(pivot_table, cmap=cmap, annot=True, fmt='.1f', linewidths=0.8, linecolor='black')
plt.title('Average Oil Temperature per Hour and Month')
plt.xlabel('Month')
plt.ylabel('Hour of Day')
plt.show()

# 按时间段绘制各时段油温箱线图
plt.figure(figsize=(10, 6))
sns.boxplot(x='Time_Period', y='predicted_values', data=data, hue='Time_Period', palette='Set3', dodge=False)
plt.title('Oil Temperature Distribution by Time Period')
plt.xlabel('Time Period')
plt.ylabel('Oil Temperature (OT)')
plt.grid(axis='y')
plt.legend([],[], frameon=False)  # 隐藏图例
plt.show()
