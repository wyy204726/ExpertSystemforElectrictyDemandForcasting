import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# 1. 读取模型预测数据
preds = pd.read_csv('model_predictions.csv', parse_dates=['date'])

# 提取时间特征：月、小时等
preds['Month'] = preds['date'].dt.month
preds['Hour'] = preds['date'].dt.hour
preds['Day'] = preds['date'].dt.day
preds['Year'] = preds['date'].dt.year

# 2. 用电量区间的聚类分析（根据模型预测值的 true_values 进行聚类）
kmeans = KMeans(n_clusters=4, random_state=0)
preds['Electricity_Cluster'] = kmeans.fit_predict(preds[['true_values']])

# 计算每个用电区间的平均油温
cluster_means = preds.groupby('Electricity_Cluster')['true_values'].mean()
print("Average oil temperature per cluster:", cluster_means)

# 3. 根据用电量区间创建等级标签
def label_electricity_level(cluster):
    if cluster == 0:
        return 'Low'
    elif cluster == 1:
        return 'Medium'
    elif cluster == 2:
        return 'High'
    else:
        return 'Very High'

preds['Electricity_Level'] = preds['Electricity_Cluster'].apply(label_electricity_level)

# 4. 设置时间衰减参数范围
alpha_values = np.arange(0.290, 0.350, 0.001)

# 用于存储每个时间衰减参数下的误差
errors = []

# 设置窗口大小
window_size = 3

# 计算每个时间衰减参数下的平均误差
for alpha in alpha_values:
    # 用于存储每个时间衰减参数下的预测结果
    ensemble_preds = []
    actual_values = preds['true_values'].values[window_size:]  # 获取真实油温数据

    # 遍历每个时间步
    for idx in range(window_size, len(preds)):
        # 获取滑动窗口内的模型预测值
        window_preds = preds.iloc[idx - window_size:idx, 1:4].values  # 获取 model1, model2, model3 的预测值
        true_values = preds.iloc[idx - window_size:idx, -1].values  # 获取真实油温（true_values）

        # 根据用电等级调整权重
        electricity_level = preds.iloc[idx, -1]  # 获取当前时刻的用电等级
        if electricity_level == 'High':
            model_weights = np.array([0.5, 0.3, 0.2])  # 高用电量时刻：优先选择预测油温较高的模型
        elif electricity_level == 'Low':
            model_weights = np.array([0.2, 0.3, 0.5])  # 低用电量时刻：优先选择预测油温较低的模型
        else:
            model_weights = np.array([0.3, 0.3, 0.3])  # 中等用电量时刻：平均分配权重

        # 根据时间衰减和用电等级的权重计算加权预测
        time_decay_weights = np.array([alpha**(window_size - t - 1) for t in range(window_size)])
        time_decay_weights /= np.sum(time_decay_weights)

        weighted_preds = np.dot(time_decay_weights, window_preds)
        weighted_preds = np.dot(weighted_preds, model_weights)  # 用电等级权重调整预测结果

        ensemble_preds.append(weighted_preds)

    # 计算时序误差的平均值
    ensemble_preds = np.array(ensemble_preds)
    error = np.mean(np.abs(ensemble_preds - actual_values))
    errors.append(error)

    # 打印每个 alpha 参数下的平均误差值
    print(f"Alpha: {alpha:.4f}, Mean Absolute Error: {error:.5f}")

# 绘制折线图
plt.figure(figsize=(10, 6))
plt.plot(alpha_values, errors, label='Mean Absolute Error', color='b')

# 去除标题、刻度、图例和网格
plt.title('')  # 不显示标题
plt.xticks([])  # 不显示x轴刻度
plt.yticks([])  # 不显示y轴刻度
plt.grid(False)  # 不显示网格
plt.legend().set_visible(False)  # 不显示图例

plt.show()

# 输出最优时间衰减参数
optimal_alpha = alpha_values[np.argmin(errors)]
print(f"Optimal time decay parameter: {optimal_alpha:.4f}")
