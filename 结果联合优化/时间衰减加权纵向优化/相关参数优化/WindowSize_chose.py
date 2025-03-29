import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. 读取数据
data = pd.read_csv('model_predictions.csv', parse_dates=['date'])

# 提取时间特征：月、小时等
data['Month'] = data['date'].dt.month
data['Hour'] = data['date'].dt.hour
data['Day'] = data['date'].dt.day
data['Year'] = data['date'].dt.year

# 2. 用电量区间的聚类分析（根据模型预测值的 true_values 进行聚类）
kmeans = KMeans(n_clusters=4, random_state=0)
data['Electricity_Cluster'] = kmeans.fit_predict(data[['true_values']])

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

data['Electricity_Level'] = data['Electricity_Cluster'].apply(label_electricity_level)

# 设置时间衰减参数
alpha = 0.2940

# 用于存储每个窗口大小下的误差
errors = []

# 逐步调优窗口大小
for window_size in range(2, 7):  # 假设窗口大小范围是从1到5
    ensemble_preds = []
    actual_values = data['true_values'].values[window_size:]

    # 遍历每个时间步
    for idx in range(window_size, len(data)):
        window_preds = data.iloc[idx - window_size:idx, 1:4].values  # 假设模型预测列为model1, model2, model3
        true_values = data.iloc[idx - window_size:idx, -1].values  # 获取真实油温（true_values）

        # 根据用电等级调整权重
        electricity_level = data.iloc[idx, -1]  # 获取当前时刻的用电等级
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

    # 计算误差（此处以MAE为例）
    ensemble_preds = np.array(ensemble_preds)
    error = np.mean(np.abs(ensemble_preds - actual_values))
    errors.append(error)

    # 打印当前窗口大小及其误差
    print(f"Window size: {window_size}, Error: {error}")

# 绘制不同窗口大小下的误差
plt.figure(figsize=(10, 6))
plt.plot(range(2, 7), errors, label='Mean Absolute Error', color='b')

# 去除标题、刻度、网格和图例
plt.title('')  # 不显示标题
plt.xticks([])  # 不显示x轴刻度
plt.yticks([])  # 不显示y轴刻度
plt.grid(False)  # 不显示网格
plt.legend().set_visible(False)  # 不显示图例

plt.show()

# 输出最优窗口大小
optimal_window_size = np.argmin(errors) + 1  # 因为窗口大小从1开始
print(f"Optimal window size: {optimal_window_size}")
