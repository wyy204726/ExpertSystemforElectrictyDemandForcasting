import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. 读取模型预测数据
preds = pd.read_csv('model_predictions.csv', parse_dates=['date'])

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

# 4. 设置时间衰减参数
alpha = 0.2940
window_size = 5

# 用于存储最终预测结果
ensemble_preds = []

# 遍历每个时间步
for idx in range(window_size, len(preds)):
    # 获取滑动窗口内的模型预测值
    window_preds = preds.iloc[idx - window_size:idx, 1:4].values  # 获取 model1, model2, model3 的预测值
    true_values = preds.iloc[idx - window_size:idx, -1].values  # 获取真实油温（true_values）

    # 获取当前时刻的用电等级
    electricity_level = preds.iloc[idx, -1]

    # 根据用电等级选择不同的模型权重
    if electricity_level == 'High':
        # 高温区间：选择与高温区间平均值最接近的模型预测值
        model_weights = np.abs(window_preds - cluster_means[2])  # 距离高温区间的平均值最近的模型
    elif electricity_level == 'Low':
        # 低温区间：选择与低温区间平均值最接近的模型预测值
        model_weights = np.abs(window_preds - cluster_means[0])  # 距离低温区间的平均值最近的模型
    else:
        # 中温区间：选择与中温区间平均值最接近的模型预测值
        model_weights = np.abs(window_preds - cluster_means[1])  # 距离中温区间的平均值最近的模型

    # 计算当前时间步的加权预测
    best_model_index = np.argmin(model_weights)  # 选择距离目标区间平均值最近的模型

    # 得到最佳模型的预测结果
    weighted_preds = window_preds[best_model_index]  # 选择最佳模型的预测结果

    ensemble_preds.append(weighted_preds)

# 保存最终的集成预测结果
ensemble_preds_df = pd.DataFrame(ensemble_preds, columns=["Final_Ensemble_Prediction"])
ensemble_preds_df.to_csv("ensemble_predictions_with_electricity_level_and_clustering.csv", index=False)

