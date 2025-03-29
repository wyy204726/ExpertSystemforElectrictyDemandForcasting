import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# 读取三个模型的预测结果
preds = pd.read_csv("model_predictions.csv")

# 设置时间衰减参数
alpha = 0.9
window_size = 5
n_models = preds.shape[1] - 1
n_steps = preds.shape[0]

# 用于存储最终预测结果
ensemble_preds = []

# 遍历每个时间步
for idx in range(window_size, n_steps):
    # 获取滑动窗口内的模型预测值
    window_preds = preds.iloc[idx - window_size:idx, 1:].values
    true_values = preds.iloc[idx - window_size:idx, 0].values  # 假设第一列为真实值

    # 计算时间衰减权重
    time_decay_weights = np.array([alpha**(window_size - t - 1) for t in range(window_size)])
    time_decay_weights /= np.sum(time_decay_weights)

    # 计算加权的预测特征（作为输入到线性回归模型的特征）
    weighted_preds = np.dot(time_decay_weights, window_preds)

    # 使用线性回归训练回归联合权重
    reg = LinearRegression()
    reg.fit(weighted_preds.reshape(-1, 1), true_values)

    # 计算当前时间步的预测
    final_pred = np.dot(weighted_preds, reg.coef_) + reg.intercept_
    ensemble_preds.append(final_pred)

# 保存结果
ensemble_preds_df = pd.DataFrame(ensemble_preds, columns=["Final_Ensemble_Prediction"])
ensemble_preds_df.to_csv("ensemble_predictions_with_regression.csv", index=False)
