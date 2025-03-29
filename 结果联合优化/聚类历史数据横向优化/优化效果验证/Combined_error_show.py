import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取预测数据和真实值
preds = pd.read_csv('model_predictions.csv', parse_dates=['date'])  # 包含 date, model1, model2, model3, true_values

# 确保日期排序
preds = preds.sort_values(by='date')

# 过滤数据，确保时间范围在2016年7月到12月之间
preds = preds[(preds['date'] >= '2016-07-01') & (preds['date'] <= '2016-12-31')]

# 计算每个模型的绝对误差
preds['Model1_Error'] = np.abs(preds['true_values'] - preds['model1'])
preds['Model2_Error'] = np.abs(preds['true_values'] - preds['model2'])
preds['Model3_Error'] = np.abs(preds['true_values'] - preds['model3'])
preds['Ensemble_Error'] = np.abs(preds['true_values'] - preds['combined_values'])

# 计算每个模型的误差平均值
model1_mean_error = preds['Model1_Error'].mean()
model2_mean_error = preds['Model2_Error'].mean()
model3_mean_error = preds['Model3_Error'].mean()
ensemble_mean_error = preds['Ensemble_Error'].mean()

# 打印平均误差
print(f"Model 1 Average Error: {model1_mean_error:.4f}")
print(f"Model 2 Average Error: {model2_mean_error:.4f}")
print(f"Model 3 Average Error: {model3_mean_error:.4f}")
print(f"Ensemble Average Error: {ensemble_mean_error:.4f}")

# 绘制误差绝对值的时间序列图
plt.figure(figsize=(14, 8))

# 每个模型的误差，指定颜色
plt.plot(preds['date'], preds['Model1_Error'], label='Model 1 Error', alpha=0.7, color='#3480b8')  # 蓝色
plt.plot(preds['date'], preds['Model2_Error'], label='Model 2 Error', alpha=0.7, color='#ffbe7a')  # 橙色
plt.plot(preds['date'], preds['Model3_Error'], label='Model 3 Error', alpha=0.7, color='#fa8878')  # 红色

# 联合预测结果的误差，指定颜色
plt.plot(preds['date'], preds['Ensemble_Error'], label='ECHOES Error', linewidth=2, color='#c82423')  # 深红色

# 设置横轴显示时间范围从2016年7月到12月，右侧不留白
plt.xlim(pd.to_datetime('2016-07-01'), preds['date'].max())

# 设置竖轴从0开始，避免下方留白
plt.ylim(bottom=0)

# 图例和标题
plt.title('Prediction Errors Over Time (July to December 2016)', fontsize=16)
plt.xlabel('Timestamp', fontsize=12)
plt.ylabel('Error (OT)', fontsize=12)
plt.legend()


# 使用tight_layout确保不会有任何多余的空白
plt.tight_layout()

# 显示图表
plt.show()
