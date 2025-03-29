import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# 设置窗口大小
window_size = 5

# 设置alpha的范围，观察不同的alpha值
alpha_values = np.arange(0.1, 1, 0.05)

# 用于存储每个alpha下的时间衰减权重
time_decay_weights_matrix = []

# 计算每个alpha下的时间衰减权重
for alpha in alpha_values:
    time_decay_weights = np.array([alpha**(window_size - t - 1) for t in range(window_size)])
    time_decay_weights /= np.sum(time_decay_weights)  # 归一化
    time_decay_weights_matrix.append(time_decay_weights)

# 转换为numpy数组方便绘制热力图
time_decay_weights_matrix = np.array(time_decay_weights_matrix)

# 创建自定义颜色映射
colors = ["#e7dbd3", "#e3f4fd", "#add3e2", "#82afda", "#3480b8"]
cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)

# 绘制热力图
plt.figure(figsize=(10, 6))
sns.heatmap(time_decay_weights_matrix, cmap=cmap, annot=True, fmt=".3f",
            xticklabels=[f"t-{i}" for i in range(window_size, 0, -1)],
            yticklabels=[f"alpha={alpha:.2f}" for alpha in alpha_values],
            linewidths=1, linecolor='black')  # 添加黑色边线

# 设置标题和标签
plt.title('Effect of Time Decay Parameter (alpha) on Weight Distribution')
plt.xlabel('Time Steps (from past to present)')
plt.ylabel('Alpha Values')

# 显示图形
plt.show()
