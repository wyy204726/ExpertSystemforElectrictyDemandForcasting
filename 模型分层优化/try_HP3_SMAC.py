import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, Dropout, BatchNormalization
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, History
import joblib
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from smac.facade.hyperband_facade import HyperbandFacade
from smac.scenario import Scenario
import matplotlib.pyplot as plt
from pathlib import Path

# 设置字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 数据预处理部分
file_path = r"D:\魏老师实验室\尝试\ETDataset-main\ETT-small\ETTm1_1000.csv"
data = pd.read_csv(file_path)
data['date'] = pd.to_datetime(data['date'], format='%Y/%m/%d %H:%M:%S', errors='coerce')

data_length = len(data)
start_idx = int(0.4 * data_length)
end_idx = int(0.7 * data_length)

test_data = data[start_idx:end_idx]
train_data = pd.concat([data[:start_idx], data[end_idx:]], ignore_index=True)

# 特征和目标列名称
features = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
target = 'OT'

# 初始化归一化器
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

train_features_scaled = feature_scaler.fit_transform(train_data[features])
train_target_scaled = target_scaler.fit_transform(train_data[[target]])
test_features_scaled = feature_scaler.transform(test_data[features])
test_target_scaled = target_scaler.transform(test_data[[target]])

train_data.loc[:, features] = train_features_scaled
train_data.loc[:, [target]] = train_target_scaled
test_data.loc[:, features] = test_features_scaled
test_data.loc[:, [target]] = test_target_scaled

joblib.dump(feature_scaler, "feature_scaler.pkl")
joblib.dump(target_scaler, "target_scaler.pkl")

# 创建时间序列数据
time_steps = 24

def create_sequences(data, time_steps):
    X_main, y = [], []
    for i in range(len(data) - time_steps):
        X_main.append(data[features].iloc[i:i + time_steps].values)
        y.append(data[target].iloc[i + time_steps])
    return np.array(X_main), np.array(y)

X_main_train, y_train = create_sequences(train_data, time_steps)
X_main_val, y_val = create_sequences(test_data, time_steps)


# 定义Bi-LSTM模型
def build_ts_lstme_model(time_steps, d, dropout_rate_3, learning_rate=0.1):
    input = Input(shape=(time_steps, d))

    dropout_default = 0.2

    # 第一层和第二层使用默认的 dropout_rate
    x = Bidirectional(LSTM(32, return_sequences=True, activation='relu'))(input)
    x = Dropout(dropout_default)(x)

    x = Bidirectional(LSTM(32, return_sequences=True, activation='relu'))(x)
    x = Dropout(dropout_default)(x)
    x = BatchNormalization()(x)

    # 第三层的 dropout_rate 由超参数控制
    x = Bidirectional(LSTM(32, return_sequences=False, activation='relu'))(x)
    x = Dropout(dropout_rate_3)(x)  # 只调整第三层的 dropout_rate
    x = BatchNormalization()(x)

    output = Dense(1, activation='linear', name='final_output')(x)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model = Model(inputs=input, outputs=output)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    return model

# 自定义目标函数，加入seed和budget参数
def objective_function(params, budget, seed):
    dropout_rate_3 = params['dropout_rate_3']  # 只优化第三层的 dropout_rate
    learning_rate = 0.2  # 使用默认的学习率

    model = build_ts_lstme_model(time_steps, X_main_train.shape[2], dropout_rate_3, learning_rate)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    history = History()  # 记录训练过程

    # 训练模型
    model.fit(X_main_train, y_train, epochs=20, batch_size=32, validation_split=0.2,
              callbacks=[early_stopping, history])

    # 预测并计算MSE
    y_pred = model.predict(X_main_val)
    mse = mean_squared_error(y_val, y_pred)

    # 可视化训练过程
    plt.plot(history.history['loss'], label='训练集损失')
    plt.plot(history.history['val_loss'], label='验证集损失')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Dropout Rate (Layer 3): {dropout_rate_3}')
    plt.legend()
    plt.show()

    return mse

# 定义参数空间
cs = ConfigurationSpace()
dropout_rate_3 = UniformFloatHyperparameter('dropout_rate_3', lower=0.1, upper=0.4)

# 使用新的方法添加超参数
cs.add(dropout_rate_3)

# 创建场景，添加min_budget和max_budget
scenario = Scenario(
    configspace=cs,
    name="LSTM Optimization",
    output_directory=Path("smac3_output"),
    deterministic=True,
    min_budget=1,  # 根据需要设置
    max_budget=20,  # 根据需要设置
)

# 使用HyperbandFacade进行优化
try:
    hyperband = HyperbandFacade(
        scenario=scenario,
        target_function=objective_function,
    )

    # 开始优化
    incumbent = hyperband.optimize()
    best_value = incumbent

    print("最终最优参数:", best_value)
    print("最终最优MSE:", objective_function(best_value, budget=20, seed=0))

except Exception as e:
    print("发生错误:", e)
