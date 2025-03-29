import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, Dropout, BatchNormalization
import joblib
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import random

# 全局计数器
function_call_counter = 0

# 全局最优MSE和参数
global_best_mse = np.inf
global_best_params = None

# 数据预处理部分
file_path = r"D:\魏老师实验室\尝试\ETDataset-main\ETT-small\ETTm1_1000_af.csv"
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

# 对训练集特征和目标进行归一化
train_features_scaled = feature_scaler.fit_transform(train_data[features])
train_target_scaled = target_scaler.fit_transform(train_data[[target]])

# 对测试集特征和目标进行归一化
test_features_scaled = feature_scaler.transform(test_data[features])
test_target_scaled = target_scaler.transform(test_data[[target]])

# 将数据重新转换为DataFrame
train_data.loc[:, features] = train_features_scaled
train_data.loc[:, [target]] = train_target_scaled
test_data.loc[:, features] = test_features_scaled
test_data.loc[:, [target]] = test_target_scaled

# 保存归一化器
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

def evaluate_model(params):
    global function_call_counter
    global global_best_mse
    global global_best_param

    function_call_counter += 1
    print(f"第 {function_call_counter} 次调用目标函数")

    # 确保只使用 dropout_rate 作为参数
    dropout_rate_3 = params[0]  # 将params中的第一个元素作为dropout_rate

    learning_rate = 0.1
    model = build_ts_lstme_model(time_steps, d=len(features), dropout_rate_3=dropout_rate_3, learning_rate=learning_rate)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    history = model.fit(X_main_train, y_train, epochs=20, batch_size=32,
                        validation_data=(X_main_val, y_val), callbacks=[early_stopping])

    y_pred = model.predict(X_main_val)
    y_pred_inv = target_scaler.inverse_transform(y_pred.reshape(-1, 1))
    y_val_inv = target_scaler.inverse_transform(y_val.reshape(-1, 1))

    mse_test = mean_squared_error(y_val_inv, y_pred_inv)
    r2_test = r2_score(y_val_inv, y_pred_inv)

    print(f"dropout_rate_3: {dropout_rate_3}")
    print(f"R2: {r2_test}, MSE: {mse_test}")

    if mse_test < global_best_mse:
        global_best_mse = mse_test
        global_best_param = dropout_rate_3

    print(f"当前最优解 - Dropout_3: {global_best_param}, MSE: {global_best_mse}")

    return mse_test

# 模拟退火优化算法
def simulated_annealing(obj, initial_params, bounds, max_iters=1000, initial_temp=1000, cooling_rate=0.95):
    current_params = initial_params
    current_mse = obj(current_params)

    best_params = current_params
    best_mse = current_mse

    temp = initial_temp

    for i in range(max_iters):
        # 生成新参数
        new_params = current_params + np.random.uniform(-0.01, 0.01, len(current_params))
        new_params = np.clip(new_params, [b[0] for b in bounds], [b[1] for b in bounds])

        # 评估新参数
        new_mse = evaluate_model(new_params)

        # 接受条件：如果新解更好，或者满足模拟退火接受条件
        if new_mse < current_mse or random.uniform(0, 1) < np.exp((current_mse - new_mse) / temp):
            current_params = new_params
            current_mse = new_mse

        # 更新全局最优
        if current_mse < best_mse:
            best_params = current_params
            best_mse = current_mse

        # 温度逐步降低
        temp *= cooling_rate

        print(f"迭代 {i+1}/{max_iters} - 最佳MSE: {best_mse}, 当前温度: {temp}")

        if temp < 1e-6:
            break

    return best_params, best_mse

def objective_function(param):
    # 参数范围
    lb = 0.1 # 变量下界
    ub = 0.2  # 变量上界

    if np.any(param < lb) or np.any(param > ub):
        print("参数越界，跳过这次训练")
        return np.inf  # 返回一个很大的值作为惩罚

    return evaluate_model(param)

# 设置初始参数和边界
initial_param = [0.1045314449695677]  # dropout rate
bound = [(0.1, 0.2)]   # dropout rate的边界

# 调用模拟退火算法进行优化
best_param, best_mse = simulated_annealing(objective_function, initial_param, bound)

print("最优参数:", best_param)
print("最优MSE:", best_mse)
