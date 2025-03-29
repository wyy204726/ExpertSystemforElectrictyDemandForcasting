import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, Dropout, BatchNormalization
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
import joblib
from scipy.optimize import minimize

# 全局计数器
function_call_counter = 0
global_best_mse = np.inf
global_best_params = None

# 数据预处理部分
file_path = r"D:\魏老师实验室\尝试\ETDataset-main\ETT-small\ETTm1_1000_af.csv"
data = pd.read_csv(file_path)
data['date'] = pd.to_datetime(data['date'], format='%Y/%m/%d %H:%M:%S', errors='coerce')

data_length = len(data)
start_idx = int(0.8 * data_length)
end_idx = int(1 * data_length)

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

    dropout_default = 0.012524493264036464

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
    global global_best_params

    function_call_counter += 1
    print(f"第 {function_call_counter} 次调用目标函数")

    dropout_rate_3 = params[0]  # 获取 dropout_rate_3 参数
    learning_rate = 5.180429417021491e-05  # 使用默认学习率

    # 构建模型
    model = build_ts_lstme_model(time_steps, d=len(features), dropout_rate_3=dropout_rate_3, learning_rate=learning_rate)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    # 训练模型
    history = model.fit(X_main_train, y_train, epochs=20, batch_size=32,
                        validation_data=(X_main_val, y_val), callbacks=[early_stopping])

    y_pred = model.predict(X_main_val)
    y_pred_inv = target_scaler.inverse_transform(y_pred.reshape(-1, 1))
    y_val_inv = target_scaler.inverse_transform(y_val.reshape(-1, 1))

    mse_test = mean_squared_error(y_val_inv, y_pred_inv)
    r2_test = r2_score(y_val_inv, y_pred_inv)

    print(f"dropout_rate_3: {dropout_rate_3}")
    print(f"R2: {r2_test}, MSE: {mse_test}")

    # 更新全局最优解
    if mse_test < global_best_mse:
        global_best_mse = mse_test
        global_best_params = dropout_rate_3

    print(f"当前最优解 - Dropout_3: {global_best_params}, MSE: {global_best_mse}")

    return mse_test

# 局部优化
def local_optimization():
    # 初始参数
    initial_param = [0.3573836852868272]  # dropout_rate_3，作为列表传递

    # 参数边界
    bounds = [(0.1, 0.5)]  # dropout_rate_3的边界

    # 使用 `minimize` 进行局部优化
    res = minimize(evaluate_model, initial_param, bounds=bounds, method='L-BFGS-B')

    print(f"最优参数: {res.x}")
    print(f"最优MSE: {res.fun}")
    return res.x, res.fun

best_params, best_mse = local_optimization()

print("最终最优参数:", best_params)
print("最终最优MSE:", best_mse)
