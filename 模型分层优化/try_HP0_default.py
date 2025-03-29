import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, Dropout, BatchNormalization
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib
from tensorflow.keras.callbacks import EarlyStopping
import time
from scipy.special import gamma

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
d=7

def create_sequences(data, time_steps):
    X_main, y = [], []
    for i in range(len(data) - time_steps):
        X_main.append(data[['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']].iloc[i:i + time_steps].values)
        y.append(data['OT'].iloc[i + time_steps])
    return np.array(X_main), np.array(y)


X_main_train, y_train = create_sequences(train_data, time_steps)
X_main_val, y_val = create_sequences(test_data, time_steps)

# 打印数据集的信息
print(f"训练数据 X_main_train 的形状: {X_main_train.shape}")
print(f"训练数据 y_train 的形状: {y_train.shape}")
print(f"验证数据 X_main_val 的形状: {X_main_val.shape}")
print(f"验证数据 y_val 的形状: {y_val.shape}")


# 定义Bi-LSTM模型
def build_ts_lstme_model(time_steps, d, dropout_rate, learning_rate):
    # 输入层
    input = Input(shape=(time_steps, d))

    x = Bidirectional(LSTM(32, return_sequences=True, activation='relu'))(input)
    x = Dropout(dropout_rate)(x)

    x = Bidirectional(LSTM(32, return_sequences=True, activation='relu'))(x)
    x = Dropout(dropout_rate)(x)
    x = BatchNormalization()(x)

    x = Bidirectional(LSTM(32, return_sequences=False, activation='relu'))(x)
    x = Dropout(dropout_rate)(x)
    x = BatchNormalization()(x)

    x = BatchNormalization()(x)
    output = Dense(1, activation='linear', name='final_output')(x)
    # 编译模型
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model = Model(inputs=input, outputs=output)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

    # 返回模型
    return model


def evaluate_model(params):
    global function_call_counter
    global global_best_mse
    global global_best_params

    function_call_counter += 1
    print(f"第 {function_call_counter} 次调用目标函数")

    # 参数解析
    dropout_rate = params[0]
    learning_rate = params[1]

    model = build_ts_lstme_model(time_steps, d, dropout_rate, learning_rate)

    # 使用早停防止过拟合
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    history = model.fit(X_main_train, y_train, epochs=20, batch_size=32,
                        validation_data=(X_main_val, y_val), callbacks=[early_stopping])

    # 预测并反标准化
    y_pred = model.predict(X_main_val)
    y_pred_inv = target_scaler.inverse_transform(y_pred.reshape(-1, 1))
    y_val_inv = target_scaler.inverse_transform(y_val.reshape(-1, 1))

    # 计算评价指标（使用均方误差）
    mse_test = mean_squared_error(y_val_inv, y_pred_inv)
    r2_test = r2_score(y_val_inv, y_pred_inv)

    print(f"dropout_rate: {dropout_rate}, learning_rate: {learning_rate}")
    print(f"R2: {r2_test}, MSE: {mse_test}")

    # 更新并输出最优MSE和对应的参数
    if mse_test < global_best_mse:
        global_best_mse = mse_test
        global_best_params = (dropout_rate, learning_rate)

    print(
        f"当前最优解 - Dropout: {global_best_params[0]}, Learning rate: {global_best_params[1]}, MSE: {global_best_mse}")

    return mse_test


# 定义PLO算法
def tansig(x):
    return 2 / (1 + np.exp(-2 * x)) - 1


def levy(d):
    beta = 1.5
    sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (
            1 / beta)
    u = np.random.randn(d) * sigma
    v = np.random.randn(d)
    # 计算随机生成的步长
    step = u / np.abs(v) ** (1 / beta)
    return step


def PLO_initialization(N, dim, ub, lb):
    # 将 lb 和 ub 转换为 NumPy 数组
    lb = np.array(lb)
    ub = np.array(ub)

    return lb + np.random.rand(N, dim) * (ub - lb)


def PLO(N, MaxFEs, lb, ub, dim, fobj):
    start_time = time.time()

    # Initialization
    FEs = 0
    it = 1
    fitness = np.full(N, np.inf)
    fitness_new = np.full(N, np.inf)

    X = PLO_initialization(N, dim, ub, lb)
    V = np.ones((N, dim))
    X_new = np.zeros((N, dim))

    for i in range(N):
        fitness[i] = fobj(X[i, :])
        FEs += 1

    sorted_indices = np.argsort(fitness)
    X = X[sorted_indices, :]
    fitness = fitness[sorted_indices]
    Bestpos = X[0, :]
    Bestscore = fitness[0]

    Convergence_curve = []
    Convergence_curve.append(Bestscore)

    # Main loop 在这里更新种群位置
    while FEs <= MaxFEs:
        X_sum = np.sum(X, axis=0)
        X_mean = X_sum / N
        w1 = tansig((FEs / MaxFEs) ** 4)
        w2 = np.exp(-(2 * FEs / MaxFEs) ** 3)

        for i in range(N):
            a = np.random.rand() / 2 + 1
            # 旋转运动
            V[i, :] = 1 * np.exp((1 - a) / 100 * FEs)
            LS = V[i, :]
            # 极光椭圆漫步
            GS = levy(dim) * (X_mean - X[i, :] + (lb + np.random.rand(dim) * (ub - lb)) / 2)
            X_new[i, :] = X[i, :] + (w1 * LS + w2 * GS) * np.random.rand(dim)

        E = np.sqrt(FEs / MaxFEs)
        A = np.random.permutation(N)
        for i in range(N):
            for j in range(dim):
                X_new[i, j] = X_new[i, j] + E * np.random.randn()  # 随机变异
            fitness_new[i] = fobj(X_new[i, :])

        best_new_indx = np.argmin(fitness_new)
        if fitness_new[best_new_indx] < Bestscore:
            Bestscore = fitness_new[best_new_indx]
            Bestpos = X_new[best_new_indx, :]

        Convergence_curve.append(Bestscore)
        FEs += N

        if FEs > MaxFEs:
            break

        print(f"Iteration: {it} Bestscore: {Bestscore}")
        it += 1

    elapsed_time = time.time() - start_time
    print(f"优化完成，时间：{elapsed_time:.4f}秒")
    return Bestpos, Bestscore, Convergence_curve


# PLO算法的调用部分
def optimize_with_plo():
    global global_best_mse
    global global_best_params
    global function_call_counter

    N = 30  # 粒子数量
    MaxFEs = 1000  # 最大函数评估次数
    lb = [0.0, 1e-5]  # 下界
    ub = [0.2, 1e-4]  # 上界
    dim = 2  # 维度（包含 dropout_rate 和 learning_rate）


    # 调用PLO优化
    Bestpos, Bestscore, Convergence_curve = PLO(N, MaxFEs, lb, ub, dim, evaluate_model)

    print(f"最优结果：Dropout: {Bestpos[0]}, Learning rate: {Bestpos[1]}, MSE: {Bestscore}")
    print(f"函数调用次数: {function_call_counter}")


optimize_with_plo()
