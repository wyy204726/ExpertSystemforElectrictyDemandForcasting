import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, Dropout, concatenate, BatchNormalization, Flatten
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


# 读取数据部分...（保持不变）
# 读取数据
file_path = r"D:\魏老师实验室\尝试\ETDataset-main\ETT-small\ETTm1_1000.csv"
data = pd.read_csv(file_path)

# 数据预处理
data['date'] = pd.to_datetime(data['date'], format='%Y/%m/%d %H:%M:%S', errors='coerce')


# 划分训练集和验证集
train_size = int(0.8 * len(data))
train_data = data[:train_size].copy()
test_data = data[train_size:].copy()

# 特征和目标列名称
features = ['HUFL','HULL','MUFL','MULL','LUFL','LULL','OT']
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
    X_main,y = [], []
    for i in range(len(data) - time_steps):
        X_main.append(data[['HUFL','HULL','MUFL','MULL','LUFL','LULL','OT']].iloc[i:i+time_steps].values)
        y.append(data['OT'].iloc[i+time_steps])
    return np.array(X_main),np.array(y)


X_main_train, y_train = create_sequences(train_data, time_steps)
X_main_val, y_val = create_sequences(test_data, time_steps)

# 打印数据集的信息
print(f"训练数据 X_main_train 的形状: {X_main_train.shape}")
print(f"训练数据 y_train 的形状: {y_train.shape}")
print(f"验证数据 X_main_val 的形状: {X_main_val.shape}")
print(f"验证数据 y_val 的形状: {y_val.shape}")


# 定义Bi-LSTM模型
def build_ts_lstme_model(time_steps, d, dropout_rate_1, learning_rate):
    # 输入层
    input = Input(shape=(time_steps, d))
    dropout_default=0.2

    x = Bidirectional(LSTM(32, return_sequences=True,activation='relu'))(input)
    x = Dropout(dropout_rate_1)(x)

    x = Bidirectional(LSTM(32, return_sequences=True,activation='relu'))(x)
    x = Dropout(dropout_default)(x)
    x = BatchNormalization()(x)

    x = Bidirectional(LSTM(32, return_sequences=False,activation='relu'))(x)
    x = Dropout(dropout_default)(x)
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

    params = params.flatten()  # 展平二维数组为一维
    dropout_rate_3 = params[0]

    model = build_ts_lstme_model(time_steps, d, dropout_rate_3, 0.001)

    # 使用早停防止过拟合
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    history = model.fit(X_main_train, y_train, epochs=50, batch_size=32,
                        validation_data=(X_main_val, y_val), callbacks=[early_stopping])

    # 预测并反标准化
    y_pred = model.predict(X_main_val)
    y_pred_inv = target_scaler.inverse_transform(y_pred.reshape(-1, 1))
    y_val_inv = target_scaler.inverse_transform(y_val.reshape(-1, 1))

    # 计算评价指标（使用均方误差）
    mse_test = mean_squared_error(y_val_inv, y_pred_inv)
    r2_test = r2_score(y_val_inv, y_pred_inv)

    print("dropout_rate_1:{dropout_rate_1}")
    print(f"R2: {r2_test}, MSE: {mse_test}")

    # 更新并输出最优MSE和对应的参数
    if mse_test < global_best_mse:
        global_best_mse = mse_test
        global_best_params = dropout_rate_3

    print(f"当前最优解 - Dropout: {global_best_params}, MSE: {global_best_mse}")

    return mse_test

def CFOA(SearchAgents_no, Max_EFs, lb, ub, dim, fobj):
    # ---------------------Initialization parameter--------------------------%
    Fisher = initialization(SearchAgents_no, dim, ub, lb)
    newFisher = Fisher.copy()
    EFs = 0
    Best_score = float('inf')
    Best_pos = np.zeros(dim)
    cg_curve = np.zeros(Max_EFs)
    fit = np.inf * np.ones(SearchAgents_no)
    newfit = fit.copy()

    # -----------------------Start iteration run-----------------------------%
    while EFs < Max_EFs:
        for i in range(SearchAgents_no):
            Flag4ub = newFisher[i, :] > ub
            Flag4lb = newFisher[i, :] < lb
            newFisher[i, :] = (newFisher[i, :] * (~(Flag4ub + Flag4lb))) + ub * Flag4ub + lb * Flag4lb
            newfit[i] = fobj(newFisher[i, :])
            if newfit[i] <= fit[i]:
                fit[i] = newfit[i]
                Fisher[i, :] = newFisher[i, :]

            if newfit[i] <= Best_score:
                Best_pos = Fisher[i, :].copy()
                Best_score = fit[i]

            EFs += 1
            cg_curve[EFs - 1] = Best_score
            if EFs >= Max_EFs:
                break

        if EFs < Max_EFs / 2:
            alpha = (1 - 3 * EFs / (2 * Max_EFs)) ** (3 * EFs / (2 * Max_EFs))
            p = np.random.rand()
            pos = np.random.permutation(SearchAgents_no)
            i = 0
            while i < SearchAgents_no:
                per = np.random.randint(3, 5)  # Randomly determine the size of the group

                # ---------------------Independent search (p < α)------------------------%
                if p < alpha or i + per - 1 > SearchAgents_no:
                    r = np.random.randint(SearchAgents_no)
                    while r == i:
                        r = np.random.randint(SearchAgents_no)

                    Exp = (fit[pos[i]] - fit[pos[r]]) / (max(fit) - Best_score)
                    rs = np.random.rand(1, dim) * 2 - 1
                    rs = np.linalg.norm(Fisher[r, :] - Fisher[i, :]) * np.random.rand() * (1 - EFs / Max_EFs) * rs / np.sqrt(np.dot(rs, rs.T))
                    newFisher[pos[i], :] = Fisher[pos[i], :] + (Fisher[pos[r], :] - Fisher[pos[i], :]) * Exp + (np.abs(Exp) ** 0.5) * rs
                    i += 1

                # ------------------------Group capture (p ≥ α)--------------------------%
                else:
                    aim = np.sum(fit[pos[i:i + per]] / np.sum(fit[pos[i:i + per]]) * Fisher[pos[i:i + per], :], axis=0)
                    newFisher[pos[i:i + per], :] = Fisher[pos[i:i + per], :] + np.random.rand(per, 1) * (aim - Fisher[pos[i:i + per], :]) + (1 - 2 * EFs / Max_EFs) * (np.random.rand(per, dim) * 2 - 1)
                    i += per

        else:
            # -------------------------Collective capture----------------------------%
            sigma = np.sqrt(2 * (1 - EFs / Max_EFs) / ((1 - EFs / Max_EFs) ** 2 + 1))
            for i in range(SearchAgents_no):
                W = np.abs(Best_pos - np.mean(Fisher, axis=0)) * (np.random.randint(1, 4) / 3) * sigma
                newFisher[i, :] = Best_pos + np.random.normal(0, W, dim)

    return Best_score, Best_pos, cg_curve

def initialization(SearchAgents_no, dim, ub, lb):
    Boundary_no = np.size(ub)  # number of boundaries

    # If the boundaries of all variables are equal and user enters a single
    # number for both ub and lb
    if Boundary_no == 1:
        Positions = np.random.rand(SearchAgents_no, dim) * (ub - lb) + lb
    else:
        # If each variable has a different lb and ub
        Positions = np.zeros((SearchAgents_no, dim))
        for i in range(dim):
            ub_i = ub[i]
            lb_i = lb[i]
            Positions[:, i] = np.random.rand(SearchAgents_no) * (ub_i - lb_i) + lb_i

    return Positions

# 定义目标函数以用于PLO
def objective_function(params):
    # 参数范围
    lb = np.array(0.1)  # 变量下界
    ub = np.array(0.4)  # 变量上界

    if np.any(params < lb) or np.any(params > ub):
        print("参数越界，跳过这次训练")
        return np.inf  # 返回一个很大的值作为惩罚

    return evaluate_model(params)


# 使用PLO算法进行超参数寻优
N = 50  # PLO算法中的种群大小
MaxFEs = 4  # 最大评估次数
lb = np.array(0.1)  # 变量下界
ub = np.array(0.4)  # 变量上界
dim = 1 #算法dim


# 构建和训练模型
d = 7
r = 24

Leeches_best_score, Leeches_best_pos, Convergence_curve = CFOA(N, MaxFEs, lb, ub, dim, objective_function)

# 打印结果
print("最优解 (Dropout率):", Leeches_best_pos)
print("最优适应度 (MSE):", Leeches_best_score)

# 注意：由于PLO算法在objective_function内部被调用，并且我们没有在这里显式地运行它，
# 所以上面的打印语句可能不会输出预期的结果，除非你之前已经以某种方式运行了objective_function或类似的逻辑。
# Plot Convergence curve
plt.plot(Convergence_curve, linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Best Cost')
plt.title('Convergence Curve')
plt.grid(True)
plt.show()