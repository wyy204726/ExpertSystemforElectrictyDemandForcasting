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
def build_ts_lstme_model(time_steps, d, dropout_rate_2, learning_rate):
    # 输入层
    input = Input(shape=(time_steps, d))
    dropout_default = 0.012524493264036464

    x = Bidirectional(LSTM(32, return_sequences=True,activation='relu'))(input)
    x = Dropout(dropout_default)(x)

    x = Bidirectional(LSTM(32, return_sequences=True,activation='relu'))(x)
    x = Dropout(dropout_rate_2)(x)
    x = BatchNormalization()(x)

    x = Bidirectional(LSTM(32, return_sequences=False,activation='relu'))(x)
    x = Dropout(0.10000001)(x)
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
    dropout_rate_2 = params[0]

    model = build_ts_lstme_model(time_steps, d, dropout_rate_2, 5.180429417021491e-05)

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

    print("dropout_rate_2:{dropout_rate_2}")
    print(f"R2: {r2_test}, MSE: {mse_test}")

    # 更新并输出最优MSE和对应的参数
    if mse_test < global_best_mse:
        global_best_mse = mse_test
        global_best_params = dropout_rate_2

    print(f"当前最优解 - Dropout: {global_best_params}, MSE: {global_best_mse}")

    return mse_test

def ED(fnc, D, NP, GEN, LBv, UBv):
    if isinstance(LBv, (int, float)):
        LB = np.ones(D) * LBv
        UB = np.ones(D) * UBv
    else:
        LB = np.array(LBv)
        UB = np.array(UBv)

    MAXFEV = NP * GEN
    eval_count = 0
    Evali = []
    ishow = 250
    ObjVal = np.zeros(NP)

    # Initial population
    Pop = LB + np.random.rand(NP, D) * (UB - LB)
    for i in range(NP):
        ObjVal[i] = fnc(Pop[i])
        eval_count += 1

    Evali.append(eval_count)
    iBest = np.argmin(ObjVal)
    GlobalMin = ObjVal[iBest]
    Xbest = Pop[iBest]
    g = 1
    Fit_store = np.zeros(GEN)
    Fit_store[g - 1] = GlobalMin
    ubest = Xbest

    while g < GEN:
        ct = 1 - np.random.rand() * g / GEN
        if np.random.rand() < 0.1:
            # Task using Eq. (2)
            id_sorted = np.argsort(ObjVal)
            kd = id_sorted[-1]
            SolD = LB + np.random.rand(D) * (UB - LB)
            f_d = fnc(SolD)
            eval_count += 1
            Pop[kd] = SolD
            ObjVal[kd] = f_d
            GlobalMin = f_d
            Xbest = SolD
        else:
            # Calculate c(t) using Eq. (9)
            a = int(np.ceil(3 * ct))
            if a == 1:
                # Structure using Eq. (3)
                for i in range(NP):
                    P = np.random.permutation(NP)[:3]
                    h, p, k = P
                    SolC = (Pop[h] + Pop[p] + Pop[k]) / 3
                    SolC += 2 * (np.random.rand(D) - 0.5) * (Xbest - SolC)
                    SolC = check_bound(SolC, UB, LB)
                    f_c = fnc(SolC)
                    eval_count += 1
                    if f_c <= GlobalMin:
                        Xbest = SolC
                        GlobalMin = f_c

            elif a == 2:
                # Technology using Eq. (5)
                for i in range(NP):
                    h = np.random.randint(NP)
                    SolB = Pop[i] + (np.random.rand(D) * (Xbest - Pop[i]) + np.random.rand(D) * (Xbest - Pop[h]))
                    SolB = check_bound(SolB, UB, LB)
                    f_b = fnc(SolB)
                    eval_count += 1
                    if f_b <= ObjVal[i]:
                        Pop[i] = SolB
                        ObjVal[i] = f_b
                        if f_b <= GlobalMin:
                            Xbest = SolB
                            GlobalMin = f_b

            elif a == 3:
                # People using Eq. (6)
                for i in range(NP):
                    Change = np.random.randint(D)
                    A = np.random.permutation(NP)[:3]
                    nb1, nb2, nb3 = A
                    SolA = Pop[i].copy()
                    SolA[Change] = Pop[i, Change] + (
                                Pop[i, Change] - (Pop[nb1, Change] + Pop[nb2, Change] + Pop[nb3, Change]) / 3) * (
                                               np.random.rand() - 0.5) * 2
                    SolA = check_bound(SolA, UB, LB)
                    f_a = fnc(SolA)
                    eval_count += 1
                    if f_a <= ObjVal[i]:
                        Pop[i] = SolA
                        ObjVal[i] = f_a
                        if f_a <= GlobalMin:
                            Xbest = SolA
                            GlobalMin = f_a

        if eval_count > MAXFEV:
            break

        if g % ishow == 0:
            print(f'Generation: {g}. Best f: {GlobalMin:.6f}')

        g += 1
        if GlobalMin < Fit_store[g - 1]:
            Fit_store[g - 1] = GlobalMin
            ubest = Xbest
        else:
            Fit_store[g - 1] = Fit_store[g - 2]

        Evali.append(eval_count)

    # Result
    f = Fit_store[g - 1]
    X = ubest
    print(f'The best result: {f:.6f}')
    return f, X, eval_count, Fit_store, Evali


def check_bound(Sol, UB, LB):
    Sol = np.clip(Sol, LB, UB)
    return Sol

# 定义目标函数以用于PLO
def objective_function(params):
    # 参数范围
    lb = np.array(0.1)  # 变量下界
    ub = np.array(0.4)  # 变量上界

    if np.any(params < lb) or np.any(params > ub):
        print("参数越界，跳过这次训练")
        return np.inf  # 返回一个很大的值作为惩罚

    return evaluate_model(params)


N = 50  # PLO算法中的种群大小
MaxFEs = 4  # 最大评估次数
lb = np.array(0.1)  # 变量下界
ub = np.array(0.4)  # 变量上界
dim = 1 #算法dim


# 构建和训练模型
d = 7
r = 24

Leeches_best_score, Leeches_best_pos, Convergence_curve = ED(objective_function, dim, N, MaxFEs, lb, ub)

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