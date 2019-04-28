from keras.models import Sequential
from keras.layers import CuDNNLSTM, Dense, Bidirectional
from sklearn.model_selection import train_test_split
import numpy as np
import glob
import gc
import tensorflow as tf
from keras.backend import tensorflow_backend

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)

max_t = 40
input_dim = 25 # 入力の次元
hidden_dim = 100 # 隠れ層の次元
output_dim = 25 # 出力の次元

# 1フレームずつずらしながら(data_num, max_t, input_dim)のデータを生成
def read_input(mcep_dir, mean, std):
    data = []
    mcep_list = glob.glob(mcep_dir + "/*")

    for mcep_path in mcep_list:
        print(mcep_path)

        mcep = np.loadtxt(mcep_path)[:, 1:] # パワーを無視

        mcep = (mcep - mean) / std

        mcep = np.r_[mcep, np.zeros((max_t - 1, input_dim))]

        # ずらしながらデータを追加(max_tに満たない場合は0を付け足して追加)
        if mcep.shape[0] >= max_t:
            for i in range(mcep.shape[0] - max_t + 1):
                data.append(mcep[i:i + max_t, :])
        else:
            zero = np.zeros((max_t - mcep.shape[0], mcep.shape[1]))
            mcep = np.r_[mcep, zero]

            data.append(mcep)

    return np.array(data).astype("float32")

# 時刻t,t+max_tのデータから時刻tの変換結果を推定する用のデータを生成
def read_teacher(mcep_dir, mean, std):
    data = []
    mcep_list = glob.glob(mcep_dir + "/*")

    for mcep_path in mcep_list:
        print(mcep_path)

        mcep = np.loadtxt(mcep_path)[:, 1:] # パワーを無視

        mcep = (mcep - mean) / std

        data.extend(mcep)

    return np.array(data).astype("float32")

def read_mcep(mcep_dir):
    data = []
    mcep_list = glob.glob(mcep_dir + "/*")

    for mcep_path in mcep_list:
        print(mcep_path)

        mcep = np.loadtxt(mcep_path)[:, 1:] # パワーを無視
        data.extend(mcep)

    return np.array(data)

# LSTMの構築(入力:(data_num, max_t, input_dim), 出力:(data_num, output_dim))
model = Sequential()
model.add(Bidirectional(CuDNNLSTM(hidden_dim, return_sequences=True), input_shape=(max_t, input_dim)))
model.add(Bidirectional(CuDNNLSTM(hidden_dim, return_sequences=True)))
model.add(Bidirectional(CuDNNLSTM(hidden_dim, return_sequences=True)))
model.add(Bidirectional(CuDNNLSTM(hidden_dim)))
model.add(Dense(output_dim))

model.compile(optimizer="adam", loss="mae")

# 全部のデータをまとめ、各次元ごとに平均と標準偏差を求める
x_temp = read_mcep("./aligned_mcep/me")
t_temp = read_mcep("./aligned_mcep/miku")
y = []
y.extend(x_temp)
y.extend(t_temp)
y = np.array(y)
mean = np.mean(y, axis=0)
std = np.std(y, axis=0)
del x_temp
del t_temp
gc.collect()

# 平均と標準偏差を保存
np.savetxt("lstm_mean.txt", mean, fmt="%0.6f")
np.savetxt("lstm_std.txt", std, fmt="%0.6f")

# 入力データと教師データの読み込み
x = read_input("./aligned_mcep/me", mean, std)
t = read_teacher("./aligned_mcep/miku", mean, std)

# 学習用データとテストデータに分離
x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.3)

# 学習
history = model.fit(x_train, t_train, epochs=50, batch_size=256, validation_data=(x_test, t_test))

# モデルを保存
model_json_str = model.to_json()
open("lstm_model.json", "w").write(model_json_str)

# 重みを保存
model.save_weights("lstm.h5")