from keras.models import Sequential
from keras.layers import CuDNNLSTM, Dense, Bidirectional,Flatten
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
input_dim = 25  # 入力の次元
hidden_dim = 100  # 隠れ層の次元
output_dim = 25  # 出力の次元

# 1フレームずつずらしながら(data_num, max_t, input_dim)のデータを生成

'''
def read_input(mcep_dir, mean, std):
    data = []
    mcep_list = glob.glob(mcep_dir + "/*")
    for mcep_path in mcep_list:
        print(mcep_path)
        mcep = np.loadtxt(mcep_path)[1:, :]  # パワーを無視        print("passed1")
        mcep = (mcep - mean) / std
        """
        何してるのかあよくわからんからけす
        mcep = np.r_[mcep.T, np.zeros((input_dim,max_t - 1)).T]

        print(mcep.shape)
        print("after")
        # ずらしながらデータを追加(max_tに満たない場合は0を付け足して追加)8000以上あるから上を通る(テストと検証データを合わせるため?)?もしくはmcepファイルの数?
        if mcep.shape[0] >= max_t:#=88073
            print("true")
            print(mcep.shape)
            for i in range(mcep.shape[0] - max_t + 1):
                print(str(i)+":"+str(mcep[i:i + max_t, :])+"is appennded")
                data.append(mcep[i:i + max_t, :])
        else:
            print("false")
            zero = np.zeros((max_t - mcep.shape[0], mcep.shape[1]))
            mcep = np.r_[mcep, zero]
            data.append(mcep)
        """
    return mcep.T#np.array(data).astype("float32")

# 時刻t,t+max_tのデータから時刻tの変換結果を推定する用のデータを生成


def read_teacher(mcep_dir, mean, std):
    data = []
    mcep_list = glob.glob(mcep_dir + "/*")
    for mcep_path in mcep_list:
        print(mcep_path)
        print(np.loadtxt(mcep_path).shape)
        mcep = np.loadtxt(mcep_path)[1:, :]  # パワーを無視
        print(mcep.shape)
        mcep = (mcep - mean) / std
        print(mcep.shape)
        data.extend(mcep)
        print(data)
    return np.array(data).astype("float32")

'''

# 1フレームずつずらしながら(data_num, max_t, input_dim)のデータを生成
def read_input(mcep_dir, mean, std):
    data = []
    mcep_list = glob.glob(mcep_dir + "/*")
    for mcep_path in mcep_list:
        print(mcep_path)
        mcep = np.loadtxt(mcep_path)
        print(mcep.shape)
        mcep=mcep.T
        mcep=np.delete(mcep,0,axis=1)
        #mcep = np.loadtxt(mcep_path)[:, 1:] # パワーを無視
        '''
        上記時点転置済みで一列目だけ消でしたい。→列は0~25の次元の行列
        (88034,25)
        '''
        mcep = (mcep.T - mean)
        mcep=mcep/std
        '''
        (88034,25)
        '''
        mcep=mcep.T
        '''
        (25,88034)
        '''
        mcep = np.r_[mcep, np.zeros((max_t - 1, input_dim))]
        '''
        (8834,25)
        '''
        print(mcep.shape)
        print(mcep[0].shape)
        print("fin sub")
        # ずらしながらデータを追加(max_tに満たない場合は0を付け足して追加)
        print(mcep.shape)#40個ずつスライドさせると最後はデータがなくなる。
        if mcep.shape[0] >= max_t:#データ数が88034あるので、trueを通る。
            print("true")
            for i in range(mcep.shape[0] - max_t + 1):#88034-39=87995回,i:0→87995
                data.append(mcep[i:i + max_t, :])#mcep[0→87995:39→89034:0→88034]
        else:
            print("false")
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
        #mcep = np.loadtxt(mcep_path).T[:, 1:] # パワーを無視
        mcep = np.loadtxt(mcep_path)
        mcep=mcep.T
        mcep=np.delete(mcep,0,axis=1)
        mcep = (mcep.T - mean) / std
        mcep=mcep.T
        data.extend(mcep)
    return np.array(data).astype("float32")






def read_mcep(mcep_dir):
    data = []
    mcep_list = glob.glob(mcep_dir + "/*")
    for mcep_path in mcep_list:
        print(mcep_path)
        mcep = np.loadtxt(mcep_path)[1:, :]  # パワーを無視
        data.extend(mcep)
    return np.array(data)


# LSTMの構築(入力:(data_num, max_t, input_dim), 出力:(data_num, output_dim))
model = Sequential()
model.add(Bidirectional(CuDNNLSTM(hidden_dim, return_sequences=True),input_shape=(max_t, input_dim)))
model.add(Bidirectional(CuDNNLSTM(hidden_dim, return_sequences=True)))
model.add(Bidirectional(CuDNNLSTM(hidden_dim, return_sequences=True)))
model.add(Bidirectional(CuDNNLSTM(hidden_dim)))
model.add(Dense(output_dim))
model.compile(optimizer="adam", loss="mae")

# 全部のデータをまとめ、各次元ごとに平均と標準偏差を求める
x_temp = read_mcep("./mcep_my_ali")
t_temp = read_mcep("./mcep_miku_ali")
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
x = read_input("./mcep_my_ali", mean, std)
t = read_teacher("./mcep_miku_ali", mean, std)

# 学習用データとテストデータに分離
x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.3)

# 学習
history = model.fit(x_train, t_train, epochs=50, batch_size=256, validation_data=(x_test, t_test))  # x_trainを入力に,t_trainを出力する対応を学習。validationはsplitせずにデータを指定
print("history begin")
# モデルを保存
model_json_str = model.to_json()
open("lstm_model.json", "w").write(model_json_str)
print("model fin")
# 重みを保存
model.save_weights("lstm.h5")
