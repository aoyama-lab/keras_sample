import os
import sys
import numpy as np
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.models import Model, model_from_json
import tensorflow as tf
from keras.backend import tensorflow_backend

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)

if len(sys.argv) != 3:
    print("usage: python convert_mcep.py <mcep_path> <converted_mcep_path>")
    exit()

mcep_path = sys.argv[1]
converted_mcep_path = sys.argv[2]

input_dim = 25 # 入力の次元
hidden_dim = 100 # 隠れ層の次元
output_dim = 25 # 出力の次元

max_t = 40
mcep = np.loadtxt(mcep_path)
power = mcep[:, 0]
mc = mcep[:, 1:]
mean = np.loadtxt("lstm_mean.txt")
std = np.loadtxt("lstm_std.txt")

mc = (mc - mean) / std # 正規化
mc = np.r_[mc, np.zeros((max_t - 1, input_dim))]

# 変換用のLSTMを読み込む
lstm = model_from_json(open("lstm_model.json").read())
lstm.load_weights("lstm.h5")

# LSTM入力用に整形
data = []

for i in range(mc.shape[0] - max_t + 1):
    data.append(mc[i:i + max_t, :])

data = np.array(data).astype("float32")

converted_mcep = lstm.predict(data) # LSTMで変換

converted_mcep = converted_mcep * std + mean # 正規化したものを戻す
converted_mcep = np.c_[power, converted_mcep]

np.savetxt(converted_mcep_path, converted_mcep, fmt="%0.6f")