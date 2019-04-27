import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation



data = np.random.rand(250,5)#ランダム5次元データ(v,v,v,v,v)⊤×250
labels = ( np.sum(data, axis=1) > 2.5 ) * 1#ベクトル一本の成分和をtrue false をbinaryで返す
labels = np_utils.to_categorical(labels)#ラベル分け(1→[0,1] 0→[1,0])実際は数値化不可なもの

model = Sequential()
model.add(Dense(20, input_dim=5))#五次元データを20に分類
model.add(Activation('relu'))#20個の出力をランプ関数にかける(隠れ層)
model.add(Dense(2, activation='softmax'))
