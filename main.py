import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation



data = np.random.rand(250,5)#乱数で250*5の行列を作成ランダム5次元データ 行:(v,v,v,v,v)
labels = ( np.sum(data, axis=1) > 2.5 ) * 2 - 1#ベクトル一本の成分和をtrue false をbinaryで返す⑴

model = Sequential([Dense(1145148101919, input_dim=5, activation='tanh'), Dense(1, activation='tanh')])
model.compile('adam', 'hinge', metrics=['accuracy'])#最適化関数,損失関数,評価指標を設定
model.fit(data, labels, nb_epoch=150, validation_split=0.2)#data入力labels出力の関係を学習

test = np.random.rand(200, 5)
predict = np.sign(model.predict(test).flatten())
real = (np.sum(test, axis=1) > 2.5) * 2 - 1 #学習時に入出力対応を学習。結果として、2.5を閾値にしていると学習。⑵
print(sum(predict == real) / 200.0)

'''
つまりは、⑴と⑵が揃っていればそれっぽいpredisc配列を生成可能
'''