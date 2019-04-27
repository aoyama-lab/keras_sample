import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation



data = np.random.rand(250,5)#乱数で250*5の行列を作成ランダム5次元データ 行:(v,v,v,v,v)
labels = ( np.sum(data, axis=1) > 2.5 ) * 1#ベクトル一本の成分和をtrue false をbinaryで返す
labels = np_utils.to_categorical(labels)#ラベル分け(1→[0,1] 0→[1,0])実際は数値化不可なもの

model = Sequential()#連続関数を定義
model.add(Dense(20, input_dim=5))#五次元データを20に分類
model.add(Activation('relu'))#20個の入力をランプ関数にかける(隠れ層)出力も20
model.add(Dense(2, activation='softmax'))#20この入力をsoftmaxで二次元出力
#.addメンバ関数は入力次元を気にしなくていい?

model.compile('rmsprop', 'categorical_crossentropy', metrics=['accuracy'])#最適化関数,損失関数,評価指標
model.fit(data, labels, nb_epoch=300, validation_split=0.2)

test = np.random.rand(200, 5)
predict = np.argmax(model.predict(test), axis=1)
real = (np.sum(test, axis=1) > 2.5) * 1 #学習時に入出力対応を学習。結果として、2.5を閾値にしていると学習。
sum(predict == real) / 200.0