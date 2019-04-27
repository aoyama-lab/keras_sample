import numpy as np
from keras.layers import Input, Dense
from keras.models import Model

data = np.random.rand(250,5)
labels = (np.sum(data, axis=1) > 2.5) * 2 - 1

inputs = Input(shape=(5,))#タプル(変更不可能な配列)
x = Dense(20, activation='tanh')(inputs)#20次元に変換
predictions = Dense(1, activation='tanh')(x)

model = Model(input=inputs, output=predictions)
model.compile('adam', 'hinge', metrics=['accuracy'])
model.fit(data, labels, nb_epoch=150, validation_split=0.2)

test = np.random.rand(200, 5)
predict = np.sign(model.predict(test).flatten())
real = (np.sum(test, axis=1) > 2.5) * 2 - 1
print(sum(predict == real) / 200.0)


'''
 一つの層は一つの行列(テンソル)積に相当
タプルで次元を宣言してやっていく。
functionAPIを使うと何が違うのかというとSequentialを使っていない。
Sequentialはノードに相当するが、functionAPIでは、ノードを積に相当させたか考え
'''