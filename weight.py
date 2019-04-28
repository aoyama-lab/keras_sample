import numpy as np
from keras.layers import Input, Dense
from keras.models import Model

data = np.random.rand(250,2)
labels = (np.sum(data, axis=1) > 0.5) * 2 - 1

input = Input(shape=(2,))
hidden = Dense(5, activation='tanh')
output = Dense(1, activation='tanh')

model = Model(input=input, output=output(hidden(input)))
model.compile('adam', 'hinge', metrics=['accuracy'])
model.fit(data, labels, nb_epoch=150, validation_split=0.2)
model.predict(np.array([[0.3, 0.1]]))

'''
1.重みつけ
2.バイアス
3.活性化
の確認

>>> hidden.get_weights()
[array([[-0.34226453, -0.6462068 , -0.64827734, -1.1769922 ,  0.23569117],//ベクトル 成分ごとの重み
       [ 0.09234374,  0.10883233, -0.6665391 , -0.24901955, -0.76028407]],
      dtype=float32), array([ 0.3772085 , -0.42288822, -0.45732254, -0.34593418, -0.39693663],
      dtype=float32)]//バイアス
'''