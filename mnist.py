from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784) / 255.0
x_test = x_test.reshape(10000, 784) / 255.0
y_train = np_utils.to_categorical(y_train, num_classes=10)



model = Sequential([
    Dense(1300, input_dim=784, activation='relu'),
    Dropout(0.4),
    Dense(10, activation='softmax')
])
model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

es = EarlyStopping(monitor='val_acc')
model.fit(x_train, y_train, batch_size=100, validation_split=0.2, callbacks=[es])#x_train入力y_train出力を学習

predict = model.predict_classes(x_test)
print(sum(predict == y_test) / 10000.0)