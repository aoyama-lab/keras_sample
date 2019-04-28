from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Conv2D, MaxPooling2D, Reshape
from keras.callbacks import EarlyStopping

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0#CNNを使うため、2次元行列のまま
x_test = x_test / 255.0
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

model = Sequential([
    Reshape((28, 28, 1), input_shape=(28, 28)),
    Conv2D(50, (5, 5), activation='relu'),
    Conv2D(50, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dropout(0.2),
    Dense(100, activation='relu'),
    Dropout(0.4),
    Dense(10, activation='softmax')
])

model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

es = EarlyStopping(monitor='val_acc')
model.fit(x_train, y_train, batch_size=100, validation_split=0.2, callbacks=[es])

model.evaluate(x_test, y_test, verbose=0)