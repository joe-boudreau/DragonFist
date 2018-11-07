import tensorflow as tensorflow
from tensorflow import keras

import numpy as numpy

import transformations

image_filter = transformations.edge_detection

# https://medium.com/@lukaszlipinski/fashion-mnist-with-keras-in-5-minuts-20ab9eb7b905

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

x_train = x_train.astype('float32') / 255
x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)

x_test = x_test.astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

model = keras.Sequential([
    keras.layers.InputLayer(input_shape=[1, 28, 28]),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(
        32,
        (2, 2),
        padding='same',
        bias_initializer=keras.initializers.Constant(0.01),
        kernel_initializer='random_uniform'
    ),
    keras.layers.MaxPool2D(padding='same'),
    keras.layers.Conv2D(
        32,
        (2, 2),
        padding='same',
        bias_initializer=keras.initializers.Constant(0.01),
        kernel_initializer='random_uniform',
        input_shape=(1, 28, 28)
    ),
    keras.layers.MaxPool2D(padding='same'),
    keras.layers.Flatten(),
    keras.layers.Dense(
        128,
        activation='relu',
        bias_initializer=keras.initializers.Constant(0.01),
        kernel_initializer='random_uniform'
    ),
    keras.layers.Dense(10, activation='softmax')
]);

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

x_train_filtered = transformations.multi(image_filter, x_train)
x_test_filtered = transformations.multi(image_filter, x_test)

model.fit(
    x_train_filtered,
    y_train,
    epochs=5,
    batch_size=32,
    validation_data=(x_test_filtered, y_test)
)

score = model.evaluate(x_test, y_test)

print('Accuracy: {0}'.format(score[1]))