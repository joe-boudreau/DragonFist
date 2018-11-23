import tensorflow as tf
from tensorflow import keras

from matplotlib import pyplot as plt
from matplotlib import cm

from pathlib import Path

import transformations

from Model import Model
from Data import Data
from Filter import Filter

data = Data(keras.datasets.fashion_mnist)

def main(image_filter=transformations.identity):
    image_filter = Filter(image_filter, data)
    image_filter.initialize()

    model = Model([
        keras.layers.Flatten(input_shape=data.input_shape),
        keras.layers.Dense(128, activation=keras.activations.relu),
        keras.layers.Dense(10, activation=keras.activations.softmax)
    ])

    model.optimizer = keras.optimizers.SGD(lr=0.01, nesterov=True)
    model.epochs = 4
    model.image_filter = image_filter

    model.initialize()
    model.fit()

    image_filter.plot()

    print('Accuracy: {0}'.format(model.test_accuracy))

#main()
main(transformations.average)
