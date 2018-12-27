import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.activations import relu, softmax

from matplotlib import pyplot as plt

import functools
import os

import transformations

from Model import Model
from Data import Data
from Filter import Filter

data = Data(keras.datasets.cifar10)

def main(
    image_filter=transformations.identity,
    filter_params={},
    preprocess=True,
    preprocess_params={}):

    image_filter = Filter(image_filter, data)

    image_filter.params = filter_params
    image_filter.preprocessing = preprocess
    image_filter.preprocess_params = preprocess_params

    image_filter.initialize()

    model = Model([
        Conv2D(32, (3,3), padding='same', input_shape=data.input_shape, activation=relu),
        Conv2D(32, (3,3), activation=relu),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.25),

        Conv2D(64, (3,3), padding='same', activation=relu),
        Conv2D(64, (3,3), activation=relu),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(512, activation=relu),
        Dropout(0.5),
        Dense(data.num_classes, activation=softmax)
    ])

    model.image_filter = image_filter

    model.initialize()
    model.fit()

    print('Accuracy: {0}'.format(model.test_accuracy))

#main(preprocess=False)
#main(transformations.edge_detection_3d)
main(transformations.gaussian, filter_params={'sigma':2})