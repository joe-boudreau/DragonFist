"""
This module contains functions that return Keras models.
Don't add anything else to this module so that anything that
wants to import all of the functions here can just import *.
"""

import keras
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.activations import relu, softmax


# TODO Don't forget about trying initializers and regularizers

def basic_cnn_model(input_shape, num_classes):
    return Sequential([
        Conv2D(32, (3,3), padding='same', input_shape=input_shape, activation=relu),
        Conv2D(32, (3,3), activation=relu),
        MaxPooling2D(pool_size=(2,2)),
        #Dropout(0.25),

        Conv2D(64, (3,3), padding='same', activation=relu),
        Conv2D(64, (3,3), activation=relu),
        MaxPooling2D(pool_size=(2, 2)),
        #Dropout(0.25),

        Flatten(),
        Dense(512, activation=relu),
        #Dropout(0.5),
        Dense(num_classes, activation=softmax)
    ])

def basic_model(input_shape, num_classes):
    return Sequential([
        Flatten(input_shape=input_shape),
        Dense(128, activation=relu),
        Dense(num_classes, activation=softmax)
    ])
