import tensorflow as tf
from tensorflow import keras

import numpy as numpy

class Data:
    def __init__(self, dataset):
        self._dataset = dataset

        self._num_classes = None
        self._rescale = 1/255.0

        self._input_shape = None

        self._train_images = None
        self._train_labels = None

        self._test_images = None
        self._test_labels = None

        self.initialize()

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def train_images(self):
        return self._train_images

    @property
    def train_labels(self):
        return self._train_labels

    @property
    def test_images(self):
        return self._test_images

    @property
    def test_labels(self):
        return self._test_labels

    def initialize(self):
        self.load()
        self.rescale()
        self.encode()

    def load(self):
        (self._train_images, self._train_labels), (self._test_images, self._test_labels) = self._dataset.load_data()
        self._num_classes = max(self._train_labels) + 1

        # some datasets create an array when you take the max of train_labels
        # (probably a dimensionality thing)
        if (type(self._num_classes) is numpy.ndarray):
            self._num_classes = self._num_classes[0]

    def rescale(self):
        self._train_images = self._train_images * self._rescale
        self._test_images = self._test_images * self._rescale
        self._input_shape = self._train_images.shape[1:]

    def encode(self):
        self._train_labels = keras.utils.to_categorical(
            self._train_labels,
            self._num_classes
        )
        self._test_labels = keras.utils.to_categorical(
            self._test_labels,
            self._num_classes
        )