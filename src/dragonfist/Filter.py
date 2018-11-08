import functools

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import transformations

class Filter:
    def __init__(self, image_filter, data):
        self._image_filter = image_filter
        self._data = data

        self._params = {}

        self._preprocessing = False
        self._preprocess_params = {}

        self._datagen = None

        self._filtered_train_images = None
        self._filtered_test_images = None

    @property
    def name(self):
        return self._image_filter.__name__

    @property
    def params(self):
        return self._params

    @property
    def preprocessing(self):
        return self._preprocessing

    @property
    def preprocess_params(self):
        return self._preprocess_params

    @property
    def data(self):
        return self._data

    @property
    def datagen(self):
        return self._datagen

    @property
    def filtered_train_images(self):
        return self._filtered_train_images

    @property
    def filtered_test_images(self):
        return self._filtered_test_images

    @params.setter
    def params(self, value):
        self._params = value

    @preprocessing.setter
    def preprocessing(self, value):
        self._preprocessing = value

    @preprocess_params.setter
    def preprocess_params(self, value):
        self._preprocess_params = value

    def initialize(self):
        if (self._params != {}):
            self._image_filter = functools.partial(
                self._image_filter,
                **self._params
            )

        if (self._preprocessing):
            self._datagen = ImageDataGenerator(
                preprocessing_function=self._image_filter,
                **self._preprocess_params
            )

            self._datagen.fit(self._data.train_images)
        else:
            self._filtered_train_images = transformations.multi(
                self._image_filter,
                self._data.train_images
            )

            self._filtered_test_images = transformations.multi(
                self._image_filter,
                self._data.test_images
            )