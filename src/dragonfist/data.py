from keras.utils import to_categorical

import numpy as np

class DataSet:
    """
    Wrapper for an image dataset.
    More classmethods could be added to this to load different types of datasets.
    """

    def __init__(self, keras_dataset=None, rescale=1, dtype=None):
        """
        Prepare a dataset from one of the keyword arguments.
        Currently the only one is a Keras dataset, but more could be added.
        """

        self._num_classes = None

        self._train_images = None
        self._train_labels = None

        self._test_images = None
        self._test_labels = None

        self._name = None

        if keras_dataset != None:
            (self._train_images, self._train_labels), (self._test_images, self._test_labels) = keras_dataset.load_data()
            self._name = keras_dataset.__name__
            self._name = self._name[self._name.rfind('.')+1:]
        # elif some other kind of data: ...
        else:
            raise ValueError('Need to pass data to load in.')

        self.initialize(rescale, dtype)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def input_shape(self):
        return self._train_images.shape[1:]

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

    @property
    def name(self):
        return self._name


    @classmethod
    def load_from_keras(cls, keras_dataset, rescale=1/255.0, dtype=None):
        """Create a dataset from a Keras dataset."""
        return cls(keras_dataset=keras_dataset, rescale=rescale, dtype=dtype)


    def initialize(self, rescale, dtype=None):
        """Prepare loaded data. Preparations are the same for all kinds of loaded data."""

        if rescale != 1:
            self._train_images = self._train_images * rescale
            self._test_images = self._test_images * rescale

        self._num_classes = np.max(self._train_labels) + 1

        # If image array is of rank 3, images are 2D and shape is missing a channel count.
        # Reshape it to include a dimension for 1 channel.
        if len(self._train_images.shape) == 3:
            self._train_images = self._train_images.reshape(self._train_images.shape + (1,))
            self._test_images  = self._test_images.reshape( self._test_images.shape  + (1,))

        # Use one-hot encoding for labels
        self._train_labels = to_categorical(
            self._train_labels,
            self._num_classes
        )
        self._test_labels = to_categorical(
            self._test_labels,
            self._num_classes
        )

        if dtype is not None:
            self._train_images = self.train_images.astype(dtype)
            self._test_images = self.test_images.astype(dtype)
            self._train_labels = self.train_labels.astype(dtype)
            self._test_labels = self.test_labels.astype(dtype)
