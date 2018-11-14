import functools
from matplotlib import pyplot as plt

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
        if self._params != {}:
            self._image_filter = functools.partial(
                self._image_filter,
                **self._params
            )

        if self._preprocessing:
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

    def plot(self, num_images=5):
        cols = ["{0}".format(i + 1) for i in range(num_images)]
        rows = ["Original", "Filtered"]

        num_rows = len(rows)
        figure, axes = plt.subplots(nrows=num_rows, ncols=num_images)

        figure.canvas.set_window_title("Filter")
        figure.suptitle(self._image_filter.__name__.title().replace("_", " "))

        for i in range(num_images):
            # Plot the original image
            figure.add_subplot(
                num_rows,
                num_images,
                i + 1
            ).set_axis_off()
            plt.imshow(self._data.test_images[i])

            # Plot the filtered image
            figure.add_subplot(
                num_rows,
                num_images,
                num_images + i + 1
            ).set_axis_off()
            plt.imshow(self._filtered_test_images[i])

            axes[0, i].axis("off")
            axes[1, i].axis("off")

        for ax, col in zip(axes[0], cols):
            ax.set_title(col)

        for ax, row in zip(axes[:, 0], rows):
            ax.axis("on")
            ax.set_frame_on(False)
            ax.tick_params(
                which='both',
                top=False,
                right=False,
                bottom=False,
                left=False,
                labeltop=False,
                labelright=False,
                labelbottom=False,
                labelleft=False
            )
            ax.set_ylabel(row, rotation=90, size="large")

        plt.show()