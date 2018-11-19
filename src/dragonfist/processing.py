from keras.preprocessing.image import ImageDataGenerator

from matplotlib import pyplot as plt
from matplotlib import cm

import numpy as np

import functools

import transformations as tr
from image_utils import ensure_is_plottable, create_image_folder


class ImageProcessor:
    """Wrapper for an ImageDataGenerator for applying a filter & preprocessing to input images."""

    def __init__(self, image_filter=tr.identity, filter_params={}, adaptor=None, preprocess_params={},
            train_images=None, fit_on_filtered=True):
        """
        image_filter:
            A callable object whose first parameter is an image array.
            Any extra parameters must either have default values, or be set by filter_params.
            Preferably a Function. Must at least be callable and have a __name__ method.

        filter_params:
            Extra arguments to pass into image_filter.

        adaptor:
            A callable object whose first argument is image_filter, which it adapts to a desired dimensionality.
            Preferably a Function. Must at least be callable and have a __name__ method.

        preprocess_params:
            Preprocessing settings for the ImageDataGenerator that will be created.

        train_images:
            A set of training images to fit the ImageDataGenerator on, if necessary.
            Can either fit now, or later by calling 'prepare_datagen'.

        fit_on_filtered:
            If True, will fit on filtered images.
            If False, will filter images after fitting to them.
            Only relevant when train_images is passed in.
        """

        self._name = image_filter.__name__
        for key, value in filter_params.items():
            self._name += '-{}_{}'.format(key, value)

        if filter_params != {}:
            image_filter = functools.partial(image_filter, **filter_params)

        if adaptor != None:
            image_filter = functools.partial(adaptor, image_filter)
            self._name += '-A{}'.format(adaptor.__name__)

        for key, value in preprocess_params.items():
            self._name += '-P{}_{}'.format(key, value)

        self._datagen = ImageDataGenerator(
            preprocessing_function=image_filter,
            **preprocess_params
        )

        # Just a helper variable for plotting
        self._has_preprocessing = preprocess_params != {}

        if train_images != None:
            self.prepare_datagen(train_images, fit_on_filtered)

    @property
    def name(self):
        return self._name

    @property
    def has_preprocessing(self):
        return self._has_preprocessing

    @property
    def has_featurewise_preprocessing(self):
        return self._datagen.featurewise_center or self._datagen.featurewise_std_normalization or self._datagen.zca_whitening

    # TODO Can only ever call this once, which I don't like...shouldn't get in the way, though.
    def prepare_datagen(self, train_images=None, fit_on_filtered=True):
        """
        Obtain an ImageDataGenerator fit on the passed training set that will
        apply the filter & preprocessing that was requested.

        train_images:
            The training set to fit on. Optional if you know that no preprocessing will be done.

        fit_on_filtered:
            If True, will fit on filtered images.
            If False, will filter images after fitting to them.
        """
        if self.has_featurewise_preprocessing:
            if fit_on_filtered:
                # NOTE: Want to fit on filtered images, not to filter fitted images!!
                # Sadly, this will cost memory...but not sure if there's a better way.
                # But, the filtered images don't need to be kept. So to save memory,
                # fit each model's datagen one at a time, and only then can they ever
                # be trained in parallel. Otherwise, would have several copies of the
                # training set at once (each with a different filter applied).
                self._datagen.fit(tr.multi(self._datagen.preprocessing_function, train_images))
            else:
                # How ImageDataGenerator fit & flow work:
                # -fit: fits to whole *original* training set
                # -flow: does the following to each individual input sample x:
                #   -if random transformations were asked for, applies one on x
                #   -calls standardize(x), which does the following:
                #       -first, applies the preprocessing function
                #       -then,  applies samplewise processing (rescale, mean, std)
                #       -last,  applies featurewise processing
                #
                # The point: it fits on the unfiltered images, and then filters them.
                # Probably not what we want!
                self._datagen.fit(train_images)

        return self._datagen


    # TODO Make 'non_blocking' a parameter
    # Also not a fan at how prepare_datagen must be called before using this...
    def plot(self, images, save_image_location='genimage'):
        assert type(images) is np.ndarray, 'Must pass a valid array of images'
        num_images = len(images)

        save_image_location = create_image_folder(save_image_location, self.name)

        cols = ["{0}".format(i + 1) for i in range(num_images)]
        if self.has_preprocessing:
            rows = ["Original", "Filtered", "Preprocessed"]
        else:
            rows = ["Original", "Filtered"]

        num_rows = len(rows)
        figure, axes = plt.subplots(nrows=num_rows, ncols=num_images)

        figure.canvas.set_window_title("Filter")
        figure.suptitle(self.name)

        # This is a different generator to show only the effects of the filter,
        # and not of any other preprocessing step.
        # self._datagen is used for getting the fully-preprocessed images.
        filtergen = ImageDataGenerator(preprocessing_function=self._datagen.preprocessing_function)

        flow_params = {'batch_size':1, 'shuffle':False, 'save_to_dir':save_image_location}

        # If plotting 2d images, use a grayscale color map
        if images.shape[-1] == 1 or len(images.shape[1:]) == 2:
            cmap = cm.gray
        else:
            cmap = None

        i = 0
        for batch_images in filtergen.flow(images, **flow_params):
            # Plot the original image
            ax = axes[0,i]
            ax.set_axis_off()
            ax.imshow(ensure_is_plottable(images[i]), cmap=cmap)

            # Plot the filtered image
            ax = axes[1,i]
            ax.set_axis_off()
            ax.imshow(ensure_is_plottable(batch_images[0]), cmap=cmap)

            i = i+1
            if i == num_images:
                break

        if len(axes) == 3:
            i = 0
            for batch_images in self._datagen.flow(images, save_prefix='P', **flow_params):
                # Plot the fully-preprocessed image
                if len(axes) == 3:
                    ax = axes[2,i]
                    ax.set_axis_off()
                    ax.imshow(ensure_is_plottable(batch_images[0]), cmap=cmap)

                i = i+1
                if i == num_images:
                    break

        for ax, col in zip(axes[0], cols):
            ax.set_title(col)

        for ax, row in zip(axes[:, 0], rows):
            ax.set_axis_on()
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
