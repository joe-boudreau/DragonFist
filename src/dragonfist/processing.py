from keras.preprocessing.image import ImageDataGenerator

from matplotlib import pyplot as plt
from matplotlib import cm

import functools

import transformations as tr
from image_utils import ensure_is_plottable, create_image_folder


def has_featurewise_preprocessing(datagen):
    return datagen.featurewise_center or datagen.featurewise_std_normalization or datagen.zca_whitening


class ImageProcessParams:
    """
    A container for parameters for configuring an ImageDataGenerator.
    Includes everything except for a dataset that the ImageDataGenerator will be fit on.
    """

    def __init__(self, image_filter=tr.identity, filter_params={}, adaptor=None, preprocess_params={}):
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
            Preprocessing settings for the ImageDataGenerator that will be created by ImageProcessor.
        """

        self.image_filter = image_filter
        self.filter_params = filter_params
        self.adaptor = adaptor
        self.preprocess_params = preprocess_params


def create_image_processor(params, dataset, fit_on_filtered=True):
    """
    Return a pair containing:
        -An ImageDataGenerator that will process images based on the passed set of parameters.
        -The name associated with the ImageDataGenerator, based on the passed parameters and dataset.

    params:
        An ImageProcessParams object containing all settings for the desired ImageDataGenerator,
        excluding the training set that it must be fit on.

    dataset:
        A DataSet which contains the training set to fit the ImageDataGenerator on,
        as well as the name of the dataset.

    fit_on_filtered:
        If True, will fit on filtered images.
        If False, will filter images after fitting to them.
        Only relevant when train_images is passed in.
    """

    name = 'D{}-I{}'.format(dataset.name, params.image_filter.__name__)
    for key, value in params.filter_params.items():
        name += '-{}_{}'.format(key, value)

    image_filter = params.image_filter
    if params.filter_params != {}:
        image_filter = functools.partial(image_filter, **params.filter_params)

    if params.adaptor != None:
        image_filter = functools.partial(params.adaptor, image_filter)
        name += '-A{}'.format(params.adaptor.__name__)

    for key, value in params.preprocess_params.items():
        name += '-P{}_{}'.format(key, value)

    datagen = ImageDataGenerator(
        preprocessing_function=image_filter,
        **params.preprocess_params
    )

    # Fit on training data
    if has_featurewise_preprocessing(datagen):
        if fit_on_filtered:
            # NOTE: Want to fit on filtered images, not to filter fitted images!!
            # Sadly, this will cost memory...but not sure if there's a better way.
            # But, the filtered images don't need to be kept. So to save memory,
            # fit each model's datagen one at a time, and only then can they ever
            # be trained in parallel. Otherwise, would have several copies of the
            # training set at once (each with a different filter applied).
            datagen.fit(tr.multi(datagen.preprocessing_function, dataset.train_images))
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
            datagen.fit(dataset.train_images)

    return (datagen, name)


# TODO Make 'non_blocking' a parameter
def plot(images, datagen, title, plot_preprocessing, save_image_location='genimage'):
    """
    Plot images after processing them with the passed ImageDataGenerator.
    Shows unprocessed images, images processed by a custom filter, and, if asked for, fully-preprocessed images.
    Also saves the images to save_image_location.
    """
    num_images = len(images)

    save_image_location = create_image_folder(save_image_location, title)

    cols = ["{0}".format(i + 1) for i in range(num_images)]
    if plot_preprocessing:
        rows = ["Original", "Filtered", "Preprocessed"]
    else:
        rows = ["Original", "Filtered"]

    num_rows = len(rows)
    figure, axes = plt.subplots(nrows=num_rows, ncols=num_images)

    figure.canvas.set_window_title("Filter")
    figure.suptitle(title)

    # This is a different generator to show only the effects of the filter,
    # and not of any other preprocessing step.
    # The original datagen is used for getting the fully-preprocessed images.
    filtergen = ImageDataGenerator(preprocessing_function=datagen.preprocessing_function)

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
        for batch_images in datagen.flow(images, save_prefix='P', **flow_params):
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
