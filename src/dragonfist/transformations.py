import skimage
from skimage import filters, color
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value

from scipy import ndimage

import numpy as np

"""
This module contains the image transformations. Each function in this module should accept and return a
Numpy array representing an image. Some may accept additional hyperparameters for tuning the filters
"""

def multi(filter_func, images):
    return np.array([filter_func(i) for i in images])

def to_gray(filter_func, image, *args, **kwargs):
    """
    Allows a 3d colour image to be passed into a 2d filter,
    by converting the image into grayscale and back again.
    """
    return color.gray2rgb(filter_func(color.rgb2gray(image), *args, **kwargs))

def compat2d(filter_func, image, *args, **kwargs):
    """
    Allows a 3d, single-channel grayscale image to be
    passed into a grayscale filter that expects a 2d input.
    """
    shape3d = image.shape
    shape2d = (image.shape[0], image.shape[1])
    return filter_func(image.reshape(shape2d), *args, **kwargs).reshape(shape3d)


def identity(image):
    return image


def edge_detection(image):
    return filters.sobel(image)

@adapt_rgb(each_channel)
def edge_detection_3d_each(image):
    return edge_detection(image)

@adapt_rgb(hsv_value)
def edge_detection_3d_hsv(image):
    return edge_detection(image)

@adapt_rgb(to_gray)
def edge_detection_3d_gray(image):
    return edge_detection(image)

def edge_detection_2d(image):
    return compat2d(edge_detection, image)

def gabor_real(image, frequency=0.8):
    return filters.gabor(image, frequency)[0]

@adapt_rgb(each_channel)
def gabor_real_each(image, *args, **kwargs):
    return gabor_real(image, *args, **kwargs)

@adapt_rgb(hsv_value)
def gabor_real_3d_hsv(image, *args, **kwargs):
    return gabor_real(image, *args, **kwargs)


def gabor_imaginary(image, frequency=0.5):
    return filters.gabor(image, frequency)[1]

def gabor_imaginary_3d(image, frequency=0.5):
    return compat3d(gabor_imaginary, image, frequency)


def gaussian(image, sigma=1):
    """
    Even with sigma = 2 stanard deviations it was still able to achieve close to 80% accuracy on the MNIST clothes
    classifier. I think in general we want to tune the filters to distort the image as much as possible while still
    achieving at least ~80% accuracy on the training/test data. More distortion = higher robustness?
    """
    return filters.gaussian(image, sigma)

def average_rows(image):
    num_cols = image.shape[1]

    transformed_image = []

    if (len(image.shape) == 3): #rgb
        for row in image:
            r = [row[i][0] for i in range(num_cols)]
            g = [row[i][1] for i in range(num_cols)]
            b = [row[i][2] for i in range(num_cols)]

            average_r = float(sum(r)) / max(num_cols, 1)
            average_g = float(sum(g)) / max(num_cols, 1)
            average_b = float(sum(b)) / max(num_cols, 1)

            transformed_row = [[average_r, average_g, average_b] for i in range(num_cols)]
            transformed_image.append(transformed_row)
    else: #b/w
        for row in image:
            average = float(sum(row[i] for i in range(num_cols))) / max(num_cols, 1)

            transformed_row = [average for i in range(num_cols)]
            transformed_image.append(transformed_row)

    return np.array(transformed_image)

def average_cols(image):
    if (len(image.shape) == 3): #rgb
        transformed_cols = []

        for col in np.transpose(image, (1, 0, 2)):
            num_rows = image.shape[1]

            r = [col[i][0] for i in range(num_rows)]
            g = [col[i][1] for i in range(num_rows)]
            b = [col[i][2] for i in range(num_rows)]

            average_r = float(sum(r)) / max(num_rows, 1)
            average_g = float(sum(g)) / max(num_rows, 1)
            average_b = float(sum(b)) / max(num_rows, 1)

            transformed_col = [[average_r, average_g, average_b] for i in range(num_rows)]

            transformed_cols.append(transformed_col)

        return np.transpose(np.array(transformed_cols), (1, 0, 2))
    else: #b/w
        return np.transpose(
            average_rows(
                np.transpose(image)
            )
        )

def average(image):
    return average_rows(average_cols(image))

def remove_colors(image, colors):
    if (len(image.shape) == 3): #rgb
        num_cols = image.shape[1]

        transformed_image = []

        for row in image:
            transformed_row = []

            for i in range(num_cols):
                transformed_row.append([
                    row[i][0] if 'r' not in colors else 0,
                    row[i][1] if 'g' not in colors else 0,
                    row[i][2] if 'b' not in colors else 0
                ])

            transformed_image.append([transformed_row for i in range(num_cols)])

        return np.array(transformed_image)
    else:
        return image

def max(image, size=3):
    return ndimage.filters.maximum_filter(image, size=size)

def min(image, size=3):
    return ndimage.filters.minimum_filter(image, size=size)

def rank(image, r=1, size=3):
    return ndimage.filters.rank_filter(image, r, size)

# TODO It looks like this actually blurs images
def sharpen(image):
    """
    This filter doesn't really work for low resolution images like the MNIST fashion dataset (28x28px). Might work if
    we try higher resolution images
    """
    blurred_img = ndimage.gaussian_filter(image, 1)

    filter_blurred_img = ndimage.gaussian_filter(blurred_img, 0.3)

    alpha = 30
    return blurred_img + alpha * (blurred_img - filter_blurred_img)


def increase_contrast(image):
    """
    This doesn't work for images which already use the maximum range of the pixels's dtype. Need to use
    the increase_contrast_saturated function below in that case
    """
    return skimage.exposure.rescale_intensity(image)


def increase_contrast_saturated(image):
    """
    This function essentially takes the top and bottom 'p' percent of pixel values in the image
    and rescales them to the max and min value allowed by the dtype. Seems to still produce accurate classifications
    even on models trained with no filter applied
    """
    p = 5
    (v_min, v_max) = np.percentile(image, (p, 100-p))
    return skimage.exposure.rescale_intensity(image, in_range=(v_min, v_max))

def median(image):
    """
    TODO:I don't know how to input the parameters(size) of median_filter through palm, so I simpily change the param size in 
    ndimage into a constant number size = 5. Seems that preprocess_params is used to solve this problem 
    """
    return ndimage.median_filter(image,size = 5)
