import skimage
from scipy import ndimage
from skimage import filters
import numpy as np

"""This module contains the image transformations. Each function in this module should accept an N-dimensional numpy 
array and return an N-dimensional numpy array. Some may accept additional hyperparameters for tuning the filters"""


def identity(images):
    return images


def edge_detection(images):
    return np.array([filters.sobel(i) for i in images])


def gabor_real(images, frequency=0.8):
    """Seems like increase in frequency correlates with model accuracy. training on images filtered with f = 0.5 results
        in accuracy around 50%. If f = 0.8 the accuracy jumps up to 80%"""
    return np.array([filters.gabor(i, frequency)[0] for i in images])


def gabor_imaginary(images, frequency=0.5):
    return np.array([filters.gabor(i, frequency)[1] for i in images])


def gaussian(images, sigma=1):
    """Even with sigma = 2 stanard deviations it was still able to achieve close to 80% accuracy on the MNIST clothes
        classifier. I think in general we want to tune the filters to distort the image as much as possible while still
        achieving at least ~80% accuracy on the training/test data. More distortion = higher robustness?"""
    return np.array([filters.gaussian(i, sigma) for i in images])


def sharpen(images):
    """This filter doesn't really work for low resolution images like the MNIST fashion dataset (28x28px). Might work if
    we try higher resolution images"""
    def sharpenImg(i):
        blurred_img = ndimage.gaussian_filter(i, 1)

        filter_blurred_img = ndimage.gaussian_filter(blurred_img, 0.3)

        alpha = 30
        return blurred_img + alpha * (blurred_img - filter_blurred_img)

    return np.array([sharpenImg(i) for i in images])


def increase_contrast(images):
    """This doesn't work for images which already use the maximum range of the pixels's dtype. Need to use
    the increase_contrast_saturated function below in that case"""
    return np.array([skimage.exposure.rescale_intensity(i) for i in images])


def increase_contrast_saturated(images):
    """This function essentially takes the top and bottom 'p' percent of pixel values in the image
    and rescales them to the max and min value allowed by the dtype. Seems to still produce accurate classifications
    even on models trained with no filter applied"""
    def inc(i):
        p = 5
        (v_min, v_max) = np.percentile(i, (p, 100-p))
        return skimage.exposure.rescale_intensity(i, in_range=(v_min, v_max))

    return np.array([inc(i) for i in images])

