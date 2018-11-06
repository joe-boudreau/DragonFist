import skimage
from scipy import ndimage
from skimage import filters, color
import numpy as np

"""This module contains the image transformations. Each function in this module should accept and return a
Numpy array representing an image. Some may accept additional hyperparameters for tuning the filters"""

def multi(filter, images):
    return np.array([filter(i) for i in images])


def compat3d(filter, image, *args):
    return color.gray2rgb(filter(color.rgb2gray(image), *args))


def identity(image):
    return image


def edge_detection(image):
    return filters.sobel(image)

def edge_detection_3d(image):
    return compat3d(edge_detection, image)


def gabor_real(image, frequency=0.8):
    """Seems like increase in frequency correlates with model accuracy. training on images filtered with f = 0.5 results
        in accuracy around 50%. If f = 0.8 the accuracy jumps up to 80%"""
    return filters.gabor(image, frequency)[0]

def gabor_real_3d(image, frequency=0.8):
    return compat3d(gabor_real, image, frequency)


def gabor_imaginary(image, frequency=0.5):
    return filters.gabor(image, frequency)[1]

def gabor_imaginary_3d(image, frequency=0.5):
    return compat3d(gabor_imaginary, image, frequency)


def gaussian(image, sigma=1):
    """Even with sigma = 2 stanard deviations it was still able to achieve close to 80% accuracy on the MNIST clothes
        classifier. I think in general we want to tune the filters to distort the image as much as possible while still
        achieving at least ~80% accuracy on the training/test data. More distortion = higher robustness?"""
    return filters.gaussian(image, sigma)


# TODO It looks like this actually blurs images
def sharpen(image):
    """This filter doesn't really work for low resolution images like the MNIST fashion dataset (28x28px). Might work if
    we try higher resolution images"""
    blurred_img = ndimage.gaussian_filter(image, 1)

    filter_blurred_img = ndimage.gaussian_filter(blurred_img, 0.3)

    alpha = 30
    return blurred_img + alpha * (blurred_img - filter_blurred_img)


def increase_contrast(image):
    """This doesn't work for images which already use the maximum range of the pixels's dtype. Need to use
    the increase_contrast_saturated function below in that case"""
    return skimage.exposure.rescale_intensity(image)


def increase_contrast_saturated(image):
    """This function essentially takes the top and bottom 'p' percent of pixel values in the image
    and rescales them to the max and min value allowed by the dtype. Seems to still produce accurate classifications
    even on models trained with no filter applied"""
    p = 5
    (v_min, v_max) = np.percentile(image, (p, 100-p))
    return skimage.exposure.rescale_intensity(image, in_range=(v_min, v_max))
