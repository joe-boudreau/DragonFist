import keras

from skimage import filters
from skimage.color.adapt_rgb import each_channel, hsv_value

import transformations as tr

from data import DataSet
from processing import ImageFilter, ImageProcessParams
from model import Claw, Palm, Fist
from model_makers import *
from attacks import *


def test_it(m):
    test_acc = m.evaluate()
    print('Test accuracy: {:.2f}%'.format(test_acc*100))


dataset = DataSet.load_from_keras(keras.datasets.fashion_mnist)

palm = Palm(dataset,
        [ImageFilter(filters.gaussian),
         ImageFilter(filters.sobel, {}, tr.compat2d)],
        {'zca_whitening':True},
        auto_train=True,
        epochs=3)
test_it(palm)

claw = Claw(dataset,
    auto_train=True,
    plot_images=0,
    epochs=1)
test_it(claw)

attackFGM(claw, dataset)
attackFGM(palm, dataset)

#test_claw(Claw(dataset.training_set,
#    ImageProcessParams(filters.gaussian, {'sigma':0.5}),
#    auto_train=True,
#    epochs=1),
#    dataset.test_set)
#
#test_claw(Claw(dataset.training_set,
#    ImageProcessParams(filters.sobel, {}, tr.compat2d, {'zca_whitening':True}),
#    auto_train=True,
#    epochs=1),
#    dataset.test_set)


#dataset = DataSet.load_from_keras(keras.datasets.cifar10)
#
#test_claw(Claw(dataset,
#    auto_train=True,
#    epochs=1))
#
#test_claw(Claw(dataset,
#    ImageProcessParams(filters.gaussian, {'sigma':1.5}),
#    auto_train=True,
#    epochs=1))
#
#test_claw(Claw(dataset,
#    ImageProcessParams(filters.sobel, {}, each_channel, {'zca_whitening':True}),
#    auto_train=True,
#    epochs=1))
#
#test_claw(Claw(dataset,
#    ImageProcessParams(filters.sobel, {}, hsv_value, {'zca_whitening':True}),
#    auto_train=True,
#    epochs=1))
#
#test_claw(Claw(dataset,
#    ImageProcessParams(filters.sobel, {}, tr.to_gray, {'zca_whitening':True}),
#    auto_train=True,
#    epochs=1))
