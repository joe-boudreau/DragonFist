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
