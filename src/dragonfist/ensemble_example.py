from tensorflow import keras
from skimage import filters

import transformations as tr

from data import DataSet
from ensembles.stacking import Stacking
from processing import ImageProcessParams
from model import Claw
from model_makers import *


def main(train):

    dataset = DataSet.load_from_keras(keras.datasets.fashion_mnist)

    claw1 = Claw(dataset, auto_train=True, retrain=False, epochs=3)
    test_claw(claw1, dataset)

    claw2 = Claw(dataset, ImageProcessParams(filters.gaussian, {'sigma': 1.0}), auto_train=True, retrain=train, epochs=3)
    test_claw(claw2, dataset)

    claw3 = Claw(dataset, ImageProcessParams(filters.sobel, {}, tr.compat2d, {'zca_whitening': True}), auto_train=True, retrain=train, epochs=3)
    test_claw(claw3, dataset)

    ensemble = Stacking(claw1, claw2, claw3)

    x = dataset.train_images
    x_test = dataset.test_images
    y = dataset.train_labels
    y_test = dataset.test_labels

    val_acc = ensemble.fit(x=x, y=y)
    print("Ensemble fit validation accuracy: " + str(val_acc))

    test_acc = ensemble.evaluate(x_test, y_test)
    print("Ensemble test set accuracy: " + str(test_acc))


def test_claw(claw, d):
    test_acc = claw.evaluate(d.test_images, d.test_labels)
    print('Test accuracy: {:.2f}%'.format(test_acc*100))


main(True)
