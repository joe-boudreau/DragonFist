from Model import Model
from tensorflow import keras
from skimage import filters

import transformations as tr

from data import DataSet
from ensembles.basic_averaging import BasicAveragingEnsemble
from processing import ImageProcessParams
from model import Claw
from model_makers import *

ID_MODEL = "1_HL_ReLu_SGD_fashion_MNIST_id.h5py"
GABOR_MODEL = "1_HL_ReLu_SGD_fashion_MNIST_gabor_real.h5py"
EDGE_DETECTION_MODEL = "1_HL_ReLu_SGD_fashion_MNIST_edge_detection.h5py"


def main(train):

    dataset = DataSet.load_from_keras(keras.datasets.fashion_mnist)

    claw1 = Claw(dataset, auto_train=True, retrain=train, epochs=1)
    # test_claw(claw1, dataset)

    claw2 = Claw(dataset, ImageProcessParams(filters.gaussian, {'sigma': 0.5}), auto_train=True, retrain=train, epochs=1)
    # test_claw(claw2, dataset)

    claw3 = Claw(dataset, ImageProcessParams(filters.sobel, {}, tr.compat2d, {'zca_whitening': True}), auto_train=True, retrain=train, epochs=1)
    # test_claw(claw3, dataset)

    ensemble = BasicAveragingEnsemble(claw1, claw2, claw3)
    ensemble.fit()
    test_acc = ensemble.evaluate(dataset.test_images, dataset.test_labels)
    print("Test Accuracy on Ensemble: {0}".format(test_acc))


def test_claw(claw, d):
    test_acc = claw.evaluate(d.test_images, d.test_labels)
    print('Test accuracy: {:.2f}%'.format(test_acc*100))


main(False)
