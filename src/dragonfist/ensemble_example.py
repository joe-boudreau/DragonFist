from Data import Data
from Filter import Filter
from Model import Model
from tensorflow import keras
import transformations
from ensembles.basicaveragingensemble import BasicAveragingEnsemble

ID_MODEL = "1_HL_ReLu_SGD_fashion_MNIST_id.h5py"
GABOR_MODEL = "1_HL_ReLu_SGD_fashion_MNIST_gabor_real.h5py"
EDGE_DETECTION_MODEL = "1_HL_ReLu_SGD_fashion_MNIST_edge_detection.h5py"


def main(train):
    data = Data(keras.datasets.fashion_mnist)

    if train:
        id_filter = Filter(transformations.identity, data)
        id_filter.initialize()
        compile_train_model(data, id_filter, ID_MODEL)

        gabor_filter = Filter(transformations.gabor_real, data)
        gabor_filter.initialize()
        compile_train_model(data, gabor_filter, GABOR_MODEL)

        edge_detection = Filter(transformations.edge_detection, data)
        edge_detection.initialize()
        compile_train_model(data, edge_detection, EDGE_DETECTION_MODEL)

    model1 = keras.models.load_model(ID_MODEL)
    model2 = keras.models.load_model(GABOR_MODEL)
    model3 = keras.models.load_model(EDGE_DETECTION_MODEL)

    test_loss, test_acc = model1.evaluate(data.test_images, data.test_labels)
    print('Accuracy on Identity Model: {0}'.format(test_acc))

    test_loss, test_acc = model2.evaluate(data.test_images, data.test_labels)
    print('Accuracy on Gabor Filter Model: {0}'.format(test_acc))

    test_loss, test_acc = model3.evaluate(data.test_images, data.test_labels)
    print('Accuracy on Edge Detection Model: {0}'.format(test_acc))

    ensemble = BasicAveragingEnsemble(model1, model2, model3)
    test_acc = ensemble.evaluate(data.test_images, data.test_labels)
    print("Test Accuracy on Ensemble: {0}".format(test_acc))


def compile_train_model(data, im_filter, file_name):
    model = Model([
        keras.layers.Flatten(input_shape=data.input_shape),
        keras.layers.Dense(128, activation=keras.activations.relu),
        keras.layers.Dense(10, activation=keras.activations.softmax)
    ])
    model.optimizer = keras.optimizers.SGD(lr=0.01, nesterov=True)
    model.epochs = 5
    model.image_filter = im_filter
    model.initialize()
    model.fit()
    model.save(file_name)


main(False)
