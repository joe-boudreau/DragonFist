import numpy
from matplotlib import pyplot as plt
from skimage import filters
from tensorflow import keras

import transformations
from data import DataSet
from model import Claw
from processing import ImageProcessParams

epochs = 10
savedModelFile = "savedModel.h5py"

def main():

    dataset = DataSet.load_from_keras(keras.datasets.fashion_mnist)
    # dataset = DataSet.load_from_keras(keras.datasets.cifar10)

    identity_claw = Claw(dataset, auto_train=False, plot_images=0, epochs=epochs)
    id_history = identity_claw.fit()
    print("Identity claw test accuracy: " + str(identity_claw.evaluate()))

    gaussian_claw = Claw(dataset=dataset, image_process_params=ImageProcessParams(filters.gaussian, {'sigma': 1.5}), auto_train=False, plot_images=0, epochs=epochs)
    gaussian_history = gaussian_claw.fit()
    print("gaussian claw test accuracy: " + str(gaussian_claw.evaluate()))

    sobel_claw = Claw(dataset=dataset, image_process_params=ImageProcessParams(transformations.edge_detection_2d), auto_train=False, plot_images=0, epochs=epochs)
    sobel_history = sobel_claw.fit()
    print("sobel claw test accuracy: " + str(sobel_claw.evaluate()))

    min_claw = Claw(dataset=dataset, image_process_params=ImageProcessParams(transformations.min), auto_train=False, plot_images=0, epochs=epochs)
    min_history = min_claw.fit()
    print("Min claw test accuracy: " + str(min_claw.evaluate()))

    max_claw = Claw(dataset=dataset, image_process_params=ImageProcessParams(transformations.max), auto_train=False, plot_images=0, epochs=epochs)
    max_history = max_claw.fit()
    print("Max claw test accuracy: " + str(gaussian_claw.evaluate()))

    rank_claw = Claw(dataset=dataset, image_process_params=ImageProcessParams(transformations.rank), auto_train=False, plot_images=0, epochs=epochs)
    rank_history = rank_claw.fit()
    print("rank claw test accuracy: " + str(rank_claw.evaluate()))

    median_claw = Claw(dataset=dataset, image_process_params=ImageProcessParams(transformations.median), auto_train=False, plot_images=0, epochs=epochs)
    median_history = median_claw.fit()
    print("median claw test accuracy: " + str(median_claw.evaluate()))

    avgcol_claw = Claw(dataset=dataset, image_process_params=ImageProcessParams(filter_function=transformations.average_cols, adaptor=transformations.compat2d), auto_train=False, plot_images=0, epochs=epochs)
    avgcol_history = avgcol_claw.fit()
    print("average cols claw test accuracy: " + str(avgcol_claw.evaluate()))

    avgrows_claw = Claw(dataset=dataset, image_process_params=ImageProcessParams(filter_function=transformations.average_rows, adaptor=transformations.compat2d), auto_train=False, plot_images=0, epochs=epochs)
    avgrows_history = avgrows_claw.fit()
    print("average rows claw test accuracy: " + str(avgrows_claw.evaluate()))


    history_dict = {"No Filter" : id_history,
                    "Gaussian Blur" : gaussian_history,
                    "Edge Detection" : sobel_history,
                    "Min" : min_history,
                    "Max" : max_history,
                    "Rank" : rank_history,
                    "Median" : median_history,
                    "Average Columns" : avgcol_history,
                    "Average Rows" : avgrows_history}

    plot_training(history_dict)

    plt.show()

def plot_training(history_dict):
    plt.figure(1, figsize=(15, 10))
    for filter_name in history_dict:
        hist = history_dict[filter_name]
        print(filter_name + " history: " + str(hist.history))
        plt.plot(hist.history['categorical_accuracy'], '.-')
    plt.title('Model Training Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.yticks(numpy.arange(0, 1, step=0.05))
    plt.xticks(numpy.arange(0, epochs, step=1))
    plt.legend(history_dict.keys(), loc='upper left')

main()
