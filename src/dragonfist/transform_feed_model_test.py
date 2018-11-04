import tensorflow as tf
from tensorflow import keras

from matplotlib import pyplot as plt
from matplotlib import cm

from pathlib import Path

import transformations

savedModelFile = "savedModel.h5py"

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

num_classes  = max(train_labels)+1
train_labels = keras.utils.to_categorical(train_labels, num_classes)
test_labels  = keras.utils.to_categorical(test_labels, num_classes)

# https://cs231n.github.io/neural-networks-2/
# preprocessing for images:
# - zero centering / mean subtraction (either all at once, or for each color channel)
# - normalization / scaling (not strictly necessary for images)
# - PCA
# - Whitening
# Keras has all of these already, in keras.preprocessing.image

def main(image_filter=transformations.identity, retrain=True, save=False):

    input_shape = train_images.shape[1:]

    if not retrain and Path(savedModelFile).is_file():
        model = keras.models.load_model(savedModelFile)

    else:
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=input_shape),
            keras.layers.Dense(128, activation=keras.activations.relu),
            keras.layers.Dense(10, activation=keras.activations.softmax)
        ])
        model.compile(optimizer=keras.optimizers.SGD(lr=0.01, nesterov=True),
                      loss=keras.losses.categorical_crossentropy,
                      metrics=[keras.metrics.categorical_accuracy])


        filtered_train_images = image_filter(train_images)

        model.fit(filtered_train_images, train_labels, epochs=4)

        if save:
            model.save(savedModelFile)

    filtered_test_images = image_filter(test_images)

    test_loss, test_acc = model.evaluate(filtered_test_images, test_labels)
    print('Test accuracy:', test_acc)

    plt.figure(1, figsize=(10, 10))
    nimages = 5
    plt.title("Test Image 0-{0} and filtered images 0-{0}".format(nimages-1))
    for i in range(5):
        plt_i = i+1
        plt.subplot(2, 5, plt_i).set_axis_off()
        plt.imshow(test_images[i], cmap=cm.gray_r)
        plt.subplot(2, 5, 5+plt_i).set_axis_off()
        plt.imshow(filtered_test_images[i], cmap=cm.gray_r)

    plt.show()


main(retrain=False, save=True)
main(transformations.increase_contrast_saturated, retrain=True)
