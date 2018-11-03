import tensorflow as tf
from tensorflow import keras
from pathlib import Path

from matplotlib import pyplot as plt

import dragonfist.transformations as transformations

savedModelFile = "savedModel.h5py"

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0


def main(image_filter=transformations.identity, retrain=True, save=False):

    if not retrain and Path(savedModelFile).is_file():
        model = keras.models.load_model(savedModelFile)

    else:
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(10, activation=tf.nn.softmax)
        ])
        model.compile(optimizer=keras.optimizers.SGD(lr=0.01, nesterov=True),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])


        filtered_train_images = image_filter(train_images)

        model.fit(filtered_train_images, train_labels, epochs=4)

        if save:
            model.save(savedModelFile)

    filtered_test_images = image_filter(test_images)

    test_loss, test_acc = model.evaluate(filtered_test_images, test_labels)
    print('Test accuracy:', test_acc)

    plt.figure(1, figsize=(10, 10))
    plt.title("Test Image 0-5 and filtered images 0-5")
    for i in range(1, 5):
        plt.subplot(2, 5, i)
        plt.imshow(test_images[i])
        plt.subplot(2, 5, 5+i)
        plt.imshow(filtered_test_images[i])

    plt.show()


main(retrain=False, save=True)
main(transformations.increase_contrast_saturated, retrain=True)
