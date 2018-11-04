import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.activations import relu, softmax

from matplotlib import pyplot as plt
from matplotlib import cm

import transformations

# TODO Can probably use any image dataset with this. If so, make it variable
(dataset, num_classes) = (keras.datasets.cifar10, 10)

(train_images, train_labels), (test_images, test_labels) = dataset.load_data()
input_shape = train_images.shape[1:]

# FOR TESTING ONLY: reduce size of training & test sets
#train_images = train_images[:100]
#train_labels = train_labels[:100]
#test_images = test_images[:100]
#test_labels = test_labels[:100]

# Use one-hot encoding for labels
train_labels = keras.utils.to_categorical(train_labels, num_classes)
test_labels  = keras.utils.to_categorical(test_labels, num_classes)

# https://cs231n.github.io/neural-networks-2/
# https://ieeexplore-ieee-org.myaccess.library.utoronto.ca/stamp/stamp.jsp?tp=&arnumber=7808140
# preprocessing for images:
# - zero centering / mean subtraction (either all at once, or for each color channel)
# - normalization / scaling (not strictly necessary for images)
# - PCA/ZCA, Whitening (course claims this isn't used with CNNs, but paper says it's the most important one to use!)
# Keras has all of these already, in keras.preprocessing.image

def main(image_filter=transformations.identity):
    # CNN for learning CIFAR-10, based on this example:
    # https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py
    # TODO Need to do some research to find a good image classifier CNN, then implement it.
    model = Sequential([
        # Conv2D(filters, kernel_size, ...)
        # Dropout(rate, ...)
        # Dense(units, ...)
        Conv2D(32, (3,3), padding='same', input_shape=input_shape, activation=relu),
        Conv2D(32, (3,3), activation=relu),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.25),

        Conv2D(64, (3,3), padding='same', activation=relu),
        Conv2D(64, (3,3), activation=relu),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(512, activation=relu),
        Dropout(0.5),
        Dense(num_classes, activation=softmax)
    ])

    # Want to use gradient descent, but example used RMSprop.
    # So, using the SGD optimzer from this example:
    # https://keras.io/getting-started/sequential-model-guide/#vgg-like-convnet
    model.compile(optimizer=keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=[keras.metrics.categorical_accuracy])

    # TODO Allow "main" to pass other parameters into ImageDataGenerator
    datagen = ImageDataGenerator(rescale=1/255.0, preprocessing_function=image_filter)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    # TODO uncomment this when any of the above are applied
    #datagen.fit(train_images)

    # Interesting methods:
    # datagen.apply_transform(x, transform_parameters)
    # datagen.random_transform(x, seed)
    # datagen.standardize(x)

    # NOTE: To save the preprocessed images to disk, pass save_to_dir='genimage'.
    # WARNING!!! It will save hundreds of images, so it might kill your hard drive.
    # Lower the training/test set size and the number of epochs before using save_to_dir.
    # TODO Add testing code to iterate over datagen.flow and pick some images to plot
    #      with matplotlib or to save to disk, instead of saving all of them.
    #
    # NOTE: By default, datagen.flow uses a batch size of 32.
    #       Using 1 would be nice but it takes too long, and
    #       using the entire set as a batch takes too much memory.
    model.fit_generator(datagen.flow(train_images, train_labels),
                        validation_data=datagen.flow(test_images, test_labels),
                        epochs=10,
                        workers=4)

main()
main(transformations.edge_detection)
