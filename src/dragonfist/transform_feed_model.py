import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.activations import relu, softmax

from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_keras import KerasModelWrapper

from matplotlib import pyplot as plt
import numpy as np

import functools
import os

import transformations

# Global settings
epochs = 10
# NOTE: By default, datagen.flow uses a batch size of 32.
#       Tweak to find best tradeoff between runtime & memory.
#       (Lower is slower, higher is more memory)
generator_batch_size=32
# NOTE: Must pick good number of workers knowing that there will
#       be filter-level parallelism too.
generator_workers=1

# Dataset-specific settings
# TODO If the model is good for any image classification,
#      set these in a function with custom parameters
#      instead of hardcoding them.
dataset = keras.datasets.cifar10
num_classes = 10
rescale = 1/255.0

# Prepare data
(train_images, train_labels), (test_images, test_labels) = dataset.load_data()
train_images = train_images * rescale
test_images = test_images * rescale
input_shape = train_images.shape[1:]

# Use one-hot encoding for labels
train_labels = keras.utils.to_categorical(train_labels, num_classes)
test_labels  = keras.utils.to_categorical(test_labels, num_classes)


def main(image_filter=transformations.identity, filter_params={}, preprocess_params={},
         plot_filtered_images=5, plot_adversarial_images=5, save_location='genimage'):

    # CNN for learning CIFAR-10, based on this example:
    # https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py
    # TODO Need to do some research to find a good image classifier CNN, then implement it.
    #      Don't forget about regularizers!
    #      Also consider constraints and initializers.
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

    # TODO Cross-validate to find best hyperparameters
    model.compile(optimizer=keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=[keras.metrics.categorical_accuracy])


    filter_name = image_filter.__name__
    if filter_params != {}:
        image_filter = functools.partial(image_filter, **filter_params)

    if plot_filtered_images > 0:
        save_to_dir = None
        if save_location != None and save_location != '':
            save_to_dir = save_location + '/' + filter_name + get_suffix_from_params(filter_params)
            os.makedirs(save_to_dir, exist_ok=True)

        # NOTE: Use a different generator here to show only the effects of the filter,
        #       and not of any other preprocessing step.
        filtergen = ImageDataGenerator(preprocessing_function=image_filter)
        image_generator = filtergen.flow(train_images[0:plot_filtered_images], train_labels[0:plot_filtered_images],
                                         batch_size=1, shuffle=False, save_to_dir=save_to_dir)

        i = 0
        fig, axs = plt.subplots(2, plot_filtered_images)
        for batch_images, batch_labels in image_generator:
            ax = axs[0,i]
            ax.imshow(train_images[i])
            ax.set_axis_off()
            ax = axs[1,i]
            ax.imshow(batch_images[0])
            ax.set_axis_off()

            i = i+1
            if i == plot_filtered_images:
                plt.show()
                break


    datagen = ImageDataGenerator(preprocessing_function=image_filter, **preprocess_params)

    # Always run this in case one of the preprocess_params enables feature-wise normalization.
    datagen.fit(train_images)

    # TODO Don't use test set as validation set
    train_generator      = datagen.flow(train_images, train_labels, batch_size=generator_batch_size)
    validation_generator = datagen.flow(test_images,  test_labels,  batch_size=generator_batch_size)

    model.fit_generator(train_generator,
                        steps_per_epoch=len(train_images)/generator_batch_size,
                        validation_data=validation_generator,
                        validation_steps=len(test_images)/generator_batch_size,
                        epochs=epochs,
                        workers=generator_workers)

    test_loss, test_acc = model.evaluate_generator(
            datagen.flow(test_images, test_labels, batch_size=generator_batch_size),
            steps=len(test_images)/generator_batch_size)
    print('Test accuracy: {}%'.format(test_acc*100))


    # FGSM
    fgsm = FastGradientMethod(KerasModelWrapper(model), keras.backend.get_session())
    fgsm_params = {'eps': 0.3,
                   'clip_min': 0.,
                   'clip_max': 1.}

    if plot_adversarial_images > 0:
        save_to_dir = None
        if save_location != None and save_location != '':
            save_to_dir = save_location + '/FGM-' + filter_name + get_suffix_from_params(filter_params)
            os.makedirs(save_to_dir, exist_ok=True)

        # NOTE: Use a different generator here to show only the effects of the filter,
        #       and not of any other preprocessing step.
        filtergen = ImageDataGenerator(preprocessing_function=image_filter)
        adversarial_images = fgsm.generate_np(test_images[0:plot_adversarial_images], **fgsm_params)
        image_generator = filtergen.flow(adversarial_images, test_labels[0:plot_adversarial_images],
                                         batch_size=1, shuffle=False, save_to_dir=save_to_dir)

        i = 0
        fig, axs = plt.subplots(2, plot_adversarial_images)
        for batch_images, batch_labels in image_generator:
            ax = axs[0,i]
            ax.imshow(test_images[i])
            ax.set_axis_off()
            ax = axs[1,i]
            ax.imshow(batch_images[0])
            ax.set_axis_off()

            i = i+1
            if i == plot_adversarial_images:
                plt.show()
                break

    # Run in batches, otherwise it uses too much memory!
    adv_batch_size = generator_batch_size*10
    num_images = len(test_images)
    num_correct = 0
    i_start = 0
    while True:
        i_end = min(i_start+adv_batch_size, num_images)
        adversarial_images = fgsm.generate_np(test_images[i_start:i_end], **fgsm_params)

        preds = model.predict_generator(
                    datagen.flow(adversarial_images, test_labels[i_start:i_end], batch_size=generator_batch_size),
                    steps=len(adversarial_images)/generator_batch_size)
        num_correct += np.sum(
                        np.argmax(preds, axis=1) ==
                        np.argmax(test_labels[i_start:i_end], axis=1))

        if i_end == num_images:
            break
        else:
            i_start+=adv_batch_size

    adv_acc = num_correct/num_images
    print('Adversarial accuracy: {}%'.format(adv_acc*100))


def get_suffix_from_params(param_dict):
    suffix = ''
    for key, value in param_dict.items():
        suffix += '-{}_{}'.format(key, value)
    return suffix


main(plot_filtered_images=0)
main(transformations.gaussian, filter_params={'sigma':2})
main(transformations.edge_detection_3d)
