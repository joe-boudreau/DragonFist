import keras

import os

from data import DataSet
import processing
from processing import ImageProcessParams
from model_makers import *


def get_data_dims(dataset):
    """
    Can pass *get_data_dims(dataset) into a model-making function
    instead of input_shape, num_classes.
    """

    return (dataset.input_shape, dataset.num_classes)


# Shared settings
default_model = basic_cnn_model
default_optimizer = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# NOTE: By default, datagen.flow uses a batch size of 32.
#       Tweak to find best tradeoff between runtime & memory.
#       (Lower is slower, higher is more memory)
generator_batch_size=32
# NOTE: Must pick good number of workers, especially if there
#       will be ensemble-level parallelism too.
generator_workers=1

class Claw:
    """
    CLAssification Worker.
    A bundle of a Keras model, train/test data, hyperparameters, etc.
    """

    def __init__(self, dataset, image_process_params=ImageProcessParams(),
            model_maker=default_model,
            optimizer=default_optimizer,
            plot_images=5, save_image_location='genimage',
            auto_train=False, retrain=False, save_model_location='models',
            epochs=10):
        """
        Create, and POSSIBLY TRAIN, a new model for a given dataset & preprocessor.

        dataset:
            An object containing train/test inputs/labels.

        model_maker:
            A callable object that returns a Keras model, sized appropriately for the given dataset.
            Preferably a Function. Must at least be callable and have a __name__ method.

        image_process_params:
            An ImageProcessor object that will filter & preprocess images before sending them to the model.

        optimizer:
            A Keras optimizer to be used by the model.

        auto_train:
            Set to False (the default) to create a new model in need of training.
            Set to True to train the model after creating it, or load a *trained* model from disk.

        retrain:
            Set to True to always retrain the model, even if it could be loaded from disk.

        save_model_location:
            The folder where models should be loaded from / saved to.
            If the model is trained during this constructor, it will be saved to disk.
        """

        self._dataset = dataset
        (self._datagen, self._name) = processing.create_image_processor(image_process_params, dataset)

        if plot_images > 0:
            has_preprocessing = image_process_params.preprocess_params != {}
            processing.plot(dataset.train_images[:plot_images], self._datagen, self._name, has_preprocessing, save_image_location)


        self._name += '-M{}'.format(model_maker.__name__)
        for key, value in optimizer.get_config().items():
            self._name += '-O{}_'.format(key)
            if type(value) is float:
                self._name += '{:.2f}'.format(value)
            else:
                self._name += '{}'.format(value)


        self._model = None
        if save_model_location != None and save_model_location != '':
            self._save_model_location = save_model_location
            if auto_train and not retrain:
                model_filepath = self.model_filepath
                try:
                    self._model = load_model(model_filepath)
                except:
                    print('***Error reading model \'{}\'. Creating new model.***'.format(model_filepath))
                    retrain = True
        else:
            # If saving manually later via 'save_model', save to current directory.
            self._save_model_location = '.'

        if self._model == None:
            # Prepare a new model
            self._model = model_maker(*get_data_dims(dataset))

            # TODO Cross-validate to find best optimizer hyperparameters
            self._model.compile(
                optimizer=optimizer,
                loss=keras.losses.categorical_crossentropy,
                metrics=[keras.metrics.categorical_accuracy])

        self._epochs = epochs

        if retrain:
            self.fit()
            if save_model_location != None and save_model_location != '':
                self.save_model()

    @property
    def model(self):
        """The underlying Keras model."""
        return self._model

    @property
    def datagen(self):
        """
        The data generator that applies filters & preprocessing to all data.
        Any other models/attacks on the same dataset must pass data through this!
        """
        return self._datagen

    @property
    def name(self):
        """
        The name of the owned Keras model, auto-generated from its properties.
        Should be unique for each model.
        """
        return self._name

    @property
    def model_filepath(self):
        """The path that helper methods will use to save/load the owned Keras model."""
        return self._save_model_location + '/' + self._name + '.h5py'

    def save_model(self):
        """
        Convenience method to save the owned Keras model with a name based on its properties.
        If a save location was passed to the constructor, it will be saved there.
        Otherwise, will save to the current working directory.

        Can still call 'model.save(path)' directly.
        """
        os.makedirs(self._save_model_location, exist_ok=True)
        self._model.save(self._save_model_location + '/' + self._name + '.h5py')


    def fit(self, epochs=0):
        """
        Train the model on the dataset that it was given.
        If needed, could access the model & datagen directly and train with those.

        By default, this trains for the number of epochs passed to the constructor.
        Or, pass in the number of epochs to train for.
        Can be called repeatedly to train more.

        Returns the output of model.fit, namely a History object.
        Refer to Keras docs for how to use it.
        """
        if epochs < 1:
            epochs = self._epochs

        # TODO Use a dedicated validation set
        train_generator = self._datagen.flow(
            self._dataset.train_images,
            self._dataset.train_labels,
            #subset="training",
            batch_size=generator_batch_size)

        validation_generator = self._datagen.flow(
            self._dataset.test_images,
            self._dataset.test_labels,
            #subset="validation",
            batch_size=generator_batch_size)

        return self._model.fit_generator(
            train_generator,
            steps_per_epoch=len(self._dataset.train_images)/generator_batch_size,
            validation_data=validation_generator,
            validation_steps=len(self._dataset.test_images)/generator_batch_size,
            epochs=epochs,
            workers=generator_workers)

    def evaluate(self):
        evaluate_generator = self._datagen.flow(
            self._dataset.test_images,
            self._dataset.test_labels,
            batch_size=generator_batch_size)

        return self._model.evaluate_generator(
            evaluate_generator,
            steps=len(self._dataset.test_images)/generator_batch_size)


class Fist:
    """
    Final Intuition STage.
    An ensemble of multiple individual models sharing the same dataset and core structure,
    but each with a different kind of preprocessing applied to its input.
    """

    def __init__(self, dataset, image_process_param_list=[ImageProcessParams()],
            model_maker=default_model,
            optimizer=default_optimizer,
            plot_images=5, save_image_location='genimage',
            epochs=10):
        """
        dataset:
            An object containing train/test inputs/labels.

        model_maker:
            A callable object that returns a Keras model, sized appropriately for the given dataset.
            Should also have a __name__ property.

        image_processors:
            A list of ImageProcessors. A model will be created for each one.

        optimizer:
            A Keras optimizer to be used by each model.
        """

        raise NotImplementedError('TODO')
