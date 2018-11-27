from keras.preprocessing.image import ImageDataGenerator

import numpy as np

import os

import processing
from processing import ImageFilter, ImageProcessParams, get_preprocessing_suffix
from model_makers import *


def get_data_dims(dataset):
    """
    Can pass *get_data_dims(dataset) into a model-making function
    instead of input_shape, num_classes.
    """

    return dataset.input_shape, dataset.num_classes

def get_optimizer_suffix(optimizer):
    suffix = ''
    for key, value in optimizer.get_config().items():
        suffix += '-O{}_'.format(key)
        if type(value) is float:
            suffix += '{:.2f}'.format(value)
        else:
            suffix += '{}'.format(value)
    return suffix

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
            An ImageProcessParams object containing a filter & set of preprocess settings,
            which will be applied on images before they are sent to the model.

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

        epochs:
            Number of epochs to train the model with, if it will be trained.
            Setting this to 0 or lower disables training.
        """

        if epochs <= 0:
            auto_train = False
            retrain = False
            epochs = 0

        self._dataset = dataset
        (self._datagen, self._name) = processing.create_image_processor(image_process_params, dataset)

        if plot_images > 0:
            has_preprocessing = image_process_params.preprocess_params != {}
            processing.plot(dataset.train_images[:plot_images], self._datagen, self._name, has_preprocessing, save_image_location)


        self._name += '-M{}'.format(model_maker.__name__)
        self._name += get_optimizer_suffix(optimizer)

        self._model = None
        if save_model_location != None and save_model_location != '':
            self._save_model_location = save_model_location
            if auto_train and not retrain:
                epoch_checkpoint = epochs
                while epoch_checkpoint > 0:
                    try:
                        self._model = load_model(self.get_model_filepath(epoch_checkpoint))
                        break
                    except:
                        epoch_checkpoint -= 1
                if epoch_checkpoint == 0:
                    print('***Error reading model. Creating new one.***')
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

        print('Model name (with epoch count): {}'.format(self.get_model_filepath(epochs)))

        if retrain or epoch_checkpoint < epochs:
            print('Training for {} epoch(s), starting at {}'.format(epochs, epoch_checkpoint))
            self.fit(epochs=epochs, initial_epoch=epoch_checkpoint)
            if save_model_location != None and save_model_location != '':
                self.save_model(epochs)

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

    def get_model_filepath(self, epoch_checkpoint=0):
        """The path that helper methods will use to save/load the owned Keras model."""
        if epoch_checkpoint > 0:
            return '{}/{}-e{}.h5py'.format(self._save_model_location, self._name, epoch_checkpoint)
        else:
            return '{}/{}.h5py'.format(self._save_model_location, self._name)

    def save_model(self, epochs=0):
        """
        Convenience method to save the owned Keras model with a name based on its properties.
        If a save location was passed to the constructor, it will be saved there.
        Otherwise, will save to the current working directory.

        Can still call 'model.save(path)' directly.
        """
        os.makedirs(self._save_model_location, exist_ok=True)
        self._model.save(self.get_model_filepath(epochs))


    def fit(self, x_train=None, y_train=None, epochs=0, initial_epoch=0):
        """
        Train the model on the dataset that it was given.
        If needed, could access the model & datagen directly and train with those.

        By default, this trains for the number of epochs passed to the constructor.
        Or, pass in the number of epochs to train for.
        Can be called repeatedly to train more.

        Returns the output of model.fit, namely a History object.
        Refer to Keras docs for how to use it.
        """
        if x_train is None:
            x_train = self._dataset.train_images
        if y_train is None:
            y_train = self._dataset.train_labels

        if epochs < 1:
            epochs = self._epochs

        train_generator = self._datagen.flow(
            x_train,
            y_train,
            subset="training",
            batch_size=generator_batch_size)

        # Get the number of samples in the training subset
        num_train = len(train_generator.x)

        validation_generator = self._datagen.flow(
            x_train,
            y_train,
            subset="validation",
            batch_size=generator_batch_size)

        # Get the number of samples in the validation subset
        num_validation = len(validation_generator.x)

        return self._model.fit_generator(
            train_generator,
            steps_per_epoch=num_train/generator_batch_size,
            validation_data=validation_generator,
            validation_steps=num_validation/generator_batch_size,
            epochs=epochs,
            initial_epoch=initial_epoch,
            workers=generator_workers)

    def evaluate(self, x_test=None, y_test=None):
        if x_test is None:
            x_test = self._dataset.test_images
        if y_test is None:
            y_test = self._dataset.test_labels

        evaluate_generator = self._datagen.flow(x_test, y_test, batch_size=generator_batch_size)
        loss, acc = self._model.evaluate_generator(evaluate_generator, steps=len(x_test)/generator_batch_size)
        return acc

    def predict(self, x):
        predict_generator = self._datagen.flow(x, batch_size=generator_batch_size, shuffle=False)
        return self._model.predict_generator(predict_generator, steps=len(x)/generator_batch_size)


class Palm:
    """
    Package-Aggregate Learning Model.
    Trains a model on differently-filtered versions of the same training set.
    """
    # TODO deal with momentum & decay...
    def __init__(self, dataset, extra_filters=[], preprocess_params={},
            model_maker=default_model,
            optimizer=keras.optimizers.SGD(lr=0.01),
            plot_images=5, save_image_location='genimage',
            auto_train=False, retrain=False, save_model_location='models',
            epochs=10):

        if epochs <= 0:
            auto_train = False
            retrain = False
            epochs = 0

        self._dataset = dataset
        self._name = 'D{}'.format(dataset.name)

        # Always include the identity transform.
        self._datagens = []
        for filter in [ImageFilter(), *extra_filters]:
            datagen = ImageDataGenerator(preprocessing_function=filter.filter_function, validation_split=0.1, **preprocess_params)
            datagen.fit(dataset.train_images)
            self._datagens.append(datagen)

            # Skip plotting & including the name of the identity transform.
            if len(self._datagens) > 1:
                self._name += '-F{}'.format(filter.name)
                if plot_images > 0:
                    has_preprocessing = preprocess_params != {}
                    processing.plot(dataset.train_images[:plot_images], datagen, filter.name, has_preprocessing, save_image_location)

        self._name += get_preprocessing_suffix(preprocess_params)
        self._name += '-M{}'.format(model_maker.__name__)
        self._name += get_optimizer_suffix(optimizer)
        # TODO pre-trained models should also get the epoch count in their name

        # TODO Consider commonizing this code instead of copying it from Claw
        self._model = None
        if save_model_location != None and save_model_location != '':
            self._save_model_location = save_model_location
            if auto_train and not retrain:
                epoch_checkpoint = epochs
                while epoch_checkpoint > 0:
                    try:
                        self._model = load_model(self.get_model_filepath(epoch_checkpoint))
                        break
                    except:
                        epoch_checkpoint -= 1
                if epoch_checkpoint == 0:
                    print('***Error reading model, creating new one.***')
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

        print('Model name (with epoch count): {}'.format(self.get_model_filepath(epochs)))

        if retrain or epoch_checkpoint < epochs:
            for current_epoch in range(epoch_checkpoint, epochs):
                print('Training epoch {} (started at {})'.format(current_epoch + 1, epoch_checkpoint))
                self.fit(epochs=current_epoch + 1, initial_epoch=current_epoch)
                if save_model_location != None and save_model_location != '':
                    self.save_model(current_epoch + 1)

    # TODO most of these are copied from Claw, ugly, horrible...

    @property
    def model(self):
        """The underlying Keras model."""
        return self._model

    @property
    def datagen(self):
        """
        """
        return self._datagens[0]

    @property
    def datagens(self):
        """
        The list of data generators that apply filters & preprocessing to all data.
        Any other models/attacks on the same dataset must pass data through this!
        """
        return self._datagens

    @property
    def name(self):
        """
        The name of the owned Keras model, auto-generated from its properties.
        Should be unique for each model.
        """
        return self._name

    def get_model_filepath(self, epoch_checkpoint=0):
        """The path that helper methods will use to save/load the owned Keras model."""
        if epoch_checkpoint > 0:
            return '{}/{}-e{}.h5py'.format(self._save_model_location, self._name, epoch_checkpoint)
        else:
            return '{}/{}.h5py'.format(self._save_model_location, self._name)

    def save_model(self, epochs=0):
        """
        Convenience method to save the owned Keras model with a name based on its properties.
        If a save location was passed to the constructor, it will be saved there.
        Otherwise, will save to the current working directory.

        Can still call 'model.save(path)' directly.
        """
        os.makedirs(self._save_model_location, exist_ok=True)
        self._model.save(self.get_model_filepath(epochs))


    def fit(self, x_train=None, y_train=None, epochs=0, initial_epoch=0):
        """
        """
        if x_train is None:
            x_train = self._dataset.train_images
        if y_train is None:
            y_train = self._dataset.train_labels

        if epochs < 1:
            epochs = self._epochs

        latest_history = None
        for datagen in self._datagens:
            train_generator = datagen.flow(
                x_train,
                y_train,
                subset="training",
                batch_size=generator_batch_size)

            # Get the number of samples in the training subset
            num_train = len(train_generator.x)

            validation_generator = datagen.flow(
                x_train,
                y_train,
                subset="validation",
                batch_size=generator_batch_size)

            # Get the number of samples in the validation subset
            num_validation = len(validation_generator.x)

            latest_history = self._model.fit_generator(
                train_generator,
                steps_per_epoch=num_train/generator_batch_size,
                validation_data=validation_generator,
                validation_steps=num_validation/generator_batch_size,
                epochs=epochs,
                initial_epoch=initial_epoch,
                workers=generator_workers)

        return latest_history

    def evaluate(self, x_test=None, y_test=None):
        if x_test is None:
            x_test = self._dataset.test_images
        if y_test is None:
            y_test = self._dataset.test_labels

        predictions = self.predict(x_test)

        num_correct = np.sum(
                        np.argmax(predictions, axis=1) ==
                        np.argmax(y_test, axis=1))
        num_images = len(y_test)

        return num_correct / num_images

    def predict(self, x):
        # TODO need a merging method...can probably train on that too, like an ensemble.
        # For now just use average, or majority vote.
        predict_generator = self._datagens[0].flow(x, batch_size=generator_batch_size, shuffle=False)
        return self._model.predict_generator(predict_generator, steps=len(x)/generator_batch_size)


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
