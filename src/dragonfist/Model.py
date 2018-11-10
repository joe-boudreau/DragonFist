from tensorflow import keras


class Model:
    def __init__(self, layers):
        self._layers = layers

        self._optimizer = keras.optimizers.SGD(
            lr=0.01,
            decay=1e-6,
            momentum=0.9,
            nesterov=True
        )
        self._loss = keras.losses.categorical_crossentropy
        self._metrics = [keras.metrics.categorical_accuracy]

        self._epochs = 10
        self._batch_size = 32
        self._workers = 1

        self._image_filter = None

        self._test_loss = 0
        self._test_accuracy = 0

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def loss(self):
        return self._loss

    @property
    def metrics(self):
        return self._metrics

    @property
    def epochs(self):
        return self._epochs

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def workers(self):
        return self._workers

    @property
    def image_filter(self):
        return self._image_filter

    @property
    def test_loss(self):
        return self._test_loss

    @property
    def test_accuracy(self):
        return self._test_accuracy

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @loss.setter
    def loss(self, value):
        self._loss = value

    @metrics.setter
    def metrics(self, value):
        self._metrics = value

    @epochs.setter
    def epochs(self, value):
        self._epochs = value

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value

    @workers.setter
    def workers(self, value):
        self._workers = value

    @image_filter.setter
    def image_filter(self, value):
        self._image_filter = value

    def initialize(self):
        self._model = keras.Sequential(self._layers)

        self._model.compile(
            optimizer=self._optimizer,
            loss=self._loss,
            metrics=self._metrics
        )

    def fit(self):
        if self.is_data_generator():
            train_generator = self._image_filter.datagen.flow(
                self._image_filter.data.train_images,
                self._image_filter.data.train_labels,
                batch_size=self._batch_size
            )

            validation_generator = self._image_filter.datagen.flow(
                self._imag_filter.data.test_images,
                self._image_filter.data.test_labels,
                batch_size=self._batch_size
            )

            self._model.fit_generator(
                train_generator,
                validation_data=validation_generator,
                epochs=self._epochs,
                workers=self._workers
            )

            self.evaluate()
        else:
            self._model.fit(
                self._image_filter.filtered_train_images,
                self._image_filter.data.train_labels,
                epochs=self._epochs,
                validation_data=(
                    self._image_filter.filtered_test_images,
                    self._image_filter.data.test_labels
                )
            )

            self.evaluate()

    def evaluate(self):
        if self.is_data_generator():
            self._test_loss, self._test_accuracy = self._model.evaluate_generator(
                self._image_filter.datagen.flow(
                    self._image_filter.data.test_images,
                    self._image_filter.data.test_labels,
                    batch_size=self._batch_size
                )
            )
        else:
            self._test_loss, self._test_accuracy = self._model.evaluate(self._image_filter.filtered_test_images,
                                                                        self._image_filter.data.test_labels, )

    def predict(self, x):
        return self._model.predict_on_batch(x)

    #TODO: Need to implement saving and loading functionality for this Model class to prevent having to retrain
    def load(self, file_name):
        self._model = keras.models.load_model(file_name)

    def save(self, file_name):
        self._model.save(file_name)

    def is_data_generator(self):
        return self._image_filter.datagen is not None
