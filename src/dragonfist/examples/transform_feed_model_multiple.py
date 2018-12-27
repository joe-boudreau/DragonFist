import tensorflow as tf
from tensorflow import keras

import json
from multiprocessing.dummy import Pool as ThreadPool

import transformations

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

num_classes  = max(train_labels)+1
train_labels = keras.utils.to_categorical(train_labels, num_classes)
test_labels  = keras.utils.to_categorical(test_labels, num_classes)

image_filters = [
    {
        'name': 'Identity',
        'function': transformations.identity
    },
    {
        'name': 'Edge detection',
        'function': transformations.edge_detection
    }
]

def create_model(image_filter):
    input_shape = train_images.shape[1:]

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

    return model

def thread_wrapper(image_filter):
    model = create_model(image_filter['function'])

    filtered_test_images = image_filter['function'](test_images)

    test_loss, test_acc = model.evaluate(filtered_test_images, test_labels)

    return '{0} - Test accuracy: {1}'.format(image_filter['name'], test_acc)

def main():
    pool = ThreadPool(len(image_filters))

    results = pool.map(thread_wrapper, image_filters)

    print(results)

main()
