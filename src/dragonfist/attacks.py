from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import functools
import logging

import cleverhans
import keras
import numpy as np
import tensorflow as tf
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import SaliencyMapMethod
from cleverhans.attacks_tf import jacobian_graph, jacobian_augmentation
from cleverhans.evaluation import batch_eval
from cleverhans.initializers import HeReLuNormalInitializer
from cleverhans.loss import CrossEntropy
from cleverhans.model import Model
from cleverhans.train import train
from cleverhans.utils import TemporaryLogLevel, set_log_level
from cleverhans.utils import to_categorical
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.utils_tf import model_eval
from cleverhans_tutorials.tutorial_models import ModelBasicCNN
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import cm
from matplotlib import pyplot as plt
from six.moves import xrange
from tensorflow.python.platform import flags

from image_utils import *


def attackFGM(entity, dataset, plot_images=5, save_image_location='atkimageFGM', ensemble=None):
    fgsm = FastGradientMethod(KerasModelWrapper(entity.model), keras.backend.get_session())
    fgsm_params = {'eps': 0.3,
                   'clip_min': 0.,
                   'clip_max': 1.}

    # TODO slicing
    test_images = dataset.test_images
    test_labels = dataset.test_labels

    if plot_images > 0:
        save_image_location = create_image_folder(save_image_location, entity.name)

        # NOTE: Use a different generator here to show only the effects of the filter,
        #       and not of any other preprocessing step.
        filtergen = ImageDataGenerator(preprocessing_function=entity.datagen.preprocessing_function)
        adversarial_images = fgsm.generate_np(test_images[0:plot_images], **fgsm_params)
        image_generator = filtergen.flow(adversarial_images,
                                         batch_size=1, shuffle=False, save_to_dir=save_image_location)

        # If plotting 2d images, use a grayscale color map
        cmap = get_cmap_for_images(test_images)

        i = 0
        fig, axs = plt.subplots(2, plot_images)
        for batch_images in image_generator:
            ax = axs[0,i]
            ax.imshow(ensure_is_plottable(test_images[i]), cmap=cmap)
            ax.set_axis_off()
            ax = axs[1,i]
            ax.imshow(ensure_is_plottable(batch_images[0]), cmap=cmap)
            ax.set_axis_off()

            i = i+1
            if i == plot_images:
                plt.show()
                break

    # Run in batches, otherwise it uses too much memory!
    generator_batch_size=32
    adv_batch_size = generator_batch_size*10
    num_images = len(test_images)
    num_correct = 0
    img_start = 0

    if ensemble != None:
        num_correct_ensemble = 0
        num_correct_other = []
        for i in range(ensemble.num_claws):
            num_correct_other.append(0)

    while True:
        img_end = min(img_start+adv_batch_size, num_images)
        adversarial_images = fgsm.generate_np(test_images[img_start:img_end], **fgsm_params)

        num_correct += np.sum(
                        np.argmax(entity.predict(adversarial_images), axis=1) ==
                        np.argmax(test_labels[img_start:img_end], axis=1))

        if ensemble != None:
            # Are adv. examples transferable to other models?
            for i in range(ensemble.num_claws):
                other_claw = ensemble.claws[i]
                if other_claw == entity: continue
                num_correct_other[i] += np.sum(
                                np.argmax(other_claw.predict(adversarial_images), axis=1) ==
                                np.argmax(test_labels[img_start:img_end], axis=1))

            # Are adv. examples transferable to the ensemble model?
            num_correct_ensemble += np.sum(
                        np.argmax(ensemble.predict(adversarial_images), axis=1) ==
                        np.argmax(test_labels[img_start:img_end], axis=1))

        if img_end == num_images:
            break
        else:
            img_start+=adv_batch_size

    adv_acc = num_correct/num_images
    print('Adversarial accuracy (FGM) on main  model ({0}): {1:.2f}%'.format(entity.name, adv_acc*100))

    if ensemble != None:
        for i in range(ensemble.num_claws):
            other_claw = ensemble.claws[i]
            if other_claw == entity: continue
            adv_acc_other = num_correct_other[i]/num_images
            print('Adversarial accuracy (FGM) on other model ({0}): {1:.2f}%'.format(other_claw.name, adv_acc_other*100))

        adv_acc_e = num_correct_ensemble/num_images
        print('Ensemble adversarial accuracy (FGM): {0:.2f}%'.format(adv_acc_e*100))


def attackJSM(entity, dataset, source_samples=3):
    num_classes = dataset.num_classes

    image_filter = entity.datagen.preprocessing_function

    print('Crafting ' + str(source_samples) + ' * ' + str(num_classes - 1) +
          ' adversarial examples')

    # Keep track of success (adversarial example classified in target)
    results = np.zeros((num_classes, source_samples), dtype='i')

    # Rate of perturbed features for each test set example and target class
    perturbations = np.zeros((num_classes, source_samples), dtype='f')

    # Initialize our array for grid visualization
    grid_shape = (num_classes, num_classes) + dataset.input_shape
    grid_viz_data = np.zeros(grid_shape, dtype='f')

    # Instantiate a SaliencyMapMethod attack object
    jsma = SaliencyMapMethod(KerasModelWrapper(entity.model), keras.backend.get_session())
    jsma_params = {'theta': 1., 'gamma': 0.1,
                   'clip_min': 0., 'clip_max': 1.,
                   'y_target': None}

    test_images = dataset.test_images
    test_labels = dataset.test_labels

    figure = None
    # Loop over the samples we want to perturb into adversarial examples
    # NOTE This is just one image at a time, but dimensionalities allow
    #      for multiple images at a time.
    # TODO Should have a version of this that generates multiple attack images at once. Or maybe not.
    encountered_classes = []
    attack_num = 0
    sample_ind = -1
    while attack_num < source_samples:
        sample_ind += 1
        # We want to find an adversarial example for each possible target class
        # (i.e. all classes that differ from the label given in the dataset)
        current_class = int(np.argmax(test_labels[sample_ind]))
        if current_class in encountered_classes:
            continue
        encountered_classes.append(current_class)

        print('--------------------------------------')
        print('Attacking input %i/%i' % (attack_num + 1, source_samples))
        samples = test_images[sample_ind:(sample_ind + 1)]
        filtered_test_image = image_filter(np.squeeze(samples)).reshape(dataset.input_shape)

        target_classes = cleverhans.utils.other_classes(num_classes, current_class)

        # For the grid visualization, keep original images along the diagonal
        grid_viz_data[current_class, current_class, :, :, :] = filtered_test_image

        # Loop over all target classes
        for target in target_classes:
            print('Generating adv. example for target class %i' % target)

            # This call runs the Jacobian-based saliency map approach
            one_hot_target = np.zeros((1, num_classes), dtype=np.float32)
            one_hot_target[0, target] = 1
            jsma_params['y_target'] = one_hot_target
            adversarial_images = jsma.generate_np(samples, **jsma_params)
            filtered_adversarial_image = image_filter(np.squeeze(adversarial_images)).reshape(dataset.input_shape)

            preds = entity.predict(adversarial_images)

            # Check if success was achieved
            res = np.sum(np.argmax(preds, axis=1) == target)

            # Compute number of modified features
            adversarial_image_reshape = adversarial_images.reshape(-1)
            test_in_reshape = test_images[sample_ind].reshape(-1)
            # NOTE Don't count extremely small perturbations, otherwise %changed can be nearly 100
            number_changed = np.where(np.abs(adversarial_image_reshape - test_in_reshape) > 1e-6)[0].shape[0]
            percent_perturb = float(number_changed) / adversarial_images.reshape(-1).shape[0]

            # Display the original and adversarial images side-by-side
            # NOTE Don't forget to filter them!
            figure = cleverhans.utils.pair_visual(filtered_test_image, filtered_adversarial_image, figure)

            # Add our adversarial example to our grid data
            grid_viz_data[target, current_class, :, :, :] = filtered_adversarial_image

            # Update the arrays for later analysis
            results[target, attack_num] = res
            perturbations[target, attack_num] = percent_perturb

        attack_num += 1

    print('--------------------------------------')

    # Compute the number of adversarial examples that were successfully found
    num_targets_tried = ((num_classes - 1) * source_samples)
    succ_rate = float(np.sum(results)) / num_targets_tried
    print('Avg. rate of successful adv. examples {0:.2f}'.format(succ_rate*100))

    # Compute the average distortion introduced by the algorithm
    percent_perturbed = np.mean(perturbations)
    print('Avg. rate of perturbed features {0:.2f}'.format(percent_perturbed*100))

    # Compute the average distortion introduced for successful samples only
    percent_perturb_succ = np.mean(perturbations * (results == 1))
    print('Avg. rate of perturbed features for successful '
          'adversarial examples {0:.2f}'.format(percent_perturb_succ*100))

    # Finally, block & display a grid of all the adversarial examples
    plt.close(figure)
    cleverhans.utils.grid_visual(grid_viz_data)


def back_to_black(model, dataset, batch_size=128, learning_rate=.001,
                  holdout=128, data_aug=6,
                  nb_epochs_s=10, lmbda=.1,
                  aug_batch_size=512):

    """
    You've been...thunderstruck

    Note: holdout needs to be a multiple of batch_size lol
    """

    set_log_level(logging.DEBUG)

    # Dictionary used to keep track and return key accuracies
    accuracies = {}

    # Create TF session
    sess = tf.Session()

    x_train, y_train = dataset.train_images, dataset.train_labels
    x_test, y_test = dataset.test_images, dataset.test_labels

    # Initialize substitute training set reserved for adversary
    x_sub = x_test[:holdout]
    y_sub = y_test[:holdout]
    y_sub_1D = np.argmax(y_sub, axis=1)

    # Redefine test set as remaining samples unavailable to adversaries
    x_test = x_test[holdout:]
    y_test = y_test[holdout:]

    # Obtain Image parameters
    img_rows, img_cols, nchannels = x_train.shape[1:4]
    nb_classes = y_train.shape[1]

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols,
                                          nchannels))
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))

    # Seed random number generator so tutorial is reproducible
    rng = np.random.RandomState([2018, 11, 23])

    bbox_preds = get_predictions_as_iterator(batch_size, model, x_train)

    # Test the accuracy on legitimate data
    accuracies['bbox_acc_legit'] = model.evaluate(x_sub, y_sub)


    # Train substitute using method from https://arxiv.org/abs/1602.02697
    print("Training the substitute model.")
    train_sub_out = train_sub(sess, x, y, bbox_preds, x_sub, y_sub_1D,
                              nb_classes, nb_epochs_s, batch_size,
                              learning_rate, data_aug, lmbda, aug_batch_size,
                              rng, img_rows, img_cols, nchannels)
    model_sub, preds_sub = train_sub_out

    # Evaluate the substitute model on clean test examples
    eval_params = {'batch_size': batch_size}
    acc = model_eval(sess, x, y, preds_sub, x_test, y_test, args=eval_params)
    accuracies['sub_acc_legit'] = acc

    # Initialize the Fast Gradient Sign Method (FGSM) attack object.
    fgsm_par = {'eps': 0.3, 'ord': np.inf, 'clip_min': 0., 'clip_max': 1.}
    fgsm = FastGradientMethod(model_sub, sess=sess)

    # Craft adversarial examples using the substitute
    x_adv_sub_array = fgsm.generate_np(x_sub, **fgsm_par)
    x_adv_sub_tensor = fgsm.generate(x, **fgsm_par)

    # Evaluate the substitute model on the adversarial test examples
    acc = model_eval(sess, x, y, model_sub.get_logits(x_adv_sub_tensor), x_test, y_test, args=eval_params)
    accuracies['sub_acc_adv'] = acc

    accuracy = model.evaluate(x_adv_sub_array, y_sub)
    accuracies['bbox_acc_adv'] = accuracy

    return accuracies


def get_predictions_as_iterator(batch_size, model, x_train):
    preds_tensor = tf.convert_to_tensor(model.predict(x_train))
    return tf.data.Dataset.from_tensor_slices(preds_tensor).batch(batch_size).make_one_shot_iterator().get_next()


class ModelSubstitute(Model):
    def __init__(self, scope, nb_classes, nb_filters=200, **kwargs):
        del kwargs
        Model.__init__(self, scope, nb_classes, locals())
        self.nb_filters = nb_filters

    def fprop(self, x, **kwargs):
        del kwargs
        my_dense = functools.partial(
            tf.layers.dense, kernel_initializer=HeReLuNormalInitializer)
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            y = tf.layers.flatten(x)
            y = my_dense(y, self.nb_filters, activation=tf.nn.relu)
            y = my_dense(y, self.nb_filters, activation=tf.nn.relu)
            logits = my_dense(y, self.nb_classes)
            return {self.O_LOGITS: logits,
                    self.O_PROBS: tf.nn.softmax(logits=logits)}


def train_sub(sess, x, y, bbox_preds, x_sub, y_sub, nb_classes,
              nb_epochs_s, batch_size, learning_rate, data_aug, lmbda,
              aug_batch_size, rng, img_rows=28, img_cols=28,
              nchannels=1):
    """
    This function creates the substitute by alternatively
    augmenting the training data and training the substitute.
    :param sess: TF session
    :param x: input TF placeholder
    :param y: output TF placeholder
    :param bbox_preds: output of black-box model predictions
    :param x_sub: initial substitute training data
    :param y_sub: initial substitute training labels
    :param nb_classes: number of output classes
    :param nb_epochs_s: number of epochs to train substitute model
    :param batch_size: size of training batches
    :param learning_rate: learning rate for training
    :param data_aug: number of times substitute training data is augmented
    :param lmbda: lambda from arxiv.org/abs/1602.02697
    :param rng: numpy.random.RandomState instance
    :return:
    """
    # Define TF model graph (for the black-box model)
    model_sub = ModelSubstitute('model_s', nb_classes)
    preds_sub = model_sub.get_logits(x)
    loss_sub = CrossEntropy(model_sub, smoothing=0)

    print("Defined TensorFlow model graph for the substitute.")

    # Define the Jacobian symbolically using TensorFlow
    grads = jacobian_graph(preds_sub, x, nb_classes)

    # Train the substitute and augment dataset alternatively
    for rho in xrange(data_aug):
        print("Substitute training epoch #" + str(rho))
        train_params = {
            'nb_epochs': nb_epochs_s,
            'batch_size': batch_size,
            'learning_rate': learning_rate
        }
        with TemporaryLogLevel(logging.WARNING, "cleverhans.utils.tf"):
            train(sess, loss_sub, x_sub, to_categorical(y_sub, nb_classes),
                  init_all=False, args=train_params, rng=rng,
                  var_list=model_sub.get_params())

        # If we are not at last substitute training iteration, augment dataset
        if rho < data_aug - 1:
            print("Augmenting substitute training data.")
            # Perform the Jacobian augmentation
            lmbda_coef = 2 * int(int(rho / 3) != 0) - 1
            x_sub = jacobian_augmentation(sess, x, x_sub, y_sub, grads,
                                          lmbda_coef * lmbda, aug_batch_size)

            print("Labeling substitute training data.")
            # Label the newly generated synthetic points using the black-box
            y_sub = np.hstack([y_sub, y_sub])
            x_sub_prev = x_sub[int(len(x_sub)/2):]
            bbox_val = batch_eval(sess, [x], [bbox_preds], [x_sub_prev], batch_size=batch_size)[0]
            # Note here that we take the argmax because the adversary
            # only has access to the label (not the probabilities) output
            # by the black-box model
            y_sub[int(len(x_sub)/2):] = np.argmax(bbox_val, axis=1)

    return model_sub, preds_sub
