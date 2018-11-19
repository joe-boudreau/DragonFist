from keras import datasets

import cleverhans
from cleverhans.attacks import FastGradientMethod, SaliencyMapMethod
from cleverhans.utils_keras import KerasModelWrapper

from skimage import filters
from skimage.color.adapt_rgb import each_channel, hsv_value

from matplotlib import pyplot as plt
import numpy as np

import os

import transformations as tr

from data import DataSet
from processing import ImageProcessParams
from model import Claw, Fist
from model_makers import *


# TODO update
def attackFGM(model, datagen, plot_images=5, save_image_location='atkimageFGM'):

    fgsm = FastGradientMethod(KerasModelWrapper(model), keras.backend.get_session())
    fgsm_params = {'eps': 0.3,
                   'clip_min': 0.,
                   'clip_max': 1.}


    image_filter = datagen.preprocessing_function
    model_name = get_model_name(image_filter)

    if plot_images > 0:
        save_image_location = create_image_folder(save_image_location, model_name)

        # NOTE: Use a different generator here to show only the effects of the filter,
        #       and not of any other preprocessing step.
        filtergen = ImageDataGenerator(preprocessing_function=image_filter)
        adversarial_images = fgsm.generate_np(test_images[0:plot_images], **fgsm_params)
        image_generator = filtergen.flow(adversarial_images,
                                         batch_size=1, shuffle=False, save_to_dir=save_image_location)

        i = 0
        fig, axs = plt.subplots(2, plot_images)
        for batch_images in image_generator:
            ax = axs[0,i]
            ax.imshow(ensure_is_plottable(test_images[i]))
            ax.set_axis_off()
            ax = axs[1,i]
            ax.imshow(ensure_is_plottable(batch_images[0]))
            ax.set_axis_off()

            i = i+1
            if i == plot_images:
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
                    datagen.flow(adversarial_images, batch_size=generator_batch_size),
                    steps=len(adversarial_images)/generator_batch_size)
        num_correct += np.sum(
                        np.argmax(preds, axis=1) ==
                        np.argmax(test_labels[i_start:i_end], axis=1))

        if i_end == num_images:
            break
        else:
            i_start+=adv_batch_size

    adv_acc = num_correct/num_images
    print('Adversarial accuracy (FGM): {0:.2f}%'.format(adv_acc*100))


# TODO update
def attackJSM(model, datagen, source_samples=3, save_image_location='atkimageJSM'):

    image_filter = datagen.preprocessing_function
    model_name = get_model_name(image_filter)
    save_image_location = create_image_folder(save_image_location, model_name)

    print('Crafting ' + str(source_samples) + ' * ' + str(num_classes - 1) +
          ' adversarial examples')

    # Keep track of success (adversarial example classified in target)
    results = np.zeros((num_classes, source_samples), dtype='i')

    # Rate of perturbed features for each test set example and target class
    perturbations = np.zeros((num_classes, source_samples), dtype='f')

    # Initialize our array for grid visualization
    grid_shape = (num_classes, num_classes) + input_shape
    grid_viz_data = np.zeros(grid_shape, dtype='f')

    # Instantiate a SaliencyMapMethod attack object
    jsma = SaliencyMapMethod(KerasModelWrapper(model), keras.backend.get_session())
    jsma_params = {'theta': 1., 'gamma': 0.1,
                   'clip_min': 0., 'clip_max': 1.,
                   'y_target': None}

    figure = None
    # Loop over the samples we want to perturb into adversarial examples
    # NOTE This is just one image at a time, but dimensionalities allow
    #      for multiple images at a time.
    # TODO Should have a version of this that generates multiple attack images at once.
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
        filtered_test_image = image_filter(np.squeeze(samples))

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
            filtered_adversarial_image = image_filter(np.squeeze(adversarial_images))

            # NOTE batch_size=1 and steps=1 is only for one-at-a-time attacks
            preds = model.predict_generator(
                        datagen.flow(adversarial_images, batch_size=1, save_to_dir=save_image_location),
                        steps=1)

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
    print('Avg. rate of successful adv. examples {0:.4f}'.format(succ_rate))

    # Compute the average distortion introduced by the algorithm
    percent_perturbed = np.mean(perturbations)
    print('Avg. rate of perturbed features {0:.4f}'.format(percent_perturbed))

    # Compute the average distortion introduced for successful samples only
    percent_perturb_succ = np.mean(perturbations * (results == 1))
    print('Avg. rate of perturbed features for successful '
          'adversarial examples {0:.4f}'.format(percent_perturb_succ))

    # Finally, block & display a grid of all the adversarial examples
    plt.close(figure)
    cleverhans.utils.grid_visual(grid_viz_data)


def get_model_name(dataset, image_filter, epochs):
    name = dataset.name + '-' + image_filter.func.__name__
    for key, value in image_filter.keywords.items():
        name += '-{}_{}'.format(key, value)
    name += '-e{}'.format(epochs)
    return name

# ---------------------------- #

def test_claw(claw):
    test_loss, test_acc = claw.evaluate()
    print('Test accuracy: {:.2f}%'.format(test_acc*100))


dataset = DataSet.load_from_keras(keras.datasets.fashion_mnist)

test_claw(Claw(dataset,
    auto_train=True,
    epochs=1))

test_claw(Claw(dataset,
    ImageProcessParams(filters.gaussian, {'sigma':0.5}),
    auto_train=True,
    epochs=1))

test_claw(Claw(dataset,
    ImageProcessParams(filters.sobel, {}, tr.compat2d, {'zca_whitening':True}),
    auto_train=True,
    epochs=1))


dataset = DataSet.load_from_keras(keras.datasets.cifar10)

test_claw(Claw(dataset,
    auto_train=True,
    epochs=1))

test_claw(Claw(dataset,
    ImageProcessParams(filters.gaussian, {'sigma':1.5}),
    auto_train=True,
    epochs=1))

test_claw(Claw(dataset,
    ImageProcessParams(filters.sobel, {}, each_channel, {'zca_whitening':True}),
    auto_train=True,
    epochs=1))

test_claw(Claw(dataset,
    ImageProcessParams(filters.sobel, {}, hsv_value, {'zca_whitening':True}),
    auto_train=True,
    epochs=1))

test_claw(Claw(dataset,
    ImageProcessParams(filters.sobel, {}, tr.to_gray, {'zca_whitening':True}),
    auto_train=True,
    epochs=1))
