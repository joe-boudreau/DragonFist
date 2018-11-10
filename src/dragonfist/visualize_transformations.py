from tensorflow import keras

from matplotlib import pyplot as plt, cm

from src.dragonfist import transformations as ts

"""A convenience script to test and visualize image filters quickly"""

#images = keras.datasets.fashion_mnist.load_data()[0][0]
images = keras.datasets.cifar10.load_data()[0][0]
images = images[:5]


filtered_images = ts.multi(ts.increase_contrast_saturated, images)

plt.figure(1, figsize=(10, 10))
for i in range(5):
    plt_i = i+1
    plt.subplot(2, 5, plt_i).set_axis_off()
    plt.imshow(images[i], cmap=cm.gray_r)
    plt.subplot(2, 5, 5+plt_i).set_axis_off()
    plt.imshow(filtered_images[i], cmap=cm.gray_r)

plt.show()