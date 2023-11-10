"""
Main file of homework 09

Created: 15.01.22, 11:24

Author: LDankert
"""

import os
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import PIL
from tensorflow.keras import layers
import time

from IPython import display
from preprocessing import preprocessing_data
from model import Discriminator, Generator
from training_step import training_step

import urllib

batch_size = 128
latent_space = 100

categories = [line.rstrip(b'\n') for line in urllib.request.urlopen('https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/master/categories.txt')]
print(categories[:10])
category = 'candle'

# Creates a folder to download the original drawings into.
# We chose to use the numpy format : 1x784 pixel vectors, with values going from 0 (white) to 255 (black). We reshape them later to 28x28 grids and normalize the pixel intensity to [-1, 1]

if not os.path.isdir('npy_files'):
    os.mkdir('npy_files')

if not os.path.exists(f'npy_files/{category}.npy'):
    url = f'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{category}.npy'
    urllib.request.urlretrieve(url, f'npy_files/{category}.npy')

images = np.load(f'npy_files/{category}.npy')
print(f'{len(images)} images to train on')

# You can limit the amount of images you use for training by setting :
train_images = images[:10000]
# You should also define a samller subset of the images for testing..
test_images = images[10001:11000]


# Notice that this to numpy format contains 1x784 pixel vectors, with values going from 0 (white) to 255 (black). We reshape them later to 28x28 grids and normalize the pixel intensity to [-1, 1]

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, np.ones(len(train_images))))
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, np.ones(len(test_images))))

train_dataset = preprocessing_data(train_dataset, batch_size)
test_dataset = preprocessing_data(test_dataset, batch_size)

number_of_epochs = 20
learning_rate = 0.001

test_disc = Discriminator()
x = test_disc(tf.zeros([1,28,28,1]), training=True)
test_disc.summary()

test_gen = Generator()
noise = tf.random.normal(shape=[1,100])
image = test_gen(noise, training=False)
test_gen.summary()

plt.imshow(image[0, :, :, 0], cmap='gray')

test_disc(image).numpy()

loss_function = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate)

discriminator = Discriminator()
generator = Generator()

discriminator_losses = np.empty(0)
generator_losses = np.empty(0)

for epoch in range(number_of_epochs):
    print(f'Epoch {epoch} is running')

    discriminator_loss, generator_loss, generated_images = training_step(train_dataset, generator, discriminator, batch_size, latent_space, loss_function, optimizer)
    discriminator_losses = np.append(discriminator_losses, discriminator_loss)
    generator_losses = np.append(generator_losses, generator_loss)

    ncols = int(number_of_epochs / 10)
    fig = plt.figure(figsize=(12, 12), tight_layout=False)
    for i in range(0, len(generated_images), ncols):
        plt.subplot(1, int(len(generated_images) / ncols), 1 if i == 0 else int((i / ncols) + 1),
                    title=f"Epoch {i}")
        plt.imshow(generated_images[i][0, :, :], cmap='gray_r')
        plt.axis('off')

plt.figure()
line1, = plt.plot(discriminator_losses)
line2, = plt.plot(generator_losses)
plt.xlabel("Training steps")
plt.ylabel("Losses")
plt.legend((line1, line2), ("Discriminator loss", "Generator loss"))
plt.show()
