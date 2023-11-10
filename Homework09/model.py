"""
model

Created: 22.01.22, 12:25

Author: LDankert
"""
from abc import ABC

import tensorflow as tf
from tensorflow.keras import layers


class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.input_layer = tf.keras.Input(shape=(28,28,1))
        self.all_layers = [
            # layers.InputLayer(batch_size, 28, 28, 1),
            layers.Conv2D(32, (3, 3), strides=2, activation='relu', padding='same', input_shape=(28, 28, 1)),
            layers.MaxPool2D(2, 2),

            layers.Conv2D(64, (3, 3), strides=2, activation='relu', padding='same'),
            layers.MaxPool2D(2, 2),

            layers.Flatten(),

            layers.Dense(512, activation='relu'),

            layers.Dense(1, activation='sigmoid')
        ]
        self.out = self.call(self.input_layer, training = True)

    def call(self, x, training):
        for layer in self.all_layers:
            try:
                x = layer(x, training)
            except:
                x = layer(x)
        return x


class Generator(tf.keras.Model):

    def __init__(self):
        super(Generator, self).__init__()
        self.input_layer = layers.Input(shape=(100))
        self.all_layers = [
            layers.Dense(7 * 7 * 128, use_bias=False),
            layers.Reshape((7, 7, 128)),
            layers.BatchNormalization(),
            layers.LeakyReLU(),

            layers.Conv2DTranspose(64, (3, 3), strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Dropout(0.2),

            layers.Conv2D(128, (3, 3), strides=1, padding='same'),

            layers.Conv2DTranspose(32, (3, 3), strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Dropout(0.2),

            layers.Conv2D(16, (3, 3), strides=1, padding='same'),

            layers.Conv2D(1, (3, 3), strides=1, padding='same', activation='tanh')
        ]
        self.out = self.call(self.input_layer, training = True)

    def call(self, x, training):
        for layer in self.all_layers:
            try:
                x = layer(x, training)
            except:
                x = layer(x)
        return x
