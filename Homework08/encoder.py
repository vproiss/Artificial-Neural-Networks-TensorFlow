"""
The encoder class

Created: 14.12.21, 22:11

Author: LDankert
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers


class Encoder(Model):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(16, 3, activation='relu', padding='same', strides=2, kernel_initializer='random_normal'),
            layers.Conv2D(8, 3, activation='relu', padding='same', strides=2, kernel_initializer='random_normal'),
            layers.Flatten(),
            layers.Dense(10, activation='relu')
        ])

    def call(self, x, training):
        x = self.encoder(x, training)
        #tf.print(f"After encoder: {tf.reduce_max(x)}")
        return x
