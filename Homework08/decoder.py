import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers


class Decoder(Model):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = tf.keras.Sequential([
        #self.decoder = [
            layers.Dense(49, activation="sigmoid"),
            layers.Reshape((7, 7, 1)),
            layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same', kernel_initializer='random_normal'),
            layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same', kernel_initializer='random_normal'),
            layers.Conv2D(1, kernel_size= 3, activation='sigmoid', padding='same')
        ])

    def call(self, x, training):
        #print('decoder')
        #for layer in self.decoder:
        #    x = layer(x)
        #    print(x.shape)
        x = self.decoder(x, training)
        #tf.print(f"After decoder: {tf.reduce_max(x)}")
        return x
