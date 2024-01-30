from encoder import Encoder
from decoder import Decoder
import tensorflow as tf
from tensorflow.keras import Model


class Autoencoder(Model):

    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def call(self, x, training):
        x = self.encoder(x, training)
        x = self.decoder(x, training)
        return x