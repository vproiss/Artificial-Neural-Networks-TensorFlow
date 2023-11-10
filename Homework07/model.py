"""
The model class

Created: 11.12.21, 23:52

Author: LDankert
"""

from lstm_cell import LSTM_Cell
from lstm_layer import LSTM_Layer
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
import tensorflow as tf


class LSTM_Model(Model):

    def __init__(self):
        super(LSTM_Model, self).__init__()
        #self.cell_layers = [LSTM_Layer(LSTM_Cell(128)), LSTM_Layer(LSTM_Cell(128))]
        self.layer = LSTM_Layer(LSTM_Cell(128))
        self.outputs = Dense(1, use_bias=False, activation="softmax")
        #self.outputs = Dense(1, use_bias=False, activation= (lambda x: tf.round(tf.nn.sigmoid(x))))

    @tf.function
    def call(self,x):
        states = self.layer.zero_states(x.shape[0])
       # for layer in self.cell_layers:
        #    states = layer.zero_states(x.shape[0])
         #   x = layer(x, states)
        x = self.layer(x, states)
        #x = self.layer2(x, states)
        x = self.outputs(x)
        return x

