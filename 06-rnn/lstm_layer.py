import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer
from lstm_cell import LSTM_Cell


class LSTM_Layer(Layer):

    def __init__(self, cells):
        super(LSTM_Layer, self).__init__()
        self.cells = cells

    def call(self, x, states):
        x = tf.transpose(x, [1,0,2])
        max_seq_len = x.shape[0]
        new_stats = tf.TensorArray(tf.float32, size=max_seq_len)
        state = states
        for i in range(max_seq_len):
            state = self.cells(x[i],state)
            new_stats = new_stats.write(i, state)
        #return tf.transpose(new_stats.stack(), [1,0,2,3])
        return new_stats.stack()

    def zero_states(self, batch_size):
        return tf.zeros([batch_size, self.cells.units]), tf.zeros([batch_size, self.cells.units])
