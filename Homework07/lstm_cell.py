"""
LSTM cell Class

Created: 11.12.21, 19:04

Author: LDankert
"""
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense


class LSTM_Cell(Layer):

    # initialize 4 gates: forget
    def __init__(self, units):
        super(LSTM_Cell, self).__init__()
        self.units = units
        # forget gate with activation function sigmoid and 1 as bias
        self.forgetGate = Dense(self.units, activation="sigmoid", bias_initializer='Ones')
        # input gate with activation function sigmoid
        self.inputGate = Dense(self.units, activation="sigmoid")
        # output gate with activation function sigmoid
        self.outputGate = Dense(self.units, activation="sigmoid")
        # cell state candidates gate with activation function tangents hyperbolic
        self.cellStateCandidates = Dense(self.units, activation="tanh")

    # the call function, takes the input x and the states from the previous time step as tuple
    def call(self, x, states):
        # splits the tuple
        hidden_state, cell_state = states
        # concatenate the h_t-1 and the input x
        input_states = tf.concat([hidden_state, x], axis=-1)
        # calculate the next cell state from the forgetGate, inputGate
        next_cell_state = tf.multiply(self.forgetGate(input_states), cell_state) + \
                          tf.multiply(self.inputGate(input_states), self.cellStateCandidates(input_states))
        # calculate the next hidden state from the output gate and the next cell state
        next_hidden_state = tf.multiply(self.outputGate(input_states), tf.math.tanh(next_cell_state))

        return next_hidden_state, next_cell_state
