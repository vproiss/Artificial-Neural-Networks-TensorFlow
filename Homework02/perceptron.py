"""
perceptron class

Created: 01.11.21, 20:04

Author: LDankert
"""
import numpy
import numpy as np


class Perceptron:

    # Constructor
    def __init__(self, input_units):
        self.weights = np.random.randn(input_units)
        self.bias = np.random.randn()
        self.alpha = 1
        self.activation = 0

    # Calculates the activation of the perceptron, takes 1d array
    def forward_step(self, inputs):
        # raise an exception if the number of inputs not match number of weights
        if np.size(inputs) != np.size(self.weights):
            raise Exception ("Inputs number doesn't match weight numbers")
        else:
            # multiply the weights with corresponding inputs and add bias
            net_input = np.sum((inputs*self.weights)) + self.bias
            self.activation = sigmoid(net_input)

    # Updates the parameters of the perceptron
    def update(self, delta):
        gradient = delta * self.activation
        self.weights = self.weights - self.alpha * gradient
        self.bias = self.bias - self.alpha * gradient

    # Setter functions
    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_weights(self, weights):
        self.weights = weights

    def set_bias(self, bias):
        self.bias = bias

# simple sigmoid function
def sigmoid(x):
    return 1/(1 + np.exp(-x))
