"""
Multi Layer Perceptron Class

Created: 01.11.21, 20:22

Author: LDankert
"""

from perceptron import Perceptron, sigmoid


class MultiLayerPerceptron:

    # Constructor, need a list of perceptrons
    def __init__(self, perceptrons):
        self.perceptrons = perceptrons
        self.outputs = Perceptron(len(self.perceptrons))

    # Inputs passed through the networks
    def forward_step(self, inputs):
        hidden_activations = []
        for perceptron in self.perceptrons:
            perceptron.forward_step(inputs)
            hidden_activations.append(perceptron.activation)
        self.outputs.forward_step(hidden_activations)

    # Update the network parameters
    # We think we have here a mistake, we maybe should use the net input not the activation for the delta!!
    def backprob_step(self, target):
        # First calc the output depending of the ground truth and update the output perceptron
        deltaOutput = (self.outputs.activation - target) * sigmoidprime(self.outputs.activation)
        self.outputs.update(deltaOutput)
        # Now update all perceptrons with delta depending on first delta
        for perceptron in self.perceptrons:
            delta = sum(deltaOutput * perceptron.weights) * sigmoidprime(perceptron.activation)
            perceptron.update(delta)

    # Output perceptron setter weigths has to match perceptrons
    def set_outputs(self,perceptron):
        if len(perceptron.weights) != len(self.perceptrons):
            raise Exception ("Weights of output perceptron does not match number of perceptrons")
        else:
            self.outputs = perceptron

    # Perceptron setter, needs a list of Perceptrons, generates a fitting output perceptron too
    def set_perceptrons(self,perceptrons):
        self.perceptrons = perceptrons
        self.outputs = Perceptron(len(self.perceptrons))

    # Iterates over all perceptrons
    def perceptron_generator(self):
        for perceptron in self.perceptrons:
            yield perceptron


# simple derivative of the sigmoid function
def sigmoidprime(x):
    s = sigmoid(x)
    return s * (1 - s)
