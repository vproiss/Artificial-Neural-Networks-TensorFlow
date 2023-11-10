"""
Modul for training function

Created: 06.11.21, 22:14

Author: VProiss
"""

import numpy as np
from perceptron import Perceptron
from multi_layer_perceptron import MultiLayerPerceptron


# Needs an list of inputs, dict of gates and number of epochs
# Returns: dict of all loses and dict of all averages
def training(inputs, Gates, epochs):
    overall_loses = {}
    overall_accuracy = {}
    # Train 4 different MLP for each target
    for name, gate in Gates.items():
        # creates a MLP with 4 Perceptrons with 2 weights to train
        mlp = MultiLayerPerceptron([Perceptron(2) for i in range(4)])
        print(f"The logical gate is {name}")
        collected_losses = []
        collected_accuracy = []
        for epoch in range(epochs):
            losses = 0
            correct = 0
            # every input and the according value from the specific gate
            for input, target in zip(inputs, gate):
                mlp.forward_step(input)
                mlp.backprob_step(target)
                # Calculate the average error with mean squared error
                loss = (target- mlp.outputs.activation) ** 2
                losses += loss
                # Check if target is hit
                if round(mlp.outputs.activation) == target:
                    correct += 1
            # Loss average for this epoch
            collected_losses.append(losses / len(inputs))
            # Accuracy for this epoch
            collected_accuracy.append(correct / len(inputs))
        overall_loses[name] = collected_losses
        overall_accuracy[name] = collected_accuracy
    return overall_loses, overall_accuracy

