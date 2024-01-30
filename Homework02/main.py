import numpy as np
import matplotlib.pyplot as plt
from training import training
from perceptron import Perceptron
from multi_layer_perceptron import MultiLayerPerceptron

# input array
inputs = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])

# gates
gate_and =  np.array([1,0,0,0])
gate_or =   np.array([1,1,1,0])
gate_nand = np.array([0,1,1,1])
gate_nor =  np.array([0,0,0,1])
gate_xor =  np.array([0,1,1,0])

all_gates = {"and": gate_and, "or": gate_or, "nand": gate_nand, "nor": gate_nor, "xor": gate_xor}

# Training Area
epochs = 1000
losses, accuracies = training(inputs, all_gates, epochs)

# Visualisation
for gate in accuracies.keys():
    x = range(epochs)
    plt.plot(x, accuracies[gate], label = "Accuracy")
    plt.plot(x, losses[gate], label = "Loss")
    plt.title(gate)
    plt.legend()
    plt.show()
