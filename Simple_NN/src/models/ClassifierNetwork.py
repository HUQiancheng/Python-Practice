import numpy as np
from BaseNetwork import BaseNetwork

class ClassifierNetwork(BaseNetwork):
    def __init__(self, layer_dims):
        """
        Initialize the network layers.
        Args:
            layer_dims (list): List containing the dimensions of each layer.
        """
        super().__init__()
        self.layer_dims = layer_dims
        self.initialize_parameters()

    def initialize_parameters(self):
        """
        Initialize weights and biases for each layer of the network.
        """
        np.random.seed(1)  # Ensure consistent initialization for reproducibility
        for i in range(1, len(self.layer_dims)):
            self.parameters['W' + str(i)] = np.random.randn(self.layer_dims[i], self.layer_dims[i - 1]) * 0.01
            self.parameters['b' + str(i)] = np.zeros((self.layer_dims[i], 1))

    def apply_activation(self, x):
        """
        Apply the sigmoid activation function.
        Args:
            x (np.array): Linear combination from a layer's output before activation.
        Returns:
            np.array: Activated output using the sigmoid function.
        """
        return 1 / (1 + np.exp(-x)) # Sigmoid activation

    def forward(self, x):
        """
        Implement forward propagation for the MLP using sigmoid activation.
        Args:
            x (np.array): Input data.
        Returns:
            np.array: The final output of the network.
        """
        activations = x
        self.activations = {'A0': activations}  # Store initial input

        # Forward pass through each layer
        L = len(self.layer_dims) - 1
        for l in range(1, L + 1):
            Z = np.dot(self.parameters['W' + str(l)], activations) + self.parameters['b' + str(l)]
            activations = self.apply_activation(Z)
            self.activations['A' + str(l)] = activations

        return activations

    def backward(self, y_true):
        """
        Implement backward propagation for the MLP.
        Args:
            y_true (np.array): True binary labels.
        Returns:
            dict: Gradients of weights and biases.
        """
        gradients = {}
        L = len(self.layer_dims) - 1  # Total number of layers

        # Output layer gradient using binary cross-entropy loss derivative with sigmoid
        dZL = self.activations['A' + str(L)] - y_true
        gradients['dW' + str(L)] = np.dot(dZL, self.activations['A' + str(L - 1)].T) / y_true.shape[1]
        gradients['db' + str(L)] = np.sum(dZL, axis=1, keepdims=True) / y_true.shape[1]

        # Backpropagate through hidden layers
        for l in reversed(range(1, L)):
            dA_prev = np.dot(self.parameters['W' + str(l + 1)].T, dZL)
            dZ = dA_prev * (self.activations['A' + str(l)] * (1 - self.activations['A' + str(l)]))
            gradients['dW' + str(l)] = np.dot(dZ, self.activations['A' + str(l - 1)].T) / y_true.shape[1]
            gradients['db' + str(l)] = np.sum(dZ, axis=1, keepdims=True) / y_true.shape[1]
            dZL = dZ

        return gradients
