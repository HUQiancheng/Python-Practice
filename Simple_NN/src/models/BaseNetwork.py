import numpy as np
from abc import ABC, abstractmethod

class BaseNetwork(ABC):
    def __init__(self, model_name="default_model"):
        """
        Initialize common properties for network models.
        """
        self.model_name = model_name
        self.return_grad = True
        self.parameters = {}

    @abstractmethod
    def forward(self, x):
        """
        Compute the forward pass.
        Args:
            x (np.array): Input data.
        """
        pass

    @abstractmethod
    def backward(self, grad):
        """
        Compute the backward pass.
        Args:
            grad (np.array): Gradient of the loss function with respect to the output.
        """
        pass

    def __call__(self, x):
        """
        Make the network callable.
        """
        return self.forward(x)

    def train(self):
        """
        Set the network to training mode.
        """
        self.return_grad = True

    def eval(self):
        """
        Set the network to evaluation mode.
        """
        self.return_grad = False

    @abstractmethod
    def save_model(self, path):
        """
        Save the model parameters to a file.
        Args:
            path (str): The path to the file where to save the model.
        """
        pass

    @abstractmethod
    def load_model(self, path):
        """
        Load the model parameters from a file.
        Args:
            path (str): The path to the file from which to load the model.
        """
        pass
