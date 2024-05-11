import numpy as np

class BaseNetwork:
    def __init__(self):
        """
        Initialize common properties for network models.
        """
        self.parameters = {}  # Initialize an empty dict for parameters
        self.activations = {}  # To store activations if necessary for backward pass

    def forward(self, x):
        """
        Compute the forward pass. Must be implemented by subclass.
        Args:
            x (np.array): Input data.
        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Forward method must be implemented by subclass.")

    def backward(self, grad):
        """
        Compute the backward pass. Must be implemented by subclass.
        Args:
            grad (np.array): Gradient of the loss function with respect to the output.
        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Backward method must be implemented by subclass.")

    def save_model(self, path):
        """
        Save the model parameters to a file using numpy.
        Args:
            path (str): The path to the file where to save the model.
        """
        np.savez(path, **self.parameters)

    def load_model(self, path):
        """
        Load the model parameters from a file using numpy.
        Args:
            path (str): The path to the file from which to load the model.
        """
        data = np.load(path, allow_pickle=True)  # Ensure that loading handles object arrays
        self.parameters = {key: data[key] for key in data.files}

    def apply_activation(self, x):
        """
        Apply activation function, to be overridden by subclasses if they require specific activations.
        Args:
            x (np.array): Linear combination from a layer's output before activation.
        Returns:
            np.array: Activated output.
        """
        raise NotImplementedError("Activation function must be implemented by subclass.")
