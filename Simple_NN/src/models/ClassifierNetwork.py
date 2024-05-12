import os
import pickle
import numpy as np
from src.models.BaseNetwork import BaseNetwork

class Classifier(BaseNetwork):
    """
    Classifier of the form y = sigmoid(X * W)
    """

    def __init__(self, num_features=2, model_name="classifier"):
        super().__init__(model_name)
        self.num_features = num_features
        self.W = None
        self.cache = None
        self.initialize_weights()

    def initialize_weights(self, weights=None):
        """
        Initialize the weight matrix W
        Args:
            weights (np.array): Optional weights for initialization
        """
        if weights is not None:
            assert weights.shape == (self.num_features + 1, 1), \
                "weights for initialization are not in the correct shape (num_features + 1, 1)"
            self.W = weights
        else:
            self.W = 0.001 * np.random.randn(self.num_features + 1, 1)

    def forward(self, X):
        """
        Performs the forward pass of the model.
        Args:
            X (np.array): N x D array of training data. Each row is a D-dimensional point.
            Note that it is changed to N x (D + 1) to include the bias term.
        Returns:
            np.array: Predicted logits for the data in X, shape N x 1
        """
        assert self.W is not None, "weight matrix W is not initialized"

        # Add a column of 1s to the data for the bias term
        batch_size, _ = X.shape
        X = np.concatenate((X, np.ones((batch_size, 1))), axis=1)

        # Linear affine transformation
        s = np.dot(X, self.W)

        # Apply sigmoid activation function
        z = self.sigmoid(s)

        # Cache relevant variables for backward pass
        self.cache = (X, s, z)

        return z

    def backward(self, dout):
        """
        Performs the backward pass of the model.
        Args:
            dout (np.array): N x M array. Upstream derivative with the same shape as forward output.
        Returns:
            np.array: Gradient of the weight matrix w.r.t the upstream gradient 'dout'.
        """
        assert self.cache is not None, "Run a forward pass before the backward pass."

        # Retrieve cached variables
        X_with_bias, s, z = self.cache

        # Calculate the gradient of the sigmoid output w.r.t. the linear layer output 's'
        ds = z * (1 - z) * dout

        # Gradient of the weights, now summing over the batch dimension
        dW = np.dot(X_with_bias.T, ds) / X_with_bias.shape[0]  # Normalizing by batch size

        return dW

    def sigmoid(self, x):
        """
        Computes the output of the sigmoid function.
        Args:
            x (np.array): Input of the sigmoid, np.array of any shape
        Returns:
            np.array: Output of the sigmoid with the same shape as input vector x
        """
        return 1 / (1 + np.exp(-x))

    def save_model(self, path='models'):
        """
        Save the model using pickle.
        Args:
            path (str): Directory where the model should be saved.
        """
        if not os.path.exists(path):
            os.makedirs(path)
        model_file = os.path.join(path, self.model_name + '.p')
        with open(model_file, 'wb') as file:
            pickle.dump(self, file)

    def load_model(self, path):
        """
        Load the model using pickle.
        Args:
            path (str): Path to the file from which to load the model.
        """
        with open(path, 'rb') as file:
            model = pickle.load(file)
        self.__dict__.update(model.__dict__)
