import numpy as np

class BinaryCrossEntropyLoss:
    def __init__(self):
        """
        Initialize any necessary properties if required.
        """
        pass

    def forward(self, y_pred, y_true):
        """
        Compute the Binary Cross-Entropy loss.
        Args:
            y_pred (np.array): Predicted probabilities (values should be between 0 and 1).
            y_true (np.array): True binary labels (0 or 1).

        Returns:
            float: The computed BCE loss.
        """
        # Avoid log(0) issues by clipping predictions to a small range near zero but not zero
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        # Compute the BCE loss
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss

    def backward(self, y_pred, y_true):
        """
        Compute the gradient of the Binary Cross-Entropy loss with respect to predictions.
        Args:
            y_pred (np.array): Predicted probabilities (values should be between 0 and 1).
            y_true (np.array): True binary labels (0 or 1).

        Returns:
            np.array: The gradient of the loss with respect to the predictions.
        """
        # Avoid division by zero issues
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        # Compute the gradient
        gradient = -(y_true / y_pred) + (1 - y_true) / (1 - y_pred)
        return gradient / len(y_true)  # Normalize by the number of samples
