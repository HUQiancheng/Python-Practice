import numpy as np

class Loss:
    def __init__(self):
        self.grad_history = []

    def forward(self, y_out, y_truth, individual_losses=False):
        raise NotImplementedError("Forward method must be implemented in subclasses.")

    def backward(self, y_out, y_truth, upstream_grad=1.):
        raise NotImplementedError("Backward method must be implemented in subclasses.")

    def __call__(self, y_out, y_truth, individual_losses=False):
        loss = self.forward(y_out, y_truth, individual_losses)
        return loss

class BCE(Loss):
    def forward(self, y_out, y_truth, individual_losses=False):
        """
        Perform the forward pass of the Binary Cross-Entropy loss function.
        Args:
            y_out (np.array): [N,] Array of predicted probabilities.
            y_truth (np.array): [N,] Array of ground truth binary labels.
            individual_losses (bool): Return each instance loss if True, or the mean if False.
        Returns:
            np.array: The computed BCE loss for each sample or mean loss across all samples.
        """
        # Reshape y_truth to match y_out if necessary
        if y_truth.ndim == 1:
            y_truth = y_truth.reshape(-1, 1)

        # Check shapes
        if y_out.shape != y_truth.shape:
            raise ValueError("Shape mismatch between predicted and ground truth values.")

        # Avoid log(0) errors by clipping predicted probabilities
        epsilon = 1e-15
        y_out = np.clip(y_out, epsilon, 1 - epsilon)

        # Compute the Binary Cross-Entropy loss
        loss_values = -y_truth * np.log(y_out) - (1 - y_truth) * np.log(1 - y_out)

        if individual_losses:
            return loss_values

        return np.mean(loss_values)

    def backward(self, y_out, y_truth):
        """
        Perform the backward pass of the Binary Cross-Entropy loss function.
        Args:
            y_out (np.array): [N,] Array of predicted probabilities.
            y_truth (np.array): [N,] Array of ground truth binary labels.
        Returns:
            np.array: Gradients of the BCE loss with respect to the predictions.
        """
        # Reshape y_truth to match y_out if necessary
        if y_truth.ndim == 1:
            y_truth = y_truth.reshape(-1, 1)

        # Ensure compatibility and clip probabilities
        epsilon = 1e-15
        y_out = np.clip(y_out, epsilon, 1 - epsilon)

        # Compute gradients
        gradients = -(y_truth / y_out) + ((1 - y_truth) / (1 - y_out))
        return gradients / y_out.shape[0]
