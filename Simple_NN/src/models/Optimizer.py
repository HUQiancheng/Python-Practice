class SGDOptimizer:
    def __init__(self, learning_rate=0.01):
        """
        Initialize the SGD optimizer with a given learning rate.
        Args:
            learning_rate (float): The step size used for each iteration of the optimization.
        """
        self.learning_rate = learning_rate

    def update_params(self, model_parameters, gradients):
        """
        Update the model parameters using stochastic gradient descent.
        Args:
            model_parameters (dict): Dictionary containing the parameters of the model (weights and biases).
            gradients (dict): Dictionary containing the gradients of the loss function with respect to the parameters.
        """
        for key in model_parameters.keys():
            # Ensure that the gradient exists for the current parameter
            if key in gradients:
                model_parameters[key] -= self.learning_rate * gradients[key]
