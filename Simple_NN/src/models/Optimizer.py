class Optimizer:
    def __init__(self, model, learning_rate=5e-5):
        """
        Initialize the optimizer with a reference to the model and a specified learning rate.
        
        Args:
            model: The model whose parameters will be optimized.
            learning_rate (float): The step size used for updating the parameters.
        """
        self.model = model
        self.lr = learning_rate

    def step(self, dw):
        """
        Perform a single gradient descent update on the model's weights.
        
        Args:
            dw (np.array): The gradient of the loss function with respect to the model's weights.
        
        Returns:
            None: This method updates the model's weights directly.
        """
        # Retrieve the current weights from the model
        weight = self.model.W

        # Update the weights using gradient descent
        weight -= self.lr * dw

        # Save the updated weights back to the model
        self.model.W = weight
