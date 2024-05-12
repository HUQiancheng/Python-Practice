import numpy as np
from src.models.Optimizer import Optimizer

class Solver:
    def __init__(self, model, data, loss_func, learning_rate, batch_size=32, print_every=5):
        """
        Initialize the Solver instance with mini-batch support.
        """
        self.model = model
        self.loss_func = loss_func
        self.opt = Optimizer(model, learning_rate)
        self.batch_size = batch_size
        self.print_every = print_every

        # Data
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']

        # Reset book-keeping variables
        self._reset()

    def _reset(self):
        """
        Reset bookkeeping variables.
        """
        self.best_val_loss = float('inf')
        self.best_W = None
        self.train_loss_history = []
        self.val_loss_history = []

    def _get_batches(self):
        """
        Generate batches from training data.
        """
        num_samples = len(self.X_train)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)  # Shuffle data indices

        for start_idx in range(0, num_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            yield self.X_train[batch_indices], self.y_train[batch_indices]

    def _step(self):
        for X_batch, y_batch in self._get_batches():
            # Forward pass
            predictions = self.model.forward(X_batch)
            loss = self.loss_func.forward(predictions, y_batch)
            # Backward pass
            gradient = self.loss_func.backward(predictions, y_batch)
            # Normalize or sum the gradients here if not handled in `backward`
            gradient = np.sum(gradient, axis=0, keepdims=True)  # Make sure it matches weight shape
            # Update parameters
            self.opt.step(gradient)


    def check_loss(self, validation=True):
        """
        Check the model's loss on either training or validation data.
        """
        X = self.X_val if validation else self.X_train
        y = self.y_val if validation else self.y_train

        # Forward pass to compute loss
        predictions = self.model.forward(X)
        loss = self.loss_func.forward(predictions, y)

        return np.mean(loss)

    def train(self, epochs=1000):
        """
        Run optimization to train the model using mini-batch updates.
        """
        for epoch in range(epochs):
            self._step()
            # Monitoring progress
            train_loss = self.check_loss(validation=False)
            val_loss = self.check_loss(validation=True)
            self.train_loss_history.append(train_loss)
            self.val_loss_history.append(val_loss)

            # Check if current validation loss is better, update best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_W = np.copy(self.model.W)

            if epoch % self.print_every == 0 or epoch == epochs - 1:
                print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

        # At the end of training, set the best weights back to the model
        self.model.W = self.best_W

    def update_best_loss(self, val_loss):
        """
        Update the best known validation loss and model weights.
        """
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_W = np.copy(self.model.W)
