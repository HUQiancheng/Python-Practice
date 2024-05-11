import numpy as np
import pandas as pd

class DataPreparation:
    def __init__(self, csv_path):
        """
        Initialize the data preparation class with the path to the CSV file.
        """
        self.csv_path = csv_path
        self.features = None
        self.labels = None

    def load_data(self):
        """
        Load data from a CSV file.
        """
        data = pd.read_csv(self.csv_path)
        self.features = data.iloc[:, :-1].values  # Exclude the label column
        self.labels = data.iloc[:, -1].values     # Assume the last column is the label

    def normalize_data(self):
        """
        Normalize the feature data using min-max scaling.
        """
        self.features = (self.features - np.min(self.features, axis=0)) / (np.max(self.features, axis=0) - np.min(self.features, axis=0))

    def split_data(self, train_size=0.7, validation_size=0.15, random_seed=42):
        """
        Split the data into training, validation, and test sets manually.
        """
        # Set the random seed for reproducibility
        np.random.seed(random_seed)

        # Determine the number of samples
        n_samples = self.features.shape[0]
        
        # Create indices and shuffle them
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        # Calculate the sizes of each subset
        n_train = int(train_size * n_samples)
        n_val = int(validation_size * n_samples)
        
        # Split into train, validation, and test sets based on shuffled indices
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]

        # Retrieve data subsets
        features_train = self.features[train_indices]
        labels_train = self.labels[train_indices]
        features_val = self.features[val_indices]
        labels_val = self.labels[val_indices]
        features_test = self.features[test_indices]
        labels_test = self.labels[test_indices]

        return features_train, labels_train, features_val, labels_val, features_test, labels_test
