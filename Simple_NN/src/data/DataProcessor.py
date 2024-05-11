import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.data.Utils_Data import split_data


class DataProcessor:
    def __init__(self, data_loader, normalize=True, train_size=0.7, validate_size=0.15, test_size=0.15):
        """
        Initialize the DataProcessor.
        Args:
            data_loader (DataLoader): An instance of DataLoader used to load the data.
            normalize (bool): Whether to normalize the features.
            train_size (float): Proportion of data to allocate to training set.
            validate_size (float): Proportion of data to allocate to validation set.
            test_size (float): Proportion of data to allocate to test set.
        """
        self.data_loader = data_loader
        self.normalize = normalize
        self.train_size = train_size
        self.validate_size = validate_size
        self.test_size = test_size

        self.train_data = None
        self.validate_data = None
        self.test_data = None
        self.scaler = StandardScaler()

    def process_data(self):
        """
        Process the loaded data: split and normalize it as required.
        """
        # Load the data
        data = self.data_loader.load_data()

        # Split the data into training, validation, and test sets
        self.train_data, self.validate_data, self.test_data = split_data(
            data, train_size=self.train_size, validate_size=self.validate_size, test_size=self.test_size
        )

        if self.normalize:
            self._normalize_data()

        return self.train_data, self.validate_data, self.test_data

    def _normalize_data(self):
        """
        Normalize features in training, validation, and test datasets.
        Only numeric features are normalized.
        """
        # Extract feature columns (excluding the target column)
        feature_columns = self.train_data.columns.difference(['target'])

        # Fit the scaler on the training data and transform all datasets
        self.train_data[feature_columns] = self.scaler.fit_transform(self.train_data[feature_columns])
        self.validate_data[feature_columns] = self.scaler.transform(self.validate_data[feature_columns])
        self.test_data[feature_columns] = self.scaler.transform(self.test_data[feature_columns])
