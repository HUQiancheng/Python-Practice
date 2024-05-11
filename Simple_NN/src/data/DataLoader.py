import os
import pandas as pd
import numpy as np

# Define the DataLoader class
class DataLoader:
    # The constructor method for the class
    def __init__(self, file_path, shuffle=False):
        # Initialize the file_path attribute with the provided file path
        self.file_path = file_path
        # Initialize the shuffle attribute. If shuffle is True, the data will be shuffled when loaded.
        self.shuffle = shuffle
        # Initialize the data attribute to None. This will hold the loaded data.
        self.data = None

    # Define a method to load data from the file
    def load_data(self):
        # Check if the file exists
        if not os.path.exists(self.file_path):
            # If the file does not exist, raise a FileNotFoundError
            raise FileNotFoundError(f"The file {self.file_path} does not exist.")
        
        # Load the data from the CSV file into a pandas DataFrame
        self.data = pd.read_csv(self.file_path)

        # If shuffle is True, shuffle the data
        if self.shuffle:
            # The sample method with frac=1 returns a random sample of items. reset_index with drop=True resets the index to the default integer index.
            self.data = self.data.sample(frac=1).reset_index(drop=True)

        # Return the loaded (and possibly shuffled) data
        return self.data