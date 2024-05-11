from src.data.DataLoader import DataLoader
from src.data.DataProcessor import DataProcessor

class BaseDataset:
    def __init__(self, file_path, shuffle=False, normalize=True):
        """
        Initialize the BaseDataset.
        Args:
            file_path (str): Path to the dataset file.
            shuffle (bool): Whether to shuffle the dataset upon loading.
            normalize (bool): Whether to normalize the dataset after loading.
        """
        self.file_path = file_path
        self.shuffle = shuffle
        self.normalize = normalize

        # Initialize DataLoader and DataProcessor
        self.data_loader = DataLoader(self.file_path, self.shuffle)
        self.data_processor = DataProcessor(self.data_loader, self.normalize)

        # Properties to store processed data
        self.train_data = None
        self.validate_data = None
        self.test_data = None

    def prepare_data(self):
        """
        Load and process the data through the DataLoader and DataProcessor, then store it.
        """
        # Process data and split into train, validation, and test sets
        self.train_data, self.validate_data, self.test_data = self.data_processor.process_data()

        return self.train_data, self.validate_data, self.test_data
