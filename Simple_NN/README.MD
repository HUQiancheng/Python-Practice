# Standard Neural Network Project

This project is designed to provide a fundamental understanding of neural network operations using very basic Python libraries. It is tailored for educational purposes, particularly for those interested in understanding machine learning workflows without the use of advanced libraries like PyTorch or TensorFlow.

## Prerequisites

The project is built using only the following libraries:
- `numpy`: For numerical operations and array handling.
- `pandas`: For data manipulation and ingestion.
- `sklearn`: For basic machine learning tools and pre-processing.
- `matplotlib`: For plotting and visualization of data and learning metrics.

Ensure that you have the above libraries installed. You can install them using pip:
```bash
pip install numpy pandas scikit-learn matplotlib
```

## Project Structure

- **dataset/data**
  - **data.csv**: The main dataset used for training and testing the neural network.
- **src**
  - **test.py**: Script for testing the neural network.
  - **trainer.py**: Contains the main training loop for the neural network.
  - **validation.py**: Handles the validation process during training.
  - **data**
    - **BaseDataset.py**: Polymorphic base class for datasets.
    - **DataLoader.py**: Facilitates data loading.
    - **DataProcessor.py**: Processes data for neural network input.
    - **Utils_Data.py**: Utilities for data handling.
  - **models**
    - **BaseNetwork.py**: Polymorphic base class for neural network models.
    - **LossFunction.py**: Implements the Binary Cross-Entropy Loss.
    - **Optimizer.py**: Basic optimizer for neural network training.
    - **Utils_Model.py**: Model-related utilities.
  - **utils**
    - **Utils.py**: General utilities for the project.
- **trained_models**
  - *Directory to store models after training.*

## Usage

### Training the Model

Navigate to the `src` directory and run the `trainer.py` script to start training the model:
```bash
python trainer.py
```

### Evaluating the Model

After training, evaluate the model's performance by running the `test.py` script:
```bash
python test.py
```

### Visualization

Plot the training and validation losses using `matplotlib` within the training or testing scripts, which helps in understanding the model's learning curve.

## Features

- **Linear Classification**: Implements a simple linear classifier.
- **Binary Cross-Entropy Loss**: Utilizes BCE loss for binary classification tasks.
- **CSV Dataset Handling**: The system is designed to work with datasets in CSV format by default.

## Notes

This project is setup in a way that allows for expansion and incorporation of more complex functionalities. The use of polymorphism in `BaseDataset` and `BaseNetwork` ensures that the project can be extended to other types of datasets or neural network architectures with minimal changes.