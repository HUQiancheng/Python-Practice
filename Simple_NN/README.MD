# Standard Neural Network Project

## Overview

This project implements a fundamental neural network framework using basic Python libraries. It provides an intuitive and educational environment to learn machine learning workflows without relying on advanced machine learning frameworks like TensorFlow or PyTorch. The project is designed to be modular, expandable, and easy to understand, especially for those seeking foundational knowledge about neural network architectures.

## Prerequisites

Ensure you have the following Python libraries installed:
- **Numpy**: For numerical operations and array handling.
- **Pandas**: For data manipulation and ingestion.
- **Scikit-learn**: For basic machine learning tools and preprocessing.
- **Matplotlib**: For plotting and visualization of data and learning metrics.

To install these dependencies, use:
```bash
pip install numpy pandas scikit-learn matplotlib
```

## Project Structure

```plaintext
C:.
│   README.MD
│   TemplateNN.ipynb
│   
├───dataset
│   └───data
│           data.csv
│
├───src
│   test.py
│   trainer.py
│   validation.py
│   __init__.py
│   
├───src/data
│   BaseDataset.py
│   DataLoader.py
│   DataProcessor.py
│   Utils_Data.py
│   __init__.py
│   
├───src/models
│   BaseNetwork.py
│   LossFunction.py
│   Optimizer.py
│   Utils_Model.py
│   __init__.py
│
├───src/utils
│   Utils.py
│   __init__.py
│
└───trained_models
```

- **dataset/data**: Contains the primary dataset (`data.csv`).
- **src/test.py**: Script for testing the neural network.
- **src/trainer.py**: Main training loop for the neural network.
- **src/validation.py**: Handles the validation process during training.
- **src/data**:
  - **BaseDataset.py**: Polymorphic base class for datasets.
  - **DataLoader.py**: Loads and shuffles data.
  - **DataProcessor.py**: Prepares data for model training.
  - **Utils_Data.py**: Utilities for data handling.
- **src/models**:
  - **BaseNetwork.py**: Base class for neural network models.
  - **LossFunction.py**: Implements the Binary Cross-Entropy Loss.
  - **Optimizer.py**: Basic optimizer for network training.
  - **Utils_Model.py**: Model-related utilities.
- **src/utils**:
  - **Utils.py**: General utilities for the project.
- **trained_models**: Directory to store models after training.

## Key Features

1. **Custom Neural Network Framework**: Modular and expandable structure, starting with a base network class and core modules like loss functions and optimizers.
   
2. **Polymorphic Data Pipeline**:
   - **BaseDataset**: A flexible data structure that orchestrates data loading and preprocessing.
   - **DataLoader**: Efficiently loads data from CSV files and optionally shuffles it.
   - **DataProcessor**: Normalizes, reshapes, and splits the data into training, validation, and test sets.
   - **Utils_Data**: Includes visualization functions for data understanding.

3. **Model Training and Testing**:
   - **trainer.py**: Main script to train the neural network.
   - **test.py**: Evaluates the trained network's performance on the test dataset.

## Usage Instructions

### Training the Model

1. Navigate to the `src` directory:
   ```bash
   cd src
   ```

2. Run the `trainer.py` script to start training the model:
   ```bash
   python trainer.py
   ```

### Evaluating the Model

After training, evaluate the model's performance by running `test.py`:
```bash
python test.py
```

### Visualization and Analysis

The data utility functions can visualize data properties via histograms, scatter plots, and line plots using `matplotlib`. Example functions are available in `Utils_Data.py`.

## Further Development

- **New Architectures**: Extend `BaseNetwork` to build more complex neural network architectures.
- **Dataset Handling**: Implement new data loaders or processors for different datasets/formats.
- **Custom Optimizers/Loss Functions**: Add custom optimization and loss calculation strategies.

The project's modular structure facilitates the addition of new features with minimal disruption.