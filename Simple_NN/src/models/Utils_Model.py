import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_decision_boundary(data, feature_columns=['x1', 'x2'], label_column='target', classifier=None):
    """
    Plot data points and the decision boundary of a classifier if provided.
    Args:
        data (DataFrame): The dataset containing features and a target label.
        feature_columns (list): List containing the two feature columns to plot.
        label_column (str): Name of the column containing the target labels.
        classifier: The trained model with a 'forward' method, optional.
    """
    if classifier and len(feature_columns) == 2:
        # Define the axis boundaries
        x_min, x_max = data[feature_columns[0]].min() - 1, data[feature_columns[0]].max() + 1
        y_min, y_max = data[feature_columns[1]].min() - 1, data[feature_columns[1]].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))

        # Prepare grid as input for classifier
        grid = np.c_[xx.ravel(), yy.ravel()]

        # Predict probabilities on the grid
        probs = classifier.forward(grid).reshape(xx.shape)

        # Plot decision boundary
        plt.contourf(xx, yy, probs, levels=[0, 0.5, 1], alpha=0.5, linestyles=['--'], cmap='RdBu')
        plt.contour(xx, yy, probs, levels=[0.5], colors='red')

    # Plot the data points
    plt.scatter(data.loc[data[label_column] == 0, feature_columns[0]], data.loc[data[label_column] == 0, feature_columns[1]], 
                color='blue', label='Class 0', alpha=0.5)
    plt.scatter(data.loc[data[label_column] == 1, feature_columns[0]], data.loc[data[label_column] == 1, feature_columns[1]], 
                color='red', label='Class 1', alpha=0.5)

    plt.title('2D Data Points and Decision Boundary')
    plt.xlabel(feature_columns[0])
    plt.ylabel(feature_columns[1])
    plt.legend()
    plt.grid(True)
    plt.show()
