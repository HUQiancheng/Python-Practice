from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def split_data(data, train_size=0.7, validate_size=0.15, test_size=0.15):
    """
    Splits the data into train, validation, and test sets.
    Args:
        data (DataFrame): The dataset to be split.
        train_size (float): Proportion of the dataset to include in the train split.
        validate_size (float): Proportion of the dataset to include in the validation split.
        test_size (float): Proportion of the dataset to include in the test split.

    Returns:
        train_data (DataFrame), validate_data (DataFrame), test_data (DataFrame)
    """
    if not (train_size + validate_size + test_size) == 1.0:
        raise ValueError("The sum of train, validate, and test sizes must equal 1.")
    
    # First split to separate out the training set
    train_data, temp_data = train_test_split(data, train_size=train_size, shuffle=True)
    
    # Adjust the proportion sizes for validation and test sets
    proportion = validate_size / (validate_size + test_size)
    
    # Second split to separate out the validation and test sets
    validate_data, test_data = train_test_split(temp_data, train_size=proportion, shuffle=True)

    return train_data, validate_data, test_data


def plot_all_histograms(data, exclude_columns=['target'], bins=20):
    """
    Plot histograms for all numeric columns in a single figure.
    Args:
        data (DataFrame): The dataset containing numeric features.
        exclude_columns (list): List of columns to exclude from the plot.
        bins (int): Number of bins for the histograms.
    """
    # Get numeric columns excluding those specified
    numeric_columns = data.select_dtypes(include='number').columns.difference(exclude_columns)

    # Determine grid size based on the number of numeric columns
    num_columns = len(numeric_columns)
    num_rows = (num_columns + 2) // 3  # Adjust this number for different grid arrangements

    fig, axes = plt.subplots(num_rows, 3, figsize=(15, num_rows * 5))
    axes = axes.flatten()  # Flatten in case of single row

    for i, col in enumerate(numeric_columns):
        axes[i].hist(data[col], bins=bins, color='skyblue', edgecolor='black')
        axes[i].set_title(f"{col} Distribution")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Count")
        axes[i].grid(True)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()

def plot_2d_data(data, feature_columns=['x1', 'x2'], label_column='target'):
    """
    Plot a 2D scatter plot for the given data if it has exactly two feature columns.
    Args:
        data (DataFrame): The dataset containing at least two features.
        feature_columns (list): List containing the two feature columns to plot.
        label_column (str): Name of the column containing the target labels.
    """
    # Ensure there are exactly two features specified for plotting
    if len(feature_columns) != 2:
        print("Plotting unavailable: Requires exactly 2 feature columns.")
        return

    # Ensure the feature columns are present in the data
    if not all(col in data.columns for col in feature_columns):
        print("Plotting unavailable: One or both feature columns not found in the data.")
        return

    # Extract features and labels
    x1 = data[feature_columns[0]]
    x2 = data[feature_columns[1]]
    labels = data[label_column]

    # Plot using different colors for each label
    plt.figure(figsize=(10, 6))
    for label in sorted(labels.unique()):
        plt.scatter(x1[labels == label], x2[labels == label], label=f'Category {label}', alpha=0.6)

    plt.xlabel(feature_columns[0])
    plt.ylabel(feature_columns[1])
    plt.title("2D Data Point Plot")
    plt.legend()
    plt.grid(True)
    plt.show()


