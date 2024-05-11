from sklearn.model_selection import train_test_split

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
