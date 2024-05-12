import os
from src.data.BaseDataset import BaseDataset
from src.models.ClassifierNetwork import Classifier
from src.models.LossFunction import BCE
from src.Solver import Solver
# file_path is correctly defined relative to the script running this code
file_path = os.path.join(os.path.dirname(__file__), 'dataset/data/data.csv')
dataset = BaseDataset(file_path, shuffle=True, normalize=True)
train_data, validate_data, test_data = dataset.prepare_data()

# Assuming you have a ClassifierNetwork class already imported and ready
num_features = train_data.shape[1] - 1  
classifier = Classifier(num_features=2)  

# Extract features and targets from prepared data
X_train = train_data.drop('target', axis=1).values
y_train = train_data['target'].values

X_val = validate_data.drop('target', axis=1).values
y_val = validate_data['target'].values


solver = Solver(model=classifier, data={'X_train': X_train, 'y_train': y_train,
                                        'X_val': X_val, 'y_val': y_val},
                loss_func=BCE(), learning_rate=0.01, batch_size=32)


solver.train(epochs=200)