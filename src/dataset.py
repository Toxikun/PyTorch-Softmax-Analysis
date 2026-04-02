import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def get_iris_data():
    """
    Downloads/imports the Iris dataset and random splits it into 70% train, 15% validation, 15% test.
    Returns PyTorch tensors.
    """
    iris = load_iris()
    X = iris.data
    y = iris.target

    #First split: 70% train, 30% temp (which will be split to 15% val, 15% test)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
    
    #Second split: 50% of 30% = 15% val, 15% test
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

    #Standardizing features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    #Convert to PyTorch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    return X_train_t, y_train_t, X_val_t, y_val_t, X_test_t, y_test_t

def expand_features(X, degree=1):
    """
    Expands the feature matrix X to include polynomial terms.
    degree=1: linear (x_i)
    degree=2: linear + quadratic (x_i^2, x_i x_j)
    degree=3: linear + quadratic + cubic (x_i^3, x_i^2 x_j, x_i x_j x_k)
    
    Hint: Can use itertools.combinations_with_replacement or manually compute these.
    Here is a skeleton.
    """
    num_samples, num_features = X.shape
    features = [X]
    
    from itertools import combinations_with_replacement
    
    if degree >= 2:
        quad_terms = []
        for i, j in combinations_with_replacement(range(num_features), 2):
            quad_terms.append((X[:, i] * X[:, j]).unsqueeze(1))
        features.append(torch.cat(quad_terms, dim=1))
        
    if degree >= 3:
        cub_terms = []
        for i, j, k in combinations_with_replacement(range(num_features), 3):
            cub_terms.append((X[:, i] * X[:, j] * X[:, k]).unsqueeze(1))
        features.append(torch.cat(cub_terms, dim=1))

    return torch.cat(features, dim=1) if len(features) > 1 else features[0]
