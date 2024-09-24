import numpy
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

def prepare_and_evaluate(selected_features, df, classifier_model, metric_function, ignore_first=False, test_size=0.30):
    """
    This function handles the middle steps of feature selection, dataset splitting,
    training, and evaluation.
    
    Parameters:
    - selected_features: Binary array of selected features (1 for selected, 0 for not selected).
    - df: The dataset (Pandas DataFrame).
    - classifier_model: The classifier model to use for training (e.g., SVC, LogisticRegression).
    - metric_function: The performance metric to evaluate the classifier (e.g., accuracy_score, f1_score).
    - ignore_first: Whether to ignore the first column (e.g., ID) of the dataset.
    - test_size: Proportion of the dataset to use for testing (default is 30%).
    
    Returns:
    - performance: The performance score based on the chosen metric function.
    - S: The number of selected features (subset size).
    """
    
    # Threshold for feature selection
    threshold = 0.5
    
    # Convert the selected features to binary (0 or 1)
    selected_features = numpy.asarray(selected_features).reshape(1, -1)
    selected_features[selected_features > threshold] = 1
    selected_features[selected_features <= threshold] = 0

    # Identify indices of the selected features
    selected_indices = numpy.where(selected_features == 1)[1]

    # If no features are selected, default to selecting the first feature
    if not selected_indices.size:
        selected_indices = [0]  # Default to the first feature

    # Number of selected features
    S = len(selected_indices)

    # Copy the dataset and remove the first column if ignore_first is True
    df_copy = pd.DataFrame(df, copy=True)
    if ignore_first:
        features_and_labels = df_copy.iloc[:, 1:]  # Exclude the first column
    else:
        features_and_labels = df_copy

    # Extract the selected features (X) and target (y)
    X = features_and_labels.iloc[:, selected_indices].values  # Features
    y = features_and_labels.iloc[:, -1].astype('category').cat.codes.values  # Target (convert to numerical codes)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # Train the classifier on the selected features
    classifier_model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = classifier_model.predict(X_test)

    # Evaluate performance using the provided metric function
    performance = metric_function(y_test, y_pred)

    return performance, S

def F1(selected_features, df, classifier_model, metric_function, ignore_first, test_size=0.30):
    """
    Returns the performance based on the selected features.
    """
    performance, _ = prepare_and_evaluate(selected_features, df, classifier_model, metric_function, ignore_first, test_size)
    return performance


def F2(selected_features, df, classifier_model, metric_function, ignore_first, test_size=0.30):
    """
    Returns a fitness score based on both classification accuracy and the size of the feature subset.
    """
    
    # Constants α and β, where α + β = 1
    # α,β are constants which control the relative importance of performance and the length of feature subset
    alpha = 0.8 # Importance of performance
    beta = 0.2  # Importance of length
    
    # Calculate T (total number of features)
    T = df.shape[1] - 1  # Total number of attributes excluding the target column

    # Call the helper function to get performance and the number of selected features (S)
    performance, S = prepare_and_evaluate(selected_features, df, classifier_model, metric_function, ignore_first, test_size)

    # Calculate fitness using the formula: α * performance + β * (T - S) / S
    fitness = alpha * performance + beta * (T - S) / S
    
    return fitness