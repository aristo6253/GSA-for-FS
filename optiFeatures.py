import numpy
import math
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix


def optiFeats(selected_features, df, clf):
    """ Identify the optimal feature set and evaluate classification performance """
    
    # Threshold for selecting features (e.g., > 0.5 means feature is selected)
    threshold = 0.5
    
    # Convert feature selection array to binary based on the threshold
    selected_features = numpy.asarray(selected_features).reshape(1, -1)
    selected_features[selected_features > threshold] = 1
    selected_features[selected_features <= threshold] = 0

    # Identify indices of selected features
    feature_indices = numpy.where(selected_features == 1)[1]
    
    # Default to selecting the first feature if no features were selected
    if feature_indices.size == 0:
        feature_indices = numpy.array([0])

    print(f"The optimal features are: {selected_features}")
    # print(f"The indices of optimal features are: {feature_indices}")
    for i, feature_index in enumerate(feature_indices):
            feature_name = df.columns[feature_index + 1]
            print(f"{i+1}. Feature Index: {feature_index + 1}, Feature Name: {feature_name}")
    
    # Extract the selected features and target variable from the dataset
    X = df.iloc[:, feature_indices].values  # Features
    y = df.iloc[:, -1].astype('category').cat.codes.values  # Target (converted to numerical codes)
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    
    # Train and evaluate a classifier (SVM in this case)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    # Calculate performance metrics (accuracy and confusion matrix)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred).flatten()
    
    # Combine results into a single array
    return numpy.concatenate(([accuracy, f1], confusion))
