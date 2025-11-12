# -*- coding: utf-8 -*-
"""
Support Vector Machine (SVM) Classification Script
--------------------------------------------------
Performs classification using a linear SVM model with Leave-One-Out cross-validation
on pre-filtered inertial sensor features.

Created: Dec 18, 2024
Author: Diogo Sequeira
"""

import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt


def run_svm_classification(file_path: str = 'filtered_features.csv') -> None:
    """
    Trains and evaluates an SVM classifier using Leave-One-Out cross-validation.

    Parameters
    ----------
    file_path : str, optional
        Path to the CSV file containing filtered features. Default is 'filtered_features.csv'.
    """
    # Load dataset
    df = pd.read_csv(file_path, sep=';')

    # Separate features (X) and labels (y)
    X = df.drop(columns=['File', 'Class'])
    y = df['Class']

    # Optionally normalize data
    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)

    # Initialize linear SVM
    svm_model = SVC(kernel='linear')

    # Leave-One-Out cross-validation setup
    loo = LeaveOneOut()
    y_true, y_pred = [], []

    # Perform Leave-One-Out training/testing
    for train_index, test_index in loo.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        svm_model.fit(X_train, y_train)
        prediction = svm_model.predict(X_test)[0]

        y_pred.append(prediction)
        y_true.append(y_test.iloc[0])

    # Final training on last split (not necessary, but kept for consistency)
    svm_model.fit(X_train, y_train)

    # Evaluation
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(conf_matrix)

    # Visualization
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=svm_model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()


if __name__ == "__main__":
    run_svm_classification()
