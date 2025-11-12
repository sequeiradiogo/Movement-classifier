# -*- coding: utf-8 -*-
"""
SVM Prediction on New Data Samples
----------------------------------
Loads new inertial data, extracts and filters features, scales them,
and performs predictions using the trained SVM model.

Created: Dec 19, 2024
Author: Diogo Sequeira
"""

import joblib
from sklearn.preprocessing import StandardScaler
from extract_files import extract
from filter_features import filter_features


def predict_new_samples(samples_path: str, model_path: str) -> None:
    """
    Predicts the classes of new data samples using the trained SVM model.

    Parameters
    ----------
    samples_path : str
        Directory path containing the new `.txt` files to analyze.
    model_path : str
        Path to the saved SVM model (.pkl file).
    """
    # Extract and filter features from new data
    new_df = filter_features(samples_path)

    # Separate features
    X_new = new_df.drop(columns=['File', 'Class'], errors='ignore')

    # Scale data (optional: ideally use the same scaler fitted on training data)
    #scaler = StandardScaler()
    #X_new_scaled = scaler.fit_transform(X_new)

    # Load trained SVM model
    model = joblib.load(model_path)

    # Predict using trained model
    predictions = model.predict(X_new)

    # Display results
    print("\nPredictions for new samples:")
    for i, pred in enumerate(predictions, start=1):
        file_name = new_df["File"].iloc[i - 1] if "File" in new_df.columns else f"Sample {i}"
        print(f"{file_name}: Predicted class = {pred}")


if __name__ == "__main__":
    # Example usage (replace with your paths)
    samples_path = r"C:\Users\35193\OneDrive\Ambiente de Trabalho\data"
    model_path = r"C:\Users\35193\OneDrive\Ambiente de Trabalho\svm_model.pkl"
    predict_new_samples(samples_path, model_path)
