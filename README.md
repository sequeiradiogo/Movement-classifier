🧭 Inertial Data Classification using SVM

This project processes raw inertial sensor data, extracts time-domain features, filters them, and classifies the resulting signals using a Support Vector Machine (SVM) model with Leave-One-Out Cross-Validation (LOOCV).

├── extractfilesfinal.py        # Extracts TSFEL features from raw sensor data
├── Short_features.py           # Filters and selects relevant features
├── SVM_Leave_one_out.py        # Trains & evaluates the SVM model
├── Predict_new_samples.py      # Classifies unseen data samples
├── combined_features.csv       # Full extracted feature dataset
├── filtered_features.csv       # Filtered dataset with selected features
└── data_raw/                   # Folder containing the original .txt sensor files

⚙️ Workflow Overview
1. Feature Extraction

extractfilesfinal.py parses the raw .txt sensor files, extracts time-domain features using the TSFEL library, and saves them into combined_features.csv.

from extractfilesfinal import extract
extract("path/to/data_raw")

2. Feature Filtering

Short_features.py selects and retains only the most relevant features for classification, saving them to filtered_features.csv.

from Short_features import filter_features
filter_features("path/to/data_raw")

3. Model Training & Evaluation

SVM_Leave_one_out.py trains a linear SVM using Leave-One-Out Cross-Validation (LOOCV) and reports:

Overall accuracy

Classification report

Confusion matrix

Run it with:

python SVM_Leave_one_out.py


Example Outputs:

Confusion Matrix

Classification Report

4. Prediction on New Data

Predict_new_samples.py applies the trained model to unseen data.
It uses the same extraction and filtering pipeline before generating predictions.

from Predict_new_samples import predict_new_samples
predict_new_samples("path/to/new_data")


Example Output:

Predicted class labels for each new sample

🧠 Dependencies

Install all required packages:

pip install pandas scikit-learn tsfel matplotlib

🗂️ Data Format

Place your raw inertial measurement .txt files inside the data_raw/ folder.

Each file should include time-stamped accelerometer and gyroscope readings in the format:

Acc: ax, ay, az
Gyro: gx, gy, gz
t: timestamp

🔁 Typical Workflow Summary

Place raw .txt files inside data_raw/

Run extractfilesfinal.py → extract features

Run Short_features.py → filter key features

Run SVM_Leave_one_out.py → train & evaluate model

Run Predict_new_samples.py → classify new data



Diogo Sequeira
Experimental Physics / Data Analysis
