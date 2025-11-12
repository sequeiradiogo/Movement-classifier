# Building Extraction from Satellite Images

This repository implements a complete pipeline for classifying inertial sensor data using a Support Vector Machine (SVM) model. The project extracts time-domain features from raw motion data, filters them for relevance, and performs classification using Leave-One-Out Cross-Validation (LOOCV) for robust evaluation.

The movements that are being classified consist of different types of soccer kicks: with the inside of the foot (parte de dentro), the outside of the foot (parte de fora), or toe poke (biqueira). There is also a rejection class (rejeicao).

## Repository Structure

```
movemet-classifier/
│
├── README.md                   # Project overview and instructions
│
├── data/
│   ├── raw        # raw txt data
│
├── src/                      # Source code
│   ├── extract_files.py      # Extracts TSFEL features from raw sensor data
│   ├── filter_features.py    # Filters and selects relevant features
│   ├── SVM_LOO.py            # Trains and evaluates the SVM model (LOOCV)
│   └── SVM_classify.py       # Classifies unseen data samples
│
└── outputs/                    
    ├── filtered_features.csv  # Example filtered feature file
    └── combined_features.csv  # Full extracted feature dataset
```

---

## Workflow Overview

### 1. Feature Extraction

`extract_files.py` parses the raw `.txt` sensor files, extracts **time-domain** features using the [TSFEL](https://tsfel.readthedocs.io/) library, and saves the combined output to `combined_features.csv`.

```python
from extract_files import extract
extract("path/to/data_raw")
```

### 2. Feature Filtering

`filter_features.py`filters the extracted dataset to retain only key features relevant for classification and saves them to `filtered_features.csv`.

```python
from filter_features import filter_features
filter_features("path/to/data_raw")
```

### 3. Model Training and Evaluation

`SVM_LOO.py` trains a linear SVM using Leave-One-Out cross-validation and reports model accuracy, classification report and confusion matrix.

![Alt text](images/confusion_matrix.png)
![Alt text](images/classification_report.png)

### 4. Prediction on New Data

`SVM_classify.py` applies the trained model to new unseen samples.
It uses the same extraction and filtering pipeline before generating predictions.

## Dependencies

Install required packages with:

```bash
pip install pandas scikit-learn tsfel matplotlib
```

## Data

Examples of the data used are found inside the data folder.



