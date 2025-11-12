# -*- coding: utf-8 -*-
"""
Data Extraction and Feature Computation Script
----------------------------------------------
Parses inertial sensor data from text files, extracts features using the
TSFEL library, and saves all computed features into a single CSV file.

Created: Nov 30, 2024
Author: Diogo Sequeira
"""

import os
import re
import pandas as pd
import tsfel


def extract(path: str) -> pd.DataFrame:
    """
    Extracts inertial sensor data from text files, computes TSFEL features,
    and combines results into a single CSV file.

    Parameters
    ----------
    path : str
        Directory path containing the `.txt` sensor files.

    Returns
    -------
    pd.DataFrame
        DataFrame containing extracted features for all files.
    """
    # Find all .txt files in the directory
    file_list = [os.path.join(path, file) for file in os.listdir(path) if file.endswith(".txt")]

    all_features = []
    fs = 10  # Sampling frequency in Hz (100 ms intervals)

    for file in file_list:
        time, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z = [], [], [], [], [], [], []

        # Parse sensor data from each file
        with open(file, "r") as f:
            for line in f:
                acc_match = re.search(r"Acc:\s*(-?\d+\.\d+),\s*(-?\d+\.\d+),\s*(-?\d+\.\d+)", line)
                if acc_match:
                    accel_x.append(float(acc_match.group(1)))
                    accel_y.append(float(acc_match.group(2)))
                    accel_z.append(float(acc_match.group(3)))

                gyro_match = re.search(r"Gyro:\s*(-?\d+\.\d+),\s*(-?\d+\.\d+),\s*(-?\d+\.\d+)", line)
                if gyro_match:
                    gyro_x.append(float(gyro_match.group(1)))
                    gyro_y.append(float(gyro_match.group(2)))
                    gyro_z.append(float(gyro_match.group(3)))

                time_match = re.search(r"t:\s*(\d+)", line)
                if time_match:
                    time.append(int(time_match.group(1)))

        # Adjust time sequence
        for i in range(len(time)):
            time[i] = time[i] + 50 * i

        # Build DataFrame
        data = {
            "Time (ms)": time,
            "Accel_X": accel_x,
            "Accel_Y": accel_y,
            "Accel_Z": accel_z,
            "Gyro_X": gyro_x,
            "Gyro_Y": gyro_y,
            "Gyro_Z": gyro_z,
        }
        df = pd.DataFrame(data).drop_duplicates()

        # Exclude time column for feature extraction
        df = df.iloc[:, 1:]

        # Configure TSFEL (exclude unwanted domains)
        cfg = tsfel.get_features_by_domain()
        for domain in ['spectral']:
            if domain in cfg:
                del cfg[domain]

        # Extract features
        features = tsfel.time_series_features_extractor(cfg, df, fs=fs)

        # Add file identifier
        features["File"] = os.path.basename(file)
        all_features.append(features)

    # Combine all results
    combined_features = pd.concat(all_features, axis=0)
    combined_features["Class"] = combined_features["File"].apply(assign_class)

    # Save to CSV
    combined_features.to_csv("combined_features.csv", index=False, sep=";")
    print("Feature extraction complete. File saved as 'combined_features.csv'.")

    return combined_features


def assign_class(file_name: str) -> str:
    """
    Extracts the class label based on the filename prefix.

    Parameters
    ----------
    file_name : str
        The name of the file.

    Returns
    -------
    str
        Class name extracted from the filename.
    """
    return file_name.split('_')[0]


if __name__ == "__main__":
    # Example usage (replace with your own path)
    directory = r"C:\Users\35193\OneDrive\Ambiente de Trabalho\data"
    extract(directory)
