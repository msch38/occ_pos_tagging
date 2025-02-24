# -*- coding: utf-8 -*-

import pandas as pd
import glob
import numpy as np

# Get all Excel files containing confusion matrices
file_paths = glob.glob("./confusion_matrix/*.xlsx")  # Update with your actual path

# Initialize counters for total correct predictions and total samples
total_correct = 0
total_samples = 0

# Dictionary to store per-class TP and total instances
class_stats = {}

# Process each confusion matrix file
for file in file_paths:
    df = pd.read_excel(file, index_col=0)  # Read Excel file

    # Ensure it's a square matrix (same number of rows & cols)
    if df.shape[0] != df.shape[1]:
        print(f"Skipping {file}: Not a valid confusion matrix.")
        continue

    # Convert to numpy array for fast calculations
    matrix = df.values  

    # Update overall accuracy calculations
    total_correct += np.trace(matrix)  # Sum of diagonal elements
    total_samples += np.sum(matrix)    # Sum of all elements in the matrix

    # Process per-class accuracy
    for i, label in enumerate(df.index):  # Assuming labels are row names
        TP = matrix[i, i]  # True Positives for class i
        total_class_samples = np.sum(matrix[i, :])  # Total instances of this class

        if label not in class_stats:
            class_stats[label] = {"TP": 0, "total": 0}

        class_stats[label]["TP"] += TP
        class_stats[label]["total"] += total_class_samples

# Compute final per-class accuracy
class_accuracies = []
for label, stats in class_stats.items():
    class_accuracy = round(stats["TP"] / stats["total"], 4) if stats["total"] > 0 else 0
    class_accuracies.append([label, class_accuracy])

# Compute overall accuracy
overall_accuracy = round(total_correct / total_samples, 4) if total_samples > 0 else 0

# Convert results to DataFrame
accuracy_df = pd.DataFrame(class_accuracies, columns=["Class", "Accuracy"])
accuracy_df.loc[len(accuracy_df)] = ["Overall Accuracy", overall_accuracy]

# Save to an Excel file
accuracy_df.to_excel("aggregated_accuracy_report.xlsx", index=False)

print("Aggregated per-class and overall accuracy saved.")
print(accuracy_df)