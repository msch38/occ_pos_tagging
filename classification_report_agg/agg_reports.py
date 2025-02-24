# -*- coding: utf-8 -*-

import pandas as pd
import glob

# Get all Excel files in the directory
file_paths = glob.glob("./class_reports/*.xlsx")  # Update with your actual path

# Initialize a dictionary to store aggregated data
agg_data = {}

# Track total correct and total samples for accuracy calculation
total_correct = 0
total_samples = 0

# Process each file
for file in file_paths:
    df = pd.read_excel(file, index_col=0)  # Read file, assuming first column has labels
    for label in df.index:
        if label not in agg_data:
            agg_data[label] = {"Precision": [], "Recall": [], "F1 Score": [], "Support": []}

        # Store values for later averaging
        agg_data[label]["Precision"].append(df.loc[label, "Precision"])
        agg_data[label]["Recall"].append(df.loc[label, "Recall"])
        agg_data[label]["F1 Score"].append(df.loc[label, "F1 Score"])
        agg_data[label]["Support"].append(df.loc[label, "Support"])

    # Extract overall accuracy from each report if available
    if "Accuracy" in df.index:
        total_correct += df.loc["Accuracy", "Support"] * df.loc["Accuracy", "Precision"]
        total_samples += df.loc["Accuracy", "Support"]

# Compute aggregated results
aggregated_results = []
all_precisions = []
all_recalls = []
all_f1s = []
all_supports = []

for label, metrics in agg_data.items():
    support = sum(metrics["Support"])  # Total support for weighted averaging
    if support == 0:
        continue  # Skip to avoid division by zero

    weighted_precision = round(sum(p * s for p, s in zip(metrics["Precision"], metrics["Support"])) / support, 2)
    weighted_recall = round(sum(r * s for r, s in zip(metrics["Recall"], metrics["Support"])) / support, 2)
    weighted_f1 = round(sum(f * s for f, s in zip(metrics["F1 Score"], metrics["Support"])) / support, 2)

    aggregated_results.append([label, weighted_precision, weighted_recall, weighted_f1, support])

    # Store values for macro and micro averaging
    all_precisions.append(weighted_precision)
    all_recalls.append(weighted_recall)
    all_f1s.append(weighted_f1)
    all_supports.append(support)

# Compute macro average (simple mean)
macro_precision = round(sum(all_precisions) / len(all_precisions), 2)
macro_recall = round(sum(all_recalls) / len(all_recalls), 2)
macro_f1 = round(sum(all_f1s) / len(all_f1s), 2)

# Compute weighted average (weighted by support)
total_support = sum(all_supports)
weighted_precision = round(sum(p * s for p, s in zip(all_precisions, all_supports)) / total_support, 2)
weighted_recall = round(sum(r * s for r, s in zip(all_recalls, all_supports)) / total_support, 2)
weighted_f1 = round(sum(f * s for f, s in zip(all_f1s, all_supports)) / total_support, 2)


# Compute micro average (sum TP, FP, FN)
micro_precision = weighted_precision  # Same as weighted in this case
micro_recall = weighted_recall  # Same as weighted
micro_f1 = weighted_f1  # Same as weighted

# Compute overall accuracy
accuracy = round(total_correct / total_samples, 2) if total_samples > 0 else 0


# Append macro, micro, and weighted averages to results
aggregated_results.extend([
    ["micro avg", micro_precision, micro_recall, micro_f1, total_support],
    ["macro avg", macro_precision, macro_recall, macro_f1, total_support],
    ["weighted avg", weighted_precision, weighted_recall, weighted_f1, total_support],
    ["accuracy", accuracy, accuracy, accuracy, total_samples]  # Accuracy is same across precision, recall, and F1
])

# Convert to DataFrame
aggregated_df = pd.DataFrame(aggregated_results, columns=["label", "precision", "recall", "f1-score", "support"])

# Save to an Excel file
aggregated_df.to_excel("aggregated_classification_report.xlsx", index=False)

print("Aggregated classification report saved.")
