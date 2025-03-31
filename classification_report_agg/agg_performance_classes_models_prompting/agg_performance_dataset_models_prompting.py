# -*- coding: utf-8 -*-
import pandas as pd
import glob
from sklearn.metrics import precision_recall_fscore_support

# Load true labels
true_labels_file = '.../data/REF_Albuc_1.xlsx' #reference file
true_labels_df = pd.read_excel(true_labels_file)

# Assuming the true labels are in a column named 'POS'
true_labels = true_labels_df['POS']

# Load prediction files
pred_files = glob.glob("./pred/*.xlsx")

# Create a set of unique classes
unique_classes = true_labels.unique()

# Prepare a dictionary to store cumulative metrics for each class
precision_per_class = {cls: [] for cls in unique_classes}
recall_per_class = {cls: [] for cls in unique_classes}
f1_per_class = {cls: [] for cls in unique_classes}

# Iterate through each prediction file
for pred_file in pred_files:
    # Load prediction data
    pred_df = pd.read_excel(pred_file)
    
    # Assuming the predictions are in a column named 'POS'
    pred_labels = pred_df['upos']
    
    # Calculate precision, recall, and f1 for each class
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, labels=unique_classes, average=None, zero_division=0)
    
    # Store the values for each class
    for i, cls in enumerate(unique_classes):
        precision_per_class[cls].append(precision[i])
        recall_per_class[cls].append(recall[i])
        f1_per_class[cls].append(f1[i])

# Now calculate the average precision, recall, and F1 for each class across all files
avg_precision = {cls: round(sum(precision_per_class[cls])/len(precision_per_class[cls]), 2) for cls in unique_classes}
avg_recall = {cls: round(sum(recall_per_class[cls])/len(recall_per_class[cls]),2) for cls in unique_classes}
avg_f1 = {cls: round(sum(f1_per_class[cls])/len(f1_per_class[cls]),2) for cls in unique_classes}

# Combine into a DataFrame for better readability
summary_df = pd.DataFrame({
    'precision': avg_precision,
    'recall': avg_recall,
    'f1': avg_f1
})


print(summary_df)
