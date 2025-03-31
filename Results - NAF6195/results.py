# -*- coding: utf-8 -*-

from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, balanced_accuracy_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import os

def calculate_and_display_metrics(y_true, y_pred, unique_pos_values_list):
    """
    Calculate and display classification metrics with enhanced confusion matrix visualization
    and multiple strategies for handling unknown tags.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    unique_pos_values_list : list
        List of unique POS tags expected in the data
    handle_unknown :
        Strategy for handling unknown tags:
        - 'ignore': exclude tokens with predicted tags not in reference
    """
    # Convert inputs to lists for easier processing
    y_true = list(y_true)
    y_pred = list(y_pred)
    
    # Find unknown tags
    reference_tags = set(unique_pos_values_list)
    pred_tags = set(y_pred)
    unknown_tags = pred_tags - reference_tags
    
    # Initialize labels for confusion matrix
    labels_for_matrix = unique_pos_values_list.copy()
    # Keep only predictions where the tag exists in reference
    valid_indices = [i for i, tag in enumerate(y_pred) if tag in reference_tags]
    y_true1 = [y_true[i] for i in valid_indices]
    y_pred1 = [y_pred[i] for i in valid_indices]
    print(f"\nWarning: Ignored {len(unknown_tags)} unknown tags: {unknown_tags}")
    print(f"Excluded {len(y_pred) - len(valid_indices)} tokens from evaluation")
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true1, 
        y_pred1, 
        labels=unique_pos_values_list,
        average=None,
        zero_division=0
    )

    # Calculate weighted averages
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true1, 
        y_pred1, 
        labels=labels_for_matrix,
        average='weighted',
        zero_division=0
    )
    
    # Calculate macro averages
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true1, 
        y_pred1, 
        labels=labels_for_matrix,
        average='macro',
        zero_division=0
    )
    
    # Calculate micro averages
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        y_true1, 
        y_pred1, 
        labels=labels_for_matrix,
        average='micro',
        zero_division=0
    )
    
    # Calculate overall accuracy
    accuracy = accuracy_score(y_true1, y_pred1)
    
    balanced_accuracy = balanced_accuracy_score(y_true1, y_pred1)
    
    # Create detailed per-class DataFrame
    detailed_metrics = pd.DataFrame({
        'POS Tag': labels_for_matrix,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Support': support
    })
    
    # Add average rows
    average_rows = pd.DataFrame({
        'POS Tag': ['Weighted Avg', 'Macro Avg', 'Micro Avg'],
        'Precision': [weighted_precision, macro_precision, micro_precision],
        'Recall': [weighted_recall, macro_recall, micro_recall],
        'F1 Score': [weighted_f1, macro_f1, micro_f1],
        'Support': [support.sum(), support.sum(), support.sum()]
    })
    
    detailed_metrics = pd.concat([detailed_metrics, average_rows], ignore_index=True)
    
    # Format numeric columns
    for col in ['Precision', 'Recall', 'F1 Score']:
        detailed_metrics[col] = detailed_metrics[col].map('{:.4f}'.format)
    
    detailed_metrics['Support'] = detailed_metrics['Support'].astype(int)
    
    # Create and plot confusion matrix
    plt.figure(figsize=(12, 8))
    cm = confusion_matrix(y_true, y_pred, labels=labels_for_matrix)
    cm_df = pd.DataFrame(cm, 
                        index=labels_for_matrix,
                        columns=labels_for_matrix)
    
    # Calculate percentages for annotations
    cm_percentages = cm_df.div(cm_df.sum(axis=1), axis=0) * 100
    
    # Create heatmap
    sns.heatmap(cm_df, 
                annot=True, 
                fmt='d',
                cmap='Blues',
                xticklabels=labels_for_matrix,
                yticklabels=labels_for_matrix)
    
    plt.title('Confusion Matrix - NAF6195')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Create percentage-based confusion matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm_percentages, 
                annot=True, 
                fmt='.1f',
                cmap='RdYlBu_r',
                xticklabels=labels_for_matrix,
                yticklabels=labels_for_matrix)
    
    plt.title('Confusion Matrix (Percentages) - NAF6195')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Display metrics
    print("\n=== CLASSIFICATION METRICS SUMMARY ===")
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    print(f"\nBalanced Accuracy: {balanced_accuracy:.4f}")
    
    print("\n=== AVERAGES COMPARISON ===")
    print("\nMicro Average:")
    print(f"Precision: {micro_precision:.4f}")
    print(f"Recall: {micro_recall:.4f}")
    print(f"F1 Score: {micro_f1:.4f}")
    
    print("\nMacro Average:")
    print(f"Precision: {macro_precision:.4f}")
    print(f"Recall: {macro_recall:.4f}")
    print(f"F1 Score: {macro_f1:.4f}")
    
    print("\nWeighted Average:")
    print(f"Precision: {weighted_precision:.4f}")
    print(f"Recall: {weighted_recall:.4f}")
    print(f"F1 Score: {weighted_f1:.4f}")
    
    print("\n=== DETAILED METRICS PER CLASS ===")
    print("\n" + detailed_metrics.to_string(index=False))
    
    # Calculate and display error analysis
    errors = pd.DataFrame({
        'True': y_true,
        'Predicted': y_pred
    })
    errors = errors[errors['True'] != errors['Predicted']]
    error_counts = errors.groupby(['True', 'Predicted']).size().reset_index(name='count')
    error_counts = error_counts.sort_values('count', ascending=False)
    
    print("\n=== TOP CLASSIFICATION ERRORS ===")
    print("\nMost common misclassifications (True -> Predicted):")
    print("\n" + error_counts.head(10).to_string(index=False))
    
    return {
        'accuracy': accuracy,
        'micro_avg': {'precision': micro_precision, 'recall': micro_recall, 'f1': micro_f1},
        'macro_avg': {'precision': macro_precision, 'recall': macro_recall, 'f1': macro_f1},
        'weighted_avg': {'precision': weighted_precision, 'recall': weighted_recall, 'f1': weighted_f1},
        'detailed_metrics': detailed_metrics,
        'confusion_matrix': cm_df,
        'confusion_matrix_percentages': cm_percentages,
        'error_analysis': error_counts,
        'unknown_tags': list(unknown_tags) if unknown_tags else []
    }

def save_results(results, base_filename):
    """Save confusion matrices, metrics, and errors to the specified folder."""
    # Get base name without extension
    base_name = os.path.splitext(os.path.basename(base_filename))[0]
    folder_name = base_name
    os.makedirs(folder_name, exist_ok=True)
    
    # Save confusion matrix image
    fig_cm, ax_cm = plt.subplots(figsize=(12, 8))
    sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig_cm.savefig(os.path.join(folder_name, f'{base_name}_confusion_matrix.png'))
    plt.close(fig_cm)
    
    # Save percentage-based confusion matrix image
    fig_cm_pct, ax_cm_pct = plt.subplots(figsize=(12, 8))
    sns.heatmap(results['confusion_matrix_percentages'], annot=True, fmt='.1f', cmap='RdYlBu_r', ax=ax_cm_pct)
    plt.title('Confusion Matrix (Percentages)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig_cm_pct.savefig(os.path.join(folder_name, f'{base_name}_confusion_matrix_percentages.png'))
    plt.close(fig_cm_pct)
    
    # Save confusion matrices to separate Excel files
    results['confusion_matrix'].to_excel(os.path.join(folder_name, f'{base_name}_confusion_matrix_counts.xlsx'))
    results['confusion_matrix_percentages'].to_excel(os.path.join(folder_name, f'{base_name}_confusion_matrix_percentages.xlsx'))
    
    # Save metrics and errors
    results['detailed_metrics'].to_excel(os.path.join(folder_name, f'{base_name}_detailed_metrics.xlsx'), index=False)
    results['error_analysis'].to_excel(os.path.join(folder_name, f'{base_name}_error_analysis.xlsx'), index=False)
    
    # Save unknown tags (if any)
    if results['unknown_tags']:
        with open(os.path.join(folder_name, f'{base_name}_unknown_tags.txt'), 'w') as f:
            f.write("\n".join(results['unknown_tags']))
    
    # Save classification metrics as a PNG plot
    fig_metrics, ax_metrics = plt.subplots(figsize=(12, 8))
    sns.heatmap(results['detailed_metrics'].set_index('POS Tag').drop(columns='Support').astype(float).transpose(), annot=True, fmt='.4f', cmap='viridis',  ax=ax_metrics)
    plt.title('Classification Metrics')
    plt.xlabel('POS Tag')
    plt.ylabel('Metric')
    plt.tight_layout()
    fig_metrics.savefig(os.path.join(folder_name, f'{base_name}_classification_metrics.png'))
    plt.close(fig_metrics)

    # Save classification metrics to Excel
    results['detailed_metrics'].to_excel(os.path.join(folder_name, f'{base_name}_classification_metrics.xlsx'), index=False)

    print(f"Results saved in folder: {folder_name}")

if __name__ == "__main__":
    file = "NAF6195_tagged_aya_zero_shot.xlsx" #prediction file
    df_gold = pd.read_excel("../data/NAF_reference.xlsx") #reference file
    df_pred = pd.read_excel(file)
    
    # Prepare the data
    combined_df = pd.concat([df_gold, df_pred], axis=1)
    combined_df = combined_df[combined_df["upos"] != "missing"]
    unique_pos_values_list = combined_df['POS'].unique().tolist()
    y_true = combined_df['POS']
    y_pred = combined_df['upos']
    
    # Calculate and display metrics
    results = calculate_and_display_metrics(y_true, y_pred, unique_pos_values_list)
    
    # Save results
    save_results(results, file)