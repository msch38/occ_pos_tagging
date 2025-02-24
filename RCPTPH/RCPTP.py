# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

def calculate_matching_percentage(ref_pos, pred_pos):
    """
    Calculate the percentage of matching POS tags between reference and prediction.
    """
    total_tags = len(ref_pos)
    correct_tags = sum([1 for ref, pred in zip(ref_pos, pred_pos) if ref == pred])
    return (correct_tags / total_tags) * 100 if total_tags > 0 else 0

def find_first_matching_sentence(reference_file, prediction_file, sentences_file):
    """
    Find and display the percentage of matching POS tags for each sentence.
    """
    # Read the files
    ref_df = pd.read_excel(reference_file)
    pred_df = pd.read_excel(prediction_file)
    
    with open(sentences_file, 'r', encoding='utf-8') as f:
        sentences = [s.strip() for s in f.readlines()]
    
    current_idx = 0
    results = []
    print("\nCalculating POS tag match percentages for each sentence...\n")
    
    for sentence_idx, sentence in enumerate(sentences):
        words = sentence.split()
        sentence_length = len(words)
        
        # Get current sentence slice from both dataframes
        ref_slice = ref_df.iloc[current_idx:current_idx + sentence_length]
        pred_slice = pred_df.iloc[current_idx:current_idx + sentence_length]
        
        # Get POS tags
        ref_pos = ref_slice['POS'].tolist()
        pred_pos = pred_slice['upos'].tolist()
        
        # Calculate the matching percentage
        match_percentage = calculate_matching_percentage(ref_pos, pred_pos)
        
        print(f"Sentence {sentence_idx + 1}: {match_percentage:.2f}% of POS tags match")
        
        # Optional: Show a breakdown if you want to compare each word's POS tag
        comparison = pd.DataFrame({
            'Word': ref_slice['Lemma'],
            'Reference_POS': ref_slice['POS'],
            'Predicted_POS': pred_slice['upos'],
            'Match': [ref == pred for ref, pred in zip(ref_pos, pred_pos)]
        })
        
        print("\nWord-by-word comparison:")
        print(comparison.to_string(index=False))
        print("-" * 80)
        sentence_result = {
            'Sentence_ID': sentence_idx + 1,
            'Match_Percentage': match_percentage
        }
        results.append(sentence_result)
        
        current_idx += sentence_length
    
    results_df = pd.DataFrame(results)
    
    # Calculate mean and standard deviation
    mean_match = np.mean(results_df['Match_Percentage'])
    std_match = np.std(results_df['Match_Percentage'])
    
    print(f"\nMean POS Tag Matching Percentage: {mean_match:.2f}%")
    print(f"Standard Deviation of Matching Percentages: {std_match:.2f}%")
    
    results_df.to_excel("result.xlsx",index=False)

def main():
    reference_file = 'REF_Albuc_1.xlsx'  # Path to your reference Excel file
    prediction_file = '../Albucasis/Albuc1_tagged_aya_prompt2.xlsx'  # Path to your prediction Excel file
    sentences_file = 'Albuc1.txt'
    
    find_first_matching_sentence(reference_file, prediction_file, sentences_file)

if __name__ == "__main__":
    main()
