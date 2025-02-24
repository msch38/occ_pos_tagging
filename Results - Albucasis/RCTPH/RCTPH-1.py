import pandas as pd
import os
from pathlib import Path

def evaluate_pos_tagging_debug(reference_file, prediction_file, sentences_file):
    """
    Debug version of the evaluation function with detailed logging
    """
    print("\nDEBUG INFORMATION:")
    print("-" * 50)
    
    # Read the data
    reference_df = pd.read_excel(reference_file)
    prediction_df = pd.read_excel(prediction_file)
    
    print(f"Reference file shape: {reference_df.shape}")
    print(f"Prediction file shape: {prediction_df.shape}")
    
    # Read sentences file
    with open(sentences_file, 'r') as f:
        sentences = f.readlines()
    print(f"Number of sentences: {len(sentences)}")
    
    # Calculate total words in sentences
    total_words = sum(len(sentence.strip().split()) for sentence in sentences)
    print(f"Total words in sentences file: {total_words}")
    
    # Create sentence IDs
    sentence_ids = []
    for sentence_idx, sentence in enumerate(sentences):
        words = sentence.strip().split()
        sentence_ids.extend([sentence_idx] * len(words))
    
    print(f"Length of sentence_ids: {len(sentence_ids)}")
    
    # Add sentence IDs to dataframes
    reference_df['Sentence_ID'] = sentence_ids[:len(reference_df)]
    prediction_df['Sentence_ID'] = sentence_ids[:len(prediction_df)]
    
    print("\nMerge Information:")
    print(f"Reference unique Lemmas: {len(reference_df['Lemma'].unique())}")
    print(f"Prediction unique words: {len(prediction_df['word'].unique())}")
    
    # Merge reference and prediction
    merged_df = pd.merge(
        reference_df, 
        prediction_df, 
        left_on=['Sentence_ID', 'Lemma'], 
        right_on=['Sentence_ID', 'word'], 
        suffixes=('_ref', '_pred')
    )
    
    print(f"Merged dataframe shape: {merged_df.shape}")
    
    # Calculate correctness
    merged_df['Correct'] = merged_df['POS'] == merged_df['upos']
    
    # Show sample of mismatches
    mismatches = merged_df[~merged_df['Correct']].head()
    print("\nSample of POS tag mismatches:")
    print(mismatches[['Sentence_ID', 'Lemma', 'POS', 'upos', 'Correct']].to_string())
    
    # Calculate ratios both ways
    # Method 1: Using all() for completely correct sentences
    correct_sentences = merged_df.groupby('Sentence_ID')['Correct'].all()
    ratio_1 = correct_sentences.mean()
    
    # Method 2: Per-word accuracy
    ratio_2 = merged_df['Correct'].mean()
    
    print(f"\nRatio by sentence (all words must be correct): {ratio_1:.4f}")
    print(f"Ratio by individual words: {ratio_2:.4f}")
    
    # Count sentences
    total_sentences = len(merged_df['Sentence_ID'].unique())
    correct_sentence_count = correct_sentences.sum()
    
    print(f"\nTotal sentences evaluated: {total_sentences}")
    print(f"Completely correct sentences: {correct_sentence_count}")
    
    return ratio_1, merged_df

def main():
    # Define file patterns
    reference_file = 'REF_Albuc_1.xlsx'
    sentences_file = 'Albuc1.txt'
    
    # Get all prediction files
    prediction_files = [f for f in os.listdir() if f.endswith('.xlsx') 
                       and f.startswith('Albuc1_tagged') 
                       and f != reference_file]
    
    # Process each prediction file
    for pred_file in prediction_files:
        print(f"\nProcessing: {pred_file}")
        print("=" * 50)
        correct_ratio, merged_df = evaluate_pos_tagging_debug(reference_file, pred_file, sentences_file)
        
        if correct_ratio is not None:
            print(f"\nFinal ratio of correctly POS-tagged sentences: {correct_ratio:.4f}")

if __name__ == "__main__":
    main()