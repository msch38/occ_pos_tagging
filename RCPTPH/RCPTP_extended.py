import pandas as pd
import numpy as np
import glob

def calculate_matching_percentage(ref_pos, pred_pos):
    """
    Calculate the percentage of matching POS tags between reference and prediction.
    """
    total_tags = len(ref_pos)
    correct_tags = sum([1 for ref, pred in zip(ref_pos, pred_pos) if ref == pred])
    return (correct_tags / total_tags) * 100 if total_tags > 0 else 0

def find_first_matching_sentence(reference_file, prediction_file, sentences_file):
    """
    Find and display the percentage of matching POS tags for each sentence and return results.
    """
    # Read the files
    ref_df = pd.read_excel(reference_file)
    pred_df = pd.read_excel(prediction_file)
    
    with open(sentences_file, 'r', encoding='utf-8') as f:
        sentences = [s.strip() for s in f.readlines()]
    
    current_idx = 0
    results = []
    print(f"\nProcessing {prediction_file}...\n")
    
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
        
        sentence_result = {
            'Prediction_File': prediction_file.split('/')[-1],
            'Sentence_ID': sentence_idx + 1,
            'Match_Percentage': match_percentage
        }
        results.append(sentence_result)
        
        current_idx += sentence_length
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate mean and standard deviation
    mean_match = np.mean(results_df['Match_Percentage'])
    std_match = np.std(results_df['Match_Percentage'])
    
    print(f"Mean POS Tag Matching Percentage for {prediction_file}: {mean_match:.2f}%")
    print(f"Standard Deviation for {prediction_file}: {std_match:.2f}%")
    
    return results_df, mean_match, std_match

def main():
    reference_file = 'NAF_reference.xlsx'  # Path to your reference Excel file
    sentences_file = 'NAF6195.txt'
    prediction_files = glob.glob('../NAF6195/*.xlsx')  # Get all prediction files
    
    all_results = []
    summary_results = []
    
    for prediction_file in prediction_files:
        results_df, mean_match, std_match = find_first_matching_sentence(reference_file, prediction_file, sentences_file)
        all_results.append(results_df)
        summary_results.append({'Prediction_File': prediction_file.split('/')[-1], 'Mean_Match': mean_match, 'Std_Dev': std_match})
    
    # Save all sentence-level results in one Excel file
    final_results_df = pd.concat(all_results, ignore_index=True)
    final_results_df.to_excel("all_results_NAF.xlsx", index=False)
    
    # Save summary results (mean and standard deviation) in another sheet
    summary_df = pd.DataFrame(summary_results)
    summary_df.to_excel("all_results_NAF.xlsx", index=False)
    
    print("All results saved in 'all_results_NAF.xlsx'")
    
if __name__ == "__main__":
    main()