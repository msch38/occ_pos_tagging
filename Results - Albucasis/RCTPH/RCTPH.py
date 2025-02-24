# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 08:27:08 2025
@author: Matthias Schöffel
@MaiL: Matthias.Schoeffel@lmu.de
"""

import os
import pandas as pd

# Folder where the files are located
folder_path = './'  # Path to the folder containing your files

# Reference file (this remains the same)
reference_file = 'REF_Albuc_1.xlsx'

# Read the reference file
reference_df = pd.read_excel(reference_file)

# Ensure the structure of the reference file is correct
assert 'Lemma' in reference_df.columns, 'Reference file must have a "Lemma" column'
assert 'POS' in reference_df.columns, 'Reference file must have a "POS" column'

# List all prediction files in the folder
prediction_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx') and f != 'REF_Albuc_1.xlsx']

# Initialize a list to store the ratios for each prediction file
ratios = []

# Loop over each prediction file in the folder
for prediction_file in prediction_files:
    prediction_path = os.path.join(folder_path, prediction_file)
    prediction_df = pd.read_excel(prediction_path)

    # Ensure the structure of the prediction file is correct
    assert 'word' in prediction_df.columns, 'Prediction file must have a "word" column'
    assert 'upos' in prediction_df.columns, 'Prediction file must have a "Upos" column'

    # Read the sentences file and assign Sentence_ID (assuming it is fixed for all prediction files)
    sentences_file = 'Albuc1.txt'  # Path to your text file with sentences
    with open(sentences_file, 'r') as f:
        sentences = f.readlines()

    # Assign Sentence_ID based on the order of sentences
    sentence_ids = []
    for sentence_idx, sentence in enumerate(sentences):
        words_in_sentence = sentence.strip().split()  # Split sentence into words and remove any leading/trailing whitespace
        sentence_ids.extend([sentence_idx] * len(words_in_sentence))  # Assign the same ID to all words in the sentence

    # Add the sentence_ids to both reference and prediction data
    reference_df['Sentence_ID'] = sentence_ids[:len(reference_df)]
    prediction_df['Sentence_ID'] = sentence_ids[:len(prediction_df)]

    # Merge reference and prediction on Word and Sentence_ID columns
    merged_df = pd.merge(reference_df, prediction_df, left_on=['Sentence_ID', 'Lemma'], right_on=['Sentence_ID', 'word'], suffixes=('_ref', '_pred'))

    # Calculate the correctness of each POS tag
    merged_df['Correct'] = merged_df['POS'] == merged_df['upos']

    # Calculate the ratio of correctly POS-tagged sentences
    correct_sentences = merged_df.groupby('Sentence_ID')['Correct'].all()
    correct_ratio = correct_sentences.mean()

    # Append the result to the list of ratios
    ratios.append((prediction_file, correct_ratio))

# Print the ratios for each prediction file
for prediction_file, correct_ratio in ratios:
    print(f'File: {prediction_file}, Ratio of correctly POS-tagged sentences: {correct_ratio:.2f}')
