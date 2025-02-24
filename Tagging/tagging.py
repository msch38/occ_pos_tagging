# -*- coding: utf-8 -*-
import json
import sys
import re
import time
import ollama
import pandas as pd
from datetime import datetime
from pathlib import Path

class OccPoSTagger:
    def __init__(self):
        self.model_name = "mistral" #model name
        self.ctx = 8196
        self.ud_tags = {"ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PRON", "PROPN", "PUNCT", "SCONJ", "VERB", "X"}
        self.problems_log = []

    def log_problem(self, problem_type, description, chunk_num=None, word=None, details=None):
        problem = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'problem_type': problem_type,
            'description': description,
            'chunk_number': chunk_num,
            'word': word,
            'details': details
        }
        self.problems_log.append(problem)

    def save_problems_log(self, output_file):
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("=== OCCITAN PoS TAGGER PROBLEM LOG ===\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                problem_types = {}
                for problem in self.problems_log:
                    prob_type = problem['problem_type']
                    if prob_type not in problem_types:
                        problem_types[prob_type] = []
                    problem_types[prob_type].append(problem)

                for prob_type, problems in problem_types.items():
                    f.write(f"\n=== {prob_type.upper()} ===\n")
                    for problem in problems:
                        f.write(f"\nTime: {problem['timestamp']}")
                        if problem['chunk_number'] is not None:
                            f.write(f"\nChunk: {problem['chunk_number']}")
                        if problem['word'] is not None:
                            f.write(f"\nWord: {problem['word']}")
                        f.write(f"\nDescription: {problem['description']}")
                        if problem['details']:
                            f.write(f"\nDetails: {problem['details']}")
                        f.write("\n" + "-"*50 + "\n")

        except Exception as e:
            print(f"Error saving problems log: {str(e)}")
            self.log_problem("LOG_SAVE_ERROR", "Failed to save problems log", details=str(e))

    def read_text_file(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except FileNotFoundError:
            self.log_problem("FILE_ERROR", f"Input file '{file_path}' not found")
            print(f"Error: Input file '{file_path}' not found.")
            sys.exit(1)
        except Exception as e:
            self.log_problem("FILE_ERROR", f"Error reading file '{file_path}'", details=str(e))
            print(f"Error reading file '{file_path}': {str(e)}")
            sys.exit(1)

    def build_chunks(self, text, chunk_size=50):
        words = text.split()
        chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
        return chunks, words

    def process_chunk(self, chunk, chunk_num, total_chunks, log_file, retries=3, backoff=2):
        print(f"\nProcessing chunk {chunk_num}/{total_chunks}")
        print("Input chunk words:", chunk)
        
        mismatched_words = []

        for attempt in range(retries):
            '''adapt the prompt'''
            try:
                prompt = """You are a medieval Occitan language expert specializing in linguistic analysis. This language is related to Catalan and Latin. In this text there is a high variety of spelling variations having the same meaning.
                This is an example for spelling variation: homps, ome, om, omen, omne, hom, home. Another example is: acayson, achaison, acheison, acheson, aqueison, caiso, caison, cason, cayson, chaizo, queison or gaug, gauc, gautz, jau, jauvi.
                Your task is to analyze the given text and assign Universal Dependencies Part-of-Speech (UD POS) tags to each word.
                Return the results as a JSON array of objects, each containing only the 'word' and 'upos' keys.
                Ensure that the JSON array is properly formatted and closed.
                The output must be only the JSON array without any additional text, explanations, or formatting
                """

                response = ollama.generate(model=self.model_name, 
                                            prompt=prompt + "\n" + chunk, 
                                            options={"num_ctx": self.ctx})

                response_content = response['response']
                print("Response model: ", response_content)
                
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"\n\n--- Chunk {chunk_num}/{total_chunks} ---\n")
                    f.write(f"Input text: {chunk}\n")
                    f.write(f"Response:\n{response_content}\n")

                json_str = self._extract_json(response_content)
                if not json_str:
                    self.log_problem("JSON_EXTRACTION_ERROR", 
                                   "Failed to extract JSON from response",
                                   chunk_num=chunk_num,
                                   details=response_content)
                    continue

                try:
                    tagged_data = json.loads(json_str)
                    if not isinstance(tagged_data, list):
                        self.log_problem("INVALID_JSON_STRUCTURE",
                                       "Response is not a list",
                                       chunk_num=chunk_num,
                                       details=json_str)
                        continue

                    chunk_dict = {'word': [], 'upos': []}
                    original_words = chunk.split()

                    word_tag_map = {item['word']: item.get('upos', 'missing') 
                                  for item in tagged_data if 'word' in item}

                    for word in original_words:
                        chunk_dict['word'].append(word)
                        tag = word_tag_map.get(word, 'missing')

                        if word not in word_tag_map:
                            self.log_problem("MISSING_TAG",
                                           "Word not found in model response",
                                           chunk_num=chunk_num,
                                           word=word)
                            mismatched_words.append((word, 'missing'))
                        if tag not in self.ud_tags:
                            self.log_problem("INVALID_TAG",
                                           f"Invalid UD tag '{tag}' assigned",
                                           chunk_num=chunk_num,
                                           word=word,
                                           details=f"Tag: {tag}")
                            tag = 'missing'

                        chunk_dict['upos'].append(tag)

                    print("\nChunk dictionary:")
                    print("Words:", chunk_dict['word'])
                    print("Tags:", chunk_dict['upos'])

                    if len(chunk_dict['word']) != len(original_words):
                        self.log_problem("WORD_COUNT_MISMATCH",
                                       "Word count mismatch between input and processed output",
                                       chunk_num=chunk_num,
                                       details=f"Expected: {len(original_words)}, Got: {len(chunk_dict['word'])}")

                    cleaned_data = [{'word': w, 'upos': t} 
                                  for w, t in zip(chunk_dict['word'], chunk_dict['upos'])]
                    return cleaned_data, mismatched_words

                except json.JSONDecodeError as e:
                    self.log_problem("JSON_DECODE_ERROR",
                                   "Failed to decode JSON response",
                                   chunk_num=chunk_num,
                                   details=str(e))
                    continue

            except Exception as e:
                self.log_problem("PROCESSING_ERROR",
                               f"Error on attempt {attempt + 1}",
                               chunk_num=chunk_num,
                               details=str(e))
                if attempt < retries - 1:
                    print(f"Retrying in {backoff} seconds...")
                    time.sleep(backoff)

        self.log_problem("CHUNK_FAILURE",
                        "Failed to process chunk after all attempts",
                        chunk_num=chunk_num)
        return None, mismatched_words

    def _extract_json(self, response_content):
        try:
            json_match = re.search(r"```json\s*\n(.*?)\n```", response_content, re.DOTALL)
            if json_match:
                return json_match.group(1)
            
            if response_content.strip().startswith('[') and response_content.strip().endswith(']'):
                return response_content.strip()
            
            json_array_match = re.search(r'(\[.*\])', response_content, re.DOTALL)
            if json_array_match:
                return json_array_match.group(1)
            
            return None
        except Exception as e:
            self.log_problem("JSON_EXTRACTION_ERROR",
                           "Error in JSON extraction",
                           details=str(e))
            return None

    def create_output_dictionary(self, processed_chunks, original_words):
        output_dict = {
            'word': [],
            'upos': []
        }
        
        try:
            word_tag_map = {}
            for chunk_data in processed_chunks:
                if chunk_data:
                    for item in chunk_data:
                        word_tag_map[item['word']] = item['upos']

            for word in original_words:
                output_dict['word'].append(word)
                tag = word_tag_map.get(word, 'missing')
                if tag not in self.ud_tags:
                    self.log_problem("INVALID_TAG_IN_OUTPUT",
                                   "Invalid tag in final output",
                                   word=word,
                                   details=f"Tag: {tag}")
                    tag = 'missing'
                output_dict['upos'].append(tag)
            print(output_dict)
            return output_dict
        except Exception as e:
            self.log_problem("OUTPUT_CREATION_ERROR",
                           "Error creating output dictionary",
                           details=str(e))
            raise

    def save_to_excel(self, output_dict, output_file):
        try:
            df = pd.DataFrame(output_dict)
            df.to_excel(output_file, index=False)
            print(f"Results successfully saved to '{output_file}'")

        except Exception as e:
            self.log_problem("FILE_SAVE_ERROR",
                       "Error saving files",
                       details=str(e))
            print(f"Error saving files: {str(e)}")
            sys.exit(1)
    
def sanitize_filename(filename):
    """
    Replace invalid characters in filename with underscores.
    """
    return re.sub(r':', '_', filename)

def main():
    start_time = time.time()

    tagger = OccPoSTagger()
    
    model_name = tagger.model_name
    
    # Input and output file paths
    input_file = "./Albuc1.txt" # text: Albuc1.txt or NAF6195.txt
    path = Path(input_file)
         
    output_file = sanitize_filename(f"{path.stem}_tagged_{model_name}_prompt2.xlsx")
    log_file = sanitize_filename(f"{path.stem}_responses_{model_name}_prompt2.txt")
    problems_file = sanitize_filename(f"{path.stem}_problems_log_{model_name}_prompt2.txt")
    mismatched_words_file = sanitize_filename(f"{path.stem}_mismatched_words_{model_name}_prompt2.txt")
    elapsed_time_file = sanitize_filename(f"{path.stem}_elapsed_time_{model_name}_prompt2.txt")
    
    # Read input text
    text = tagger.read_text_file(input_file)

    # Step 1: Build chunks
    chunks, original_words = tagger.build_chunks(text, chunk_size=50)
    print(f"Created {len(chunks)} chunks from input text")

    # Step 2: Process chunks
    processed_chunks = []
    mismatched_words = []  # List to collect mismatched words
    for i, chunk in enumerate(chunks, 1):
        chunk_result, chunk_mismatched_words = tagger.process_chunk(chunk, i, len(chunks), log_file)
        processed_chunks.append(chunk_result)
        mismatched_words.extend(chunk_mismatched_words)

    # Save mismatched words (original + output) to a file
    if mismatched_words:
        with open(mismatched_words_file, 'w', encoding='utf-8') as f:
            for original_word, output_word in mismatched_words:
                f.write(f"{original_word}\t{output_word}\n")  # Tab-separated for easy reading
        print(f"Mismatched words saved to '{mismatched_words_file}'")

    # Steps 3 & 4: Create and validate output dictionary
    output_dict = tagger.create_output_dictionary(processed_chunks, original_words)

    # Step 5: Save to Excel
    tagger.save_to_excel(output_dict, output_file)
    
    # Save problems log
    tagger.save_problems_log(problems_file)
    
    total_time = time.time() - start_time
    print(f"\nTotal processing time: {total_time:.2f} seconds")
    print(f"Problems log saved to '{problems_file}'")
    
    with open(elapsed_time_file, 'w', encoding='utf-8') as f:
        f.write(f"Total processing time for model '{model_name}' and text '{path.stem}': {total_time:.2f} seconds\n")

if __name__ == "__main__":
    main()
