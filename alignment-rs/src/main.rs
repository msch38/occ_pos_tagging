use std::fs::File;
use std::io::{self, Read, BufReader};
use std::path::Path;
use std::time::Instant;
use serde::{Deserialize, Serialize};
use regex::Regex;
use rust_xlsxwriter::*;


// Custom SequenceMatcher implementation
struct SequenceMatcher<'a> {
    a: &'a str,
    b: &'a str,
}

impl<'a> SequenceMatcher<'a> {
    fn new(a: &'a str, b: &'a str) -> Self {
        SequenceMatcher { a, b }
    }

    fn find_longest_match(&self, a_start: usize, a_end: usize, b_start: usize, b_end: usize) -> (usize, usize, usize) {
        let a_slice = self.a[a_start..a_end].as_bytes();
        let b_slice = self.b[b_start..b_end].as_bytes();
        
        let mut longest = 0;
        let mut best_i = a_start;
        let mut best_j = b_start;
        
        // Create a matrix to store lengths of longest common substrings
        let mut lengths = vec![vec![0; b_slice.len() + 1]; a_slice.len() + 1];
        
        // Fill the matrix
        for (i, a_char) in a_slice.iter().enumerate() {
            for (j, b_char) in b_slice.iter().enumerate() {
                if a_char == b_char {
                    lengths[i + 1][j + 1] = lengths[i][j] + 1;
                    if lengths[i + 1][j + 1] > longest {
                        longest = lengths[i + 1][j + 1];
                        best_i = i + 1 - longest;
                        best_j = j + 1 - longest;
                    }
                }
            }
        }
        
        (best_i + a_start, best_j + b_start, longest)
    }

    fn get_matching_blocks(&self) -> Vec<(usize, usize, usize)> {
        let mut queue = vec![(0, self.a.len(), 0, self.b.len())];
        let mut matching_blocks = Vec::new();
        
        while let Some((a_start, a_end, b_start, b_end)) = queue.pop() {
            let (i, j, k) = self.find_longest_match(a_start, a_end, b_start, b_end);
            
            if k > 0 {
                if a_start < i && b_start < j {
                    queue.push((a_start, i, b_start, j));
                }
                if i + k < a_end && j + k < b_end {
                    queue.push((i + k, a_end, j + k, b_end));
                }
                matching_blocks.push((i, j, k));
            }
        }
        
        matching_blocks.sort_by_key(|&(i, _, _)| i);
        matching_blocks
    }

    fn ratio(&self) -> f64 {
        if self.a.is_empty() && self.b.is_empty() {
            return 1.0;
        }
        if self.a.is_empty() || self.b.is_empty() {
            return 0.0;
        }

        let matches: usize = self.get_matching_blocks()
            .iter()
            .map(|&(_, _, length)| length)
            .sum();

        let length = self.a.len() + self.b.len();
        (2.0 * matches as f64) / length as f64
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct WordUpos {
    word: String,
    upos: String,
}

#[derive(Debug)]
struct AlignmentResult {
    reference_word: String,
    matched_word: Option<String>,
    similarity: f64,
}

fn read_txt_file(file_path: &str) -> io::Result<String> {
    let mut content = String::new();
    let file = File::open(file_path)?;
    let mut reader = BufReader::new(file);
    reader.read_to_string(&mut content)?;
    Ok(content.trim().to_string())
}

fn tokenize_text(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|s| s.to_string())
        .collect()
}

fn extract_json_data(file_path: &str) -> io::Result<Vec<WordUpos>> {
    let content = read_txt_file(file_path)?;
    let re = Regex::new(r#""word":\s*"([^"]+)",\s*"upos":\s*"([^"]+)"#).unwrap();
    
    let results: Vec<WordUpos> = re
        .captures_iter(&content)
        .map(|cap| WordUpos {
            word: cap[1].to_string(),
            upos: cap[2].to_string(),
        })
        .collect();

    println!("Extracted {} word-UPOS pairs", results.len());
    Ok(results)
}

fn string_similarity(a: &str, b: &str) -> f64 {
    let matcher = SequenceMatcher::new(a, b);
    matcher.ratio()
}

fn split_into_chunks<T: Clone>(list: &[T], chunk_size: usize) -> Vec<Vec<T>> {
    list.chunks(chunk_size)
        .map(|chunk| chunk.to_vec())
        .collect()
}

fn align_string_lists_chunked(
    reference_list: &[String],
    comparison_list: &[WordUpos],
    similarity_threshold: f64,
    chunk_size: usize,
) -> Vec<AlignmentResult> {
    let mut all_alignments = Vec::new();
    let mut used_comparison_indices = std::collections::HashSet::new();

    // Process comparison list to get just the words
    let processed_comparison: Vec<String> = comparison_list
        .iter()
        .map(|item| item.word.clone())
        .collect();

    // Process reference list in chunks
    for chunk_start in (0..reference_list.len()).step_by(chunk_size) {
        let chunk_end = std::cmp::min(chunk_start + chunk_size, reference_list.len());
        let reference_chunk = &reference_list[chunk_start..chunk_end];

        // Process each word in the current chunk
        for ref_word in reference_chunk {
            let mut best_match = None;
            let mut best_score = 0.0;
            let mut best_index = None;

            // Compare with all unused comparison words
            for (i, comp_word) in processed_comparison.iter().enumerate() {
                if !used_comparison_indices.contains(&i) {
                    let score = string_similarity(ref_word, comp_word);
                    if score > best_score && score >= similarity_threshold {
                        best_match = Some(comp_word.clone());
                        best_score = score;
                        best_index = Some(i);
                    }
                }
            }

            if let Some(index) = best_index {
                used_comparison_indices.insert(index);
                all_alignments.push(AlignmentResult {
                    reference_word: ref_word.clone(),
                    matched_word: best_match,
                    similarity: best_score,
                });
            } else {
                all_alignments.push(AlignmentResult {
                    reference_word: ref_word.clone(),
                    matched_word: None,
                    similarity: 0.0,
                });
            }
        }

        println!("Processed chunk {}-{} of {}", chunk_start, chunk_end, reference_list.len());
    }

    all_alignments
}

fn save_to_excel(results: &[AlignmentResult], output_path: &str) -> Result<(), XlsxError> {
    let mut workbook = Workbook::new();
    let mut worksheet = workbook.add_worksheet();

    // Write headers
    worksheet.write_string(0, 0, "Reference Word")?;
    worksheet.write_string(0, 1, "Matched Word")?;
    worksheet.write_string(0, 2, "Similarity")?;

    // Write data
    for (row, result) in results.iter().enumerate() {
        let row = row as u32 + 1;
        worksheet.write_string(row, 0, &result.reference_word)?;
        worksheet.write_string(row, 1, result.matched_word.as_deref().unwrap_or("<<MISSING>>"))?;
        worksheet.write_number(row, 2, result.similarity)?;
    }

    workbook.save(output_path)?;
    Ok(())
}

fn main() -> io::Result<()> {
    let file_path1 = "NAF6195_responses_phi4_zero_shot.txt";
    let file_path2 = "NAF6195.txt";
    
    let text_content2 = read_txt_file(file_path2)?;
    let reference = tokenize_text(&text_content2);
    let comparison = extract_json_data(file_path1)?;

    let chunk_size = 50;
    let start_time = Instant::now();
    let results = align_string_lists_chunked(&reference, &comparison, 0.5, chunk_size);
    let total_time = start_time.elapsed();

    // Generate output file path
    let path = Path::new(file_path1);
    let stem = path.file_stem().unwrap().to_str().unwrap();
    let output_file_path = format!("alignment_{}.xlsx", stem);

    // Save results to Excel
    save_to_excel(&results, &output_file_path).expect("Failed to save Excel file");

    // Print results
    println!("\nAlignment Results:");
    println!("Reference Word | Matched Word | Similarity");
    println!("{}", "-".repeat(45));
    
    for result in &results {
        let match_str = result.matched_word.as_deref().unwrap_or("<<MISSING>>");
        let score_str = if result.matched_word.is_some() {
            format!("{:.2}", result.similarity)
        } else {
            "0.00".to_string()
        };
        println!(
            "{:13} | {:11} | {}",
            result.reference_word, match_str, score_str
        );
    }

    println!(
        "\nTotal time taken for alignment process: {:.2} seconds",
        total_time.as_secs_f64()
    );
    println!("\nResults saved to {}", output_file_path);

    Ok(())
}