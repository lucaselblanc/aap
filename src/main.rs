/******************************************************************************************************
 * This file is part of the AAP distribution: (https://github.com/lucaselblanc/aap) *
 * Copyright (c) 2024, 2025 Lucas Leblanc.                                                            *
 * Distributed under the MIT software license, see the accompanying.                                  *
 * file COPYING or https://www.opensource.org/licenses/mit-license.php.                               *
 ******************************************************************************************************/

/*****************************************
 * Probabilistic Abbreviation Algorithm  *
 * Written by Lucas Leblanc              *
******************************************/

use std::collections::HashMap;
use std::io::{self, Write};

fn main() {
    let content = include_str!("dictionary[PT-BR].txt");
    let dictionary: Vec<String> = content
        .lines()
        .map(|line| line.trim().to_uppercase())
        .collect();
    
    let entropy_table = build_entropy_table(&dictionary);

    loop {
        let abbreviation = get_user_input("[exit to close], Enter abbreviation: ");
        
        if abbreviation.to_lowercase() == "exit" {
            println!("Program stopped!");
            break;
        }

        let candidates = get_candidates(&dictionary, &abbreviation);

        if candidates.is_empty() {
            println!("No words found for the abbreviation '{}'.", abbreviation);
            continue;
        }

        let best_match = apply_second_test(&candidates, &abbreviation, &entropy_table);

        if let Some(word) = best_match {
            println!("Most likely word for '{}': {}", abbreviation, word);
        }
    }
}

/// Entropy Table: H(c) = -log2(P(c))
fn build_entropy_table(dictionary: &[String]) -> HashMap<char, f64> {
    let mut counts = HashMap::new();
    let mut total_chars = 0.0f64;

    for word in dictionary {
        for c in word.chars().filter(|c| c.is_alphabetic()) {
            *counts.entry(c).or_insert(0.0f64) += 1.0;
            total_chars += 1.0;
        }
    }

    counts.into_iter()
        .map(|(c, count)| {
            let probability: f64 = count / total_chars;
            let entropy = -probability.log2(); 
            (c, entropy)
        })
        .collect()
}

fn get_user_input(prompt: &str) -> String {
    print!("{} ", prompt);
    io::stdout().flush().unwrap();
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    input.trim().to_uppercase()
}

fn get_candidates(dictionary: &[String], abbreviation: &str) -> Vec<String> {
    dictionary
        .iter()
        .filter(|word| is_sequence_natural(word, abbreviation))
        .cloned()
        .collect()
}

/// First Proof
fn is_sequence_natural(word: &str, abbreviation: &str) -> bool {
    let abbrev_chars: Vec<char> = abbreviation.chars().collect();
    let word_chars: Vec<char> = word.chars().collect();

    if abbrev_chars.is_empty() || word_chars.is_empty() 
        || abbrev_chars[0] != word_chars[0] 
        || abbrev_chars.last() != word_chars.last() {
        return false;
    }

    let mut word_iter = word_chars.iter();
    for &target in &abbrev_chars {
        if !word_iter.any(|&c| c == target) {
            return false;
        }
    }
    true
}

/// Second Proof -> Shannon Entropy
fn apply_second_test(candidates: &[String], abbreviation: &str, entropy_table: &HashMap<char, f64>) -> Option<String> {
    let default_entropy = 10.0f64;

    let abbrev_info: f64 = abbreviation.chars()
        .map(|c| *entropy_table.get(&c).unwrap_or(&default_entropy))
        .sum();

    let mut scored_candidates: Vec<(String, f64)> = candidates
        .iter()
        .map(|word| {
            let word_info: f64 = word.chars()
                .map(|c| *entropy_table.get(&c).unwrap_or(&default_entropy))
                .sum();

            let score = abbrev_info / word_info;
            (word.clone(), score)
        })
        .collect();

    scored_candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("\nTop candidates (Information Density):");
    for (word, score) in scored_candidates.iter().take(3) {
        println!(" - {} ({:.2}%)", word, score * 100.0);
    }

    scored_candidates.first().map(|(word, score)| format!("{} ({:.2}%)", word, score * 100.0))
}