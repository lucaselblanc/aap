use std::fs;
use std::io::{self, Write};

/* Probabilistic Abbreviation Algorithm (PAA) */
/* Written by Lucas Leblanc*/

fn main() {
    let dictionary = load_dictionary("dictionary[PT-BR].txt");

    loop {       

          let abbreviation = get_user_input("[exit to close], Enter abbreviation: ");
          let candidates = get_candidates(&dictionary, &abbreviation);

          if abbreviation.to_lowercase() == "exit" {
              println!("Program stopped!");
              break;
          }

          if candidates.is_empty() {
              println!("No words found for the abbreviation '{}'.", abbreviation);
          }

          let best_match = apply_second_test(&candidates, &abbreviation);

          if let Some(word) = best_match {
              println!("Most likely word for the abbreviation '{}': {}", abbreviation, word);
          } else {
              println!("No words could be identified with high accuracy!");
          }
    }
}

fn load_dictionary(file_path: &str) -> Vec<String> {
    let content = fs::read_to_string(file_path)
        .expect("Error reading dictionary file!");
    let words: Vec<String> = content.lines().map(|line| line.to_string()).collect();
    words
}

fn get_user_input(prompt: &str) -> String {
    print!("{} ", prompt);
    io::stdout().flush().unwrap();
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    input.trim().to_uppercase()
}

//Convert letters -> numbers 
fn letter_to_number(c: char) -> i32 {
    //Primes: C, D, F, H, L, N, R, T, X.
    match c.to_ascii_uppercase() {
        'A' => 0,
        'B' => 1,
        'C' => 2, //prime
        'D' => 3, //prime
        'E' => 4,
        'F' => 5, //prime
        'G' => 6,
        'H' => 7, //prime
        'I' => 8,
        'J' => 9,
        'K' => 10,
        'L' => 11, //prime
        'M' => 12,
        'N' => 13, //prime
        'O' => 14,
        'P' => 15,
        'Q' => 16,
        'R' => 17, //prime
        'S' => 18,
        'T' => 19, //prime
        'U' => 20,
        'V' => 21,
        'W' => 22,
        'X' => 23, //prime
        'Y' => 24,
        'Z' => 25,
        _ => -1,
    }
}

//Filter
fn get_candidates(dictionary: &[String], abbreviation: &str) -> Vec<String> {
    dictionary
        .iter()
        .filter(|word| is_sequence_natural(word, abbreviation))
        .cloned()
        .collect()
}

//First proof:
fn is_sequence_natural(word: &str, abbreviation: &str) -> bool {
    let abbrev_numbers: Vec<i32> = abbreviation
        .chars()
        .map(letter_to_number)
        .filter(|&num| num >= 0)
        .collect();

    let word_numbers: Vec<i32> = word
        .chars()
        .map(letter_to_number)
        .filter(|&num| num >= 0)
        .collect();

    if abbrev_numbers.is_empty() 
        || word_numbers.is_empty() 
        || abbrev_numbers[0] != word_numbers[0] 
        || abbrev_numbers.last() != word_numbers.last()
    {
        return false;
    }

    let mut abbrev_iter = abbrev_numbers.iter();
    let mut current_abbrev = abbrev_iter.next();

    for num in word_numbers {
        if Some(&num) == current_abbrev {
            current_abbrev = abbrev_iter.next();
        }
        if current_abbrev.is_none() {
            return true;
        }
    }

    false
}

//Second proof:
fn apply_second_test(candidates: &[String], abbreviation: &str) -> Option<String> {
    let prime_sum: i32 = abbreviation.chars()
        .filter(|&c| is_prime(letter_to_number(c)))
        .map(letter_to_number)
        .sum();

    let scored_candidates: Vec<(String, f64)> = candidates
        .iter()
        .map(|word| {
            let composite_total: i32 = word.chars()
                .filter(|&c| !is_prime(letter_to_number(c)))
                .map(letter_to_number)
                .sum();

            let prime_count = word.chars()
                .filter(|&c| is_prime(letter_to_number(c)))
                .count();

            //Adjustment to abbreviation to keep or discard prime numbers:
            /*let score = composite_total as f64 / prime_sum as f64;*/
            let score = (composite_total as f64 + prime_count as f64) / prime_sum as f64;
            println!("Word: '{}', composite_total: {}, prime_sum: {}, score: {}", word, composite_total, prime_sum, score);

            (format!("{} {:.7}%", word, score), score)
        })
        .collect();

    println!(
        "Candidates after the first proof: {:?}",
        scored_candidates.iter().map(|(desc, _)| desc.clone()).collect::<Vec<_>>()
    );

    scored_candidates
        .into_iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .map(|(desc, _)| desc)
}

fn is_prime(num: i32) -> bool {
    matches!(num, 2 | 3 | 5 | 7 | 11 | 13 | 17 | 19 | 23)
}
