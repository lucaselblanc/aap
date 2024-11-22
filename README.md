# Probabilistic Abbreviation Algorithm (PAA)

## Description

The Probabilistic Abbreviation Algorithm (PAA) is designed to recover the original word from a short abbreviation containing the most important and characteristic letters of the word. The algorithm operates through two fundamental tests (proofs):

First Proof: It selects candidate words based on the natural order of numbers derived from the characters in both the abbreviation and the candidate words. This ensures that the sequence of letters in the abbreviation aligns with that in the candidate word.

Second Proof: A probabilistic calculation is performed by comparing the differences between the prime and composite values associated with each letter. This step calculates a score by considering the relevance of prime and composite numbers in the context of the abbreviation, with each type of number playing a crucial role in determining the overall match.

This approach is primarily based on arithmetic operations involving both prime and composite numbers, where each type of number contributes significantly to the final scoring of potential word matches.

---

## Installation for Linux

1. Clone this repository:
    ```bash
    ~/$ git clone https://github.com/lucaselblanc/aap.git
    ```

2. Install rust:
    ```bash
    ~/$ sudo apt-get update
    ~/$ sudo apt-get upgrade
    ~/$ sudo apt-get install rustc
    ```

3. Compile the project:
    ```bash
    ~/$ cd aap
    ~/aap$ make
    ```

4. Run the program:
    ```bash
    ~/aap/target/release$ cargo run aap
    ```
