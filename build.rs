use std::env;
use std::fs;
use std::path::Path;

/* Probabilistic Abbreviation Algorithm (PAA) */
/* Written by lucasblanc*/

fn main() {

    let out_dir = env::var("OUT_DIR").unwrap();
    let target_dir = Path::new(&out_dir).parent().unwrap().parent().unwrap();

    let source = Path::new("src/dictionary[PT-BR].txt");
    let destination = target_dir.join("release").join("dictionary[PT-BR].txt");

    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent).unwrap();
    }

    fs::copy(source, &destination).unwrap();

    println!("cargo:rerun-if-changed=src/dictionary[PT-BR].txt");
}