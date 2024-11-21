TARGET = aap

SRC = src/main.rs

DICTIONARY = src/dictionary[PT-BR].txt

CARGO = cargo

all:
	$(CARGO) build --release --bin main

run: all
	$(CARGO) run --bin main -- $(DICTIONARY)

clean:
	$(CARGO) clean

test-doc:
	$(CARGO) test --doc

test: all
	$(CARGO) test

.PHONY: all run clean test-doc test