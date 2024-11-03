# WangAnLang

## A language for approximating imperative programs as a neural nets

## Project Structure

The project is organized as follows:

```
.
├── modules
│   ├── parsetree.py            # Python script for parsing tree generation
│   └── type_inference.py       # Python script for type inference
├── proposal
│   ├── build.sh                # Shell script for building the proposal
│   ├── proposal.tex            # LaTeX source file for the proposal document
│   └── refs.bib                # Bibliography file for LaTeX references
├── src
│   ├── json_generator          # Directory for JSON generation module
│   ├── parser                  # Directory for parser module
│   └── type_checker            # Directory for type checker module
├── main.rs                     # Main Rust source file
├── .gitignore                  # Git ignore file
├── Cargo.lock                  # Cargo lock file for Rust dependencies
├── Cargo.toml                  # Cargo configuration file for Rust dependencies
├── main.py                     # Main Python script 
├── README.md                   # Project README file
└── test.prog                   # Test program file
```

## Description

- **modules/**: Contains Python scripts for parsing and type inference.
- **proposal/**: Contains files related to the proposal document, including LaTeX source, bibliography, and build script.
- **src/**: Contains subdirectories for modules handling JSON generation, parsing, and type checking.
- **main.rs**: The main source file for the Rust program.
- **main.py**: The main Python script.
- **Cargo.toml** and **Cargo.lock**: Rust package manager files for dependencies.

## Build Instructions

1. To build the proposal, navigate to the `proposal` directory and run:
   ```bash
   ./build.sh
   ```

2. For Rust code, use Cargo:
   ```bash
   cargo build
   ```

3. For Python scripts, ensure all dependencies are installed, then execute:
   ```bash
   python main.py
   ```

## Dependencies

- **Python**: List dependencies (e.g., `requirements.txt` if available).
- **Rust**: Dependencies are managed by Cargo.
- **LaTeX**: Required for compiling the proposal document.