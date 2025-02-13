mod ast;
mod ast_hardening;
mod interpreter;
mod interpreter_soft;
mod parser;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let filename = &args
        .get(1)
        .unwrap_or_else(|| panic!("usage: {} FILENAME [FUNC NAME] [ARGS ...]", args[0]));

    let code = &std::fs::read_to_string(filename).unwrap();
    let mut tokens = parser::Tokens::new(code).peekable();
    let trees = parser::make_token_trees(&mut tokens).unwrap();
    let program = parser::parse_program(trees);
    
    println!("tokens = {:?}", program);
}
