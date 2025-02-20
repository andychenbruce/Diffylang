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
    let program_parsed_tree = parser::parse_program(trees).unwrap();
    let program = ast::make_program(program_parsed_tree, interpreter::HARD_AST_INIT);

    match args.get(2) {
        Some(func_name) => {
            let func_args: Vec<ast::eval::EvalVal<i64, f64, bool>> = args[3..]
                .iter()
                .map(|x| {
                    if x.contains('.') {
                        ast::eval::EvalVal::Float(x.parse::<f64>().unwrap())
                    } else {
                        ast::eval::EvalVal::Int(x.parse::<i64>().unwrap())
                    }
                })
                .collect();

            let output = ast::eval::run_function(
                &interpreter::HardEvaluator {},
                &program,
                func_name,
                func_args,
            );

            println!("output = {:?}", output);
        }
        None => {
            let stuff = ast::eval::eval_test_cases(&interpreter::HardEvaluator {}, &program);

            println!("test cases = {:?}", stuff);
        }
    }
}
