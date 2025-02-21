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

    let program_parsed_tree = parser::parse_program_from_file(filename).unwrap();
    let program = ast::make_program(&program_parsed_tree, interpreter::HARD_AST_INIT);

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

            let mut soft_program =
                ast::make_program(&program_parsed_tree, interpreter_soft::SOFT_AST_INIT);
            let stuff = ast::eval::eval_test_cases(
                &interpreter_soft::SoftEvaluator {
                    sigmoid_variance: 0.1,
                    equality_variance: 0.1,
                    sigma_list: 0.1,
                },
                &soft_program,
            );

            println!("test cases = {:?}", stuff);

            let grad: interpreter_soft::Gradient = stuff.into_iter().fold(
                interpreter_soft::make_oneshot(soft_program.num_ids, crate::ast::LitId(None)),
                |acc, x| acc + x.gradient,
            );

            interpreter_soft::apply_gradient_program(&mut soft_program, &grad);
            let _rehardened = ast_hardening::harden_ast(soft_program);
        }
    }
}
