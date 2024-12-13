mod ast;
mod ast_hardening;
mod interpreter;
mod interpreter_soft;
mod parser;

const LEARNING_RATE: f64 = 0.5;
const NUM_ITERS: i32 = 100;
const VARIANCE_MIN: f64 = 0.0001;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let filename = &args
        .get(1)
        .unwrap_or_else(|| panic!("usage: {} FILENAME [FUNC NAME] [ARGS ...]", args[0]));

    let program_parse_tree_result = parsel::parse_str(&std::fs::read_to_string(filename).unwrap());

    let program_parse_tree: parser::ProgramParseTree = match program_parse_tree_result {
        Ok(x) => x,
        Err(err) => panic!(
            "error parsing program at {:?}: {:?}",
            err.span().start(),
            err
        ),
    };

    let hard_program_ast =
        ast::make_program(program_parse_tree.clone(), interpreter::HARD_AST_INIT);
    let mut soft_program_ast =
        ast::make_program(program_parse_tree, interpreter_soft::SOFT_AST_INIT);

    let hard_evaluator = interpreter::HardEvaluator {};

    if let Some(func_name) = args.get(2) {
        let soft_evaluator = interpreter_soft::SoftEvaluator {
            sigmoid_variance: VARIANCE_MIN,
            equality_variance: VARIANCE_MIN,
            sigma_list: VARIANCE_MIN,
        };

        let func_args: Vec<ast::eval::EvalVal<i64, f64, bool, i64>> = args[3..]
            .iter()
            .map(|x| {
                if x.contains('.') {
                    ast::eval::EvalVal::Float(x.parse::<f64>().unwrap())
                } else {
                    ast::eval::EvalVal::Int(x.parse::<i64>().unwrap())
                }
            })
            .collect();

        let val = ast::eval::run_function::<_, _, _, _, interpreter::HardEvaluator>(
            &hard_evaluator,
            &hard_program_ast,
            func_name,
            func_args.clone(),
        );
        eprintln!("val = {:?}", val);

        let soft_args: Vec<interpreter_soft::SoftValue> = func_args
            .iter()
            .map(|arg| match arg {
                ast::eval::EvalVal::Int(x) => interpreter_soft::SoftValue::Int(
                    interpreter_soft::make_int(*x, ast::LitId(None), soft_program_ast.num_ids),
                ),
                ast::eval::EvalVal::Float(x) => interpreter_soft::SoftValue::Float(
                    interpreter_soft::make_float(*x, ast::LitId(None), soft_program_ast.num_ids),
                ),
                _ => unreachable!(),
            })
            .collect();

        let soft_val = ast::eval::run_function::<_, _, _, _, interpreter_soft::SoftEvaluator>(
            &soft_evaluator,
            &soft_program_ast,
            func_name,
            soft_args,
        );

        eprintln!("soft val = {:?}", soft_val);
    } else {
        eprintln!(
            "test cases = {:?}",
            ast::eval::eval_test_cases::<_, _, _, _, interpreter::HardEvaluator>(
                &hard_evaluator,
                &hard_program_ast
            )
        );

        for i in 0..NUM_ITERS {
            let equality_variance = f64::max((-(i as f64) / 300.0).exp() * 2.0, VARIANCE_MIN);
            let sigmoid_variance = f64::max((-(i as f64) / 300.0).exp() * 1.0, VARIANCE_MIN);
            let sigma_list = f64::max((-(i as f64) / 300.0).exp() * 1.0, VARIANCE_MIN);

            let soft_evaluator = interpreter_soft::SoftEvaluator {
                sigmoid_variance,
                equality_variance,
                sigma_list,
            };

            let soft_cases =
                ast::eval::eval_test_cases::<_, _, _, _, interpreter_soft::SoftEvaluator>(
                    &soft_evaluator,
                    &soft_program_ast,
                );

            eprintln!("soft test cases = {:?}", soft_cases);

            let average_grad = soft_cases.into_iter().fold(
                interpreter_soft::make_oneshot(soft_program_ast.num_ids, crate::ast::LitId(None)),
                |acc, new| {
                    acc + match new {
                        ast::eval::EvalVal::Bool(x) => x.gradient,
                        _ => unreachable!(),
                    }
                },
            );

            // println!("average grad = {:?}", average_grad);

            interpreter_soft::apply_gradient_program(
                &mut soft_program_ast,
                &(average_grad * LEARNING_RATE),
            );
        }

        println!(
            "final soft ast = {}",
            serde_json::to_string_pretty(&soft_program_ast).unwrap()
        );
        let hardened_ast = ast_hardening::harden_ast(soft_program_ast.clone());

        eprintln!(
            "test cases fixed maybe = {:?}",
            ast::eval::eval_test_cases::<_, _, _, _, interpreter::HardEvaluator>(
                &hard_evaluator,
                &hardened_ast
            )
        );
    }
}
