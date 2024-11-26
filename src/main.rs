mod ast;
mod interpreter;
mod interpreter_soft;
mod parser;
mod type_checker;

fn main() {
    let program_parse_tree_result =
        parsel::parse_str(&std::fs::read_to_string("test.prog").unwrap());

    let program_parse_tree: parser::ProgramParseTree = match program_parse_tree_result {
        Ok(x) => x,
        Err(err) => panic!("bruh at {:?}: {:?}", err.span().start(), err),
    };

    let mut program_ast = ast::make_program(program_parse_tree);

    let val = interpreter::run_function(&program_ast, "pow2", vec![interpreter::Value::Int(3)]);
    eprintln!("val = {:?}", val);

    let soft_val = interpreter_soft::soft_run_function(
        &program_ast,
        "pow2",
        vec![interpreter_soft::ValueType::Int(3.0)],
    );

    eprintln!("soft val = {:?}", soft_val);

    eprintln!(
        "test cases = {:?}",
        interpreter::eval_test_cases(&program_ast)
    );

    for _ in 0..5 {
        let soft_cases = interpreter_soft::soft_eval_test_cases(&program_ast);

        eprintln!("soft test cases = {:?}", soft_cases);

        let average_grad = soft_cases.into_iter().fold(
            interpreter_soft::make_oneshot(program_ast.num_ids, crate::ast::LitId(None)),
            |acc, new| acc + new.1,
        );

        interpreter_soft::apply_gradient_program(&mut program_ast, &average_grad);
    }

    eprintln!(
        "test cases fixed maybe = {:?}",
        interpreter::eval_test_cases(&program_ast)
    );
}
