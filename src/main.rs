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

    let program_ast = ast::make_program(program_parse_tree);

    let val = interpreter::run_function(&program_ast, "test", vec![interpreter::Value::Int(3)]);
    println!("val = {:?}", val);
    let soft_val = interpreter_soft::soft_run_function(
        &program_ast,
        "test",
        vec![interpreter_soft::ValueType::Int(3.0)],
    );

    println!("soft val = {:?}", soft_val);

    println!(
        "test cases = {:?}",
        interpreter::eval_test_cases(&program_ast)
    );
    println!(
        "hard test cases = {:?}",
        interpreter_soft::soft_eval_test_cases(&program_ast)
    );
}
